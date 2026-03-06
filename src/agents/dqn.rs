use std::collections::VecDeque;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng, Rng};
use serde_json::json;

use crate::agents::ModelSnapshot;
use crate::config::Config;
use crate::nn::{Adam, Network, build_layer_sizes, build_activations};

// ─────────────────────────────────────────────────────────
// Experience replay buffer
// ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct Experience {
    state:      Vec<f64>,
    action:     usize,
    reward:     f64,
    next_state: Vec<f64>,
    done:       bool,
}

struct ReplayBuffer {
    buf:      VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        ReplayBuffer { buf: VecDeque::with_capacity(capacity), capacity }
    }

    fn push(&mut self, exp: Experience) {
        if self.buf.len() >= self.capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(exp);
    }

    fn sample(&self, n: usize, rng: &mut impl Rng) -> Vec<Experience> {
        let len = self.buf.len();
        (0..n).map(|_| self.buf[rng.random_range(0..len)].clone()).collect()
    }

    fn len(&self) -> usize { self.buf.len() }
}

// ─────────────────────────────────────────────────────────
// DQN Agent
// ─────────────────────────────────────────────────────────

pub struct DqnAgent {
    online_net:  Network,
    target_net:  Network,
    optimizer:   Adam,
    buffer:      ReplayBuffer,
    pub epsilon: f64,
    epsilon_end: f64,
    epsilon_decay: f64,
    gamma:       f64,
    batch_size:  usize,
    target_freq: usize,
    warmup:      usize,
    pub steps:   usize,
    rng:         StdRng,
    env_name:    String,
    state_size:  usize,
    action_size: usize,
}

impl DqnAgent {
    pub fn new(
        state_size: usize,
        action_size: usize,
        cfg: &Config,
        rng: &mut StdRng,
    ) -> Self {
        let layer_sizes = build_layer_sizes(state_size, &cfg.network.hidden_layers, action_size);
        let activations = build_activations(&cfg.network.activation, layer_sizes.len() - 1);

        let online_net = Network::new(&layer_sizes, &activations, rng);
        let target_net = online_net.clone();
        let optimizer  = Adam::new(&online_net, cfg.algorithm.learning_rate);

        let dc = &cfg.algorithm.dqn;
        DqnAgent {
            online_net,
            target_net,
            optimizer,
            buffer:        ReplayBuffer::new(dc.buffer_size),
            epsilon:       dc.epsilon_start,
            epsilon_end:   dc.epsilon_end,
            epsilon_decay: dc.epsilon_decay,
            gamma:         cfg.algorithm.gamma,
            batch_size:    dc.batch_size,
            target_freq:   dc.target_update_freq,
            warmup:        dc.warmup_steps,
            steps:         0,
            rng:           StdRng::seed_from_u64(cfg.environment.seed + 1),
            env_name:      cfg.environment.name.clone(),
            state_size,
            action_size,
        }
    }

    /// ε-greedy action selection
    pub fn select_action(&mut self, state: &[f64], training: bool) -> usize {
        if training && self.rng.random::<f64>() < self.epsilon {
            self.rng.random_range(0..self.action_size)
        } else {
            let q = self.online_net.forward_no_grad(state);
            q.iter()
             .enumerate()
             .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
             .map(|(i, _)| i)
             .unwrap_or(0)
        }
    }

    /// Store a transition and decay epsilon
    pub fn store_transition(
        &mut self,
        state: &[f64],
        action: usize,
        reward: f64,
        next_state: &[f64],
        done: bool,
    ) {
        self.buffer.push(Experience {
            state: state.to_vec(),
            action,
            reward,
            next_state: next_state.to_vec(),
            done,
        });
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_end);
        self.steps  += 1;

        if self.steps % self.target_freq == 0 {
            self.target_net.copy_weights_from(&self.online_net);
        }
    }

    /// Sample a mini-batch and do one gradient step.
    /// Returns average TD-error loss, or None if not ready yet.
    pub fn update(&mut self) -> Option<f64> {
        if self.buffer.len() < self.warmup.max(self.batch_size) {
            return None;
        }

        let batch = self.buffer.sample(self.batch_size, &mut self.rng);
        let mut acc_grads  = self.online_net.zero_grads();
        let mut total_loss = 0.0_f64;

        for exp in &batch {
            // --- Target Q-value (Bellman) ---
            let q_next   = self.target_net.forward_no_grad(&exp.next_state);
            let max_next = if exp.done { 0.0 } else {
                q_next.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            };
            let target = exp.reward + self.gamma * max_next;

            // --- Predicted Q-values (with gradient tracking) ---
            let q_pred = self.online_net.forward(&exp.state);
            let td_err = q_pred[exp.action] - target;
            total_loss += td_err * td_err;

            // Gradient only for the taken action (MSE)
            let mut grad_out = vec![0.0; self.action_size];
            grad_out[exp.action] = 2.0 * td_err / self.batch_size as f64;

            let grads = self.online_net.backward(&grad_out);
            Network::add_grads(&mut acc_grads, &grads);
        }

        self.optimizer.step(&mut self.online_net, &acc_grads);
        Some(total_loss / self.batch_size as f64)
    }

    /// Greedy evaluation (no exploration)
    pub fn evaluate(&mut self, state: &[f64]) -> usize {
        let q = self.online_net.forward_no_grad(state);
        q.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }

    pub fn save(&self, path: &str, episodes: usize, best_avg: f64) -> anyhow::Result<()> {
        let snapshot = ModelSnapshot {
            algorithm:         "dqn".to_string(),
            environment:       self.env_name.clone(),
            state_size:        self.state_size,
            action_size:       self.action_size,
            training_episodes: episodes,
            best_avg_reward:   best_avg,
            policy_network:    self.online_net.clone(),
            value_network:     None,
            metadata: json!({
                "epsilon_final": self.epsilon,
                "total_steps":   self.steps,
                "buffer_size":   self.buffer.len(),
            }),
        };
        snapshot.save(path)
    }
}
