use rand::{rngs::StdRng, RngExt};
use serde_json::json;

use crate::agents::ModelSnapshot;
use crate::config::Config;
use crate::nn::{Adam, Network, build_layer_sizes, build_activations, softmax, normalize, compute_returns};

// ─────────────────────────────────────────────────────────
// REINFORCE agent (Monte-Carlo policy gradient)
// ─────────────────────────────────────────────────────────

pub struct ReinforceAgent {
    network:     Network,
    optimizer:   Adam,
    gamma:       f64,
    normalize:   bool,
    baseline:    String,      // "none" | "mean"
    env_name:    String,
    state_size:  usize,
    action_size: usize,
}

impl ReinforceAgent {
    pub fn new(
        state_size: usize,
        action_size: usize,
        cfg: &Config,
        rng: &mut StdRng,
    ) -> Self {
        let layer_sizes = build_layer_sizes(state_size, &cfg.network.hidden_layers, action_size);
        let activations = build_activations(&cfg.network.activation, layer_sizes.len() - 1);
        let network    = Network::new(&layer_sizes, &activations, rng);
        let optimizer  = Adam::new(&network, cfg.algorithm.learning_rate);

        ReinforceAgent {
            network,
            optimizer,
            gamma:       cfg.algorithm.gamma,
            normalize:   cfg.algorithm.reinforce.normalize_returns,
            baseline:    cfg.algorithm.reinforce.baseline.clone(),
            env_name:    cfg.environment.name.clone(),
            state_size,
            action_size,
        }
    }

    /// Sample action from the current policy distribution
    pub fn select_action(&mut self, state: &[f64], rng: &mut impl rand::Rng) -> (usize, f64) {
        let logits = self.network.forward_no_grad(state);
        let probs  = softmax(&logits);

        // Sample proportional to probability
        let u: f64 = rng.random();
        let mut cumulative = 0.0;
        let mut action = self.action_size - 1;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if u <= cumulative {
                action = i;
                break;
            }
        }
        let log_prob = probs[action].max(1e-10).ln();
        (action, log_prob)
    }

    /// Greedy action for evaluation
    pub fn greedy_action(&self, state: &[f64]) -> usize {
        let logits = self.network.forward_no_grad(state);
        let probs  = softmax(&logits);
        probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }

    /// Run one gradient update over a completed episode trajectory.
    /// `states`, `actions`, `rewards` are parallel arrays from one episode.
    pub fn update(
        &mut self,
        states:  &[Vec<f64>],
        actions: &[usize],
        rewards: &[f64],
    ) -> f64 {
        let mut returns = compute_returns(rewards, self.gamma);

        // Baseline subtraction
        if self.baseline == "mean" {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            for r in returns.iter_mut() { *r -= mean; }
        }

        // Normalise
        if self.normalize && returns.len() > 1 {
            returns = normalize(&returns);
        }

        let n = states.len();
        let mut acc_grads = self.network.zero_grads();
        let mut total_loss = 0.0_f64;

        for t in 0..n {
            // Forward pass to get current logits (needed for backprop)
            let logits  = self.network.forward(&states[t]);
            let probs   = softmax(&logits);
            let log_prob = probs[actions[t]].max(1e-10).ln();

            // Loss = -G_t * log π(a|s)
            let g = returns[t];
            total_loss += -g * log_prob;

            // Gradient of -G * log π(a|s) w.r.t. logits (via softmax):
            //   ∂L/∂z_j = G * (π_j - 𝟙[j==a])
            let grad_out: Vec<f64> = probs.iter().enumerate().map(|(j, &p)| {
                g * (p - if j == actions[t] { 1.0 } else { 0.0 })
            }).collect();

            let grads = self.network.backward(&grad_out);
            Network::add_grads(&mut acc_grads, &grads);
        }

        // Average and apply
        Network::scale_grads(&mut acc_grads, 1.0 / n as f64);
        self.optimizer.step(&mut self.network, &acc_grads);

        total_loss / n as f64
    }

    pub fn save(&self, path: &str, episodes: usize, best_avg: f64) -> anyhow::Result<()> {
        let snapshot = ModelSnapshot {
            algorithm:         "reinforce".to_string(),
            environment:       self.env_name.clone(),
            state_size:        self.state_size,
            action_size:       self.action_size,
            training_episodes: episodes,
            best_avg_reward:   best_avg,
            policy_network:    self.network.clone(),
            value_network:     None,
            metadata:          json!({ "baseline": self.baseline }),
        };
        snapshot.save(path)
    }
}
