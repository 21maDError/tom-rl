use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde_json::json;

use crate::agents::{ModelSnapshot, Transition};
use crate::config::Config;
use crate::nn::{Adam, Network, build_layer_sizes, build_activations, softmax, normalize};

// ─────────────────────────────────────────────────────────
// PPO Agent  (actor-critic, clipped surrogate, GAE)
// ─────────────────────────────────────────────────────────

pub struct PpoAgent {
    actor:        Network,
    critic:       Network,
    actor_opt:    Adam,
    critic_opt:   Adam,
    gamma:        f64,
    gae_lambda:   f64,
    clip_eps:     f64,
    epochs:       usize,
    batch_size:   usize,
    entropy_coef: f64,
    value_coef:   f64,
    env_name:     String,
    state_size:   usize,
    action_size:  usize,
    rng:          StdRng,
}

impl PpoAgent {
    pub fn new(
        state_size: usize,
        action_size: usize,
        cfg: &Config,
        rng: &mut StdRng,
    ) -> Self {
        // Actor: state -> action logits
        let actor_sizes = build_layer_sizes(state_size, &cfg.network.hidden_layers, action_size);
        let actor_acts  = build_activations(&cfg.network.activation, actor_sizes.len() - 1);
        let actor       = Network::new(&actor_sizes, &actor_acts, rng);

        // Critic: state -> scalar value
        let critic_sizes = build_layer_sizes(state_size, &cfg.network.hidden_layers, 1);
        let critic_acts  = build_activations(&cfg.network.activation, critic_sizes.len() - 1);
        let critic       = Network::new(&critic_sizes, &critic_acts, rng);

        let actor_opt  = Adam::new(&actor, cfg.algorithm.learning_rate);
        let critic_opt = Adam::new(&critic, cfg.algorithm.learning_rate);

        let pc = &cfg.algorithm.ppo;
        PpoAgent {
            actor, critic, actor_opt, critic_opt,
            gamma:        cfg.algorithm.gamma,
            gae_lambda:   pc.gae_lambda,
            clip_eps:     pc.clip_epsilon,
            epochs:       pc.epochs,
            batch_size:   pc.batch_size,
            entropy_coef: pc.entropy_coef,
            value_coef:   pc.value_coef,
            env_name:     cfg.environment.name.clone(),
            state_size,
            action_size,
            rng:          StdRng::seed_from_u64(cfg.environment.seed + 99),
        }
    }

    /// Sample action from the current policy; also returns value estimate and log-prob.
    pub fn select_action(&mut self, state: &[f64]) -> (usize, f64, f64) {
        let logits = self.actor.forward_no_grad(state);
        let probs  = softmax(&logits);
        let value  = self.critic.forward_no_grad(state)[0];

        // Stochastic sampling
        let u: f64 = self.rng.random();
        let mut cumulative = 0.0;
        let mut action = self.action_size - 1;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if u <= cumulative { action = i; break; }
        }
        let log_prob = probs[action].max(1e-10).ln();
        (action, log_prob, value)
    }

    /// Greedy action for evaluation
    pub fn greedy_action(&self, state: &[f64]) -> usize {
        let logits = self.actor.forward_no_grad(state);
        let probs  = softmax(&logits);
        probs.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }

    /// Critic value estimate (for bootstrapping, no stochastic sampling needed)
    pub fn value_estimate(&self, state: &[f64]) -> f64 {
        self.critic.forward_no_grad(state)[0]
    }

    /// Compute GAE advantages and value targets from a collected trajectory.
    /// `last_value`: V(s_T) — 0 if terminal, else critic estimate.
    fn compute_gae(
        &self,
        transitions: &[Transition],
        last_value: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = transitions.len();
        let mut advantages = vec![0.0_f64; n];
        let mut returns    = vec![0.0_f64; n];

        let mut gae  = 0.0_f64;
        let mut v_next = last_value;

        for i in (0..n).rev() {
            let t   = &transitions[i];
            let v   = t.value;
            let r   = t.reward;
            let mask = if t.done { 0.0 } else { 1.0 };

            let delta = r + self.gamma * v_next * mask - v;
            gae       = delta + self.gamma * self.gae_lambda * mask * gae;
            advantages[i] = gae;
            returns[i]    = gae + v;
            v_next        = v;
        }

        (advantages, returns)
    }

    /// Full PPO update over a collected batch of transitions.
    /// Returns (mean_actor_loss, mean_critic_loss).
    pub fn update(&mut self, transitions: &[Transition], last_value: f64) -> (f64, f64) {
        let (mut advantages, returns) = self.compute_gae(transitions, last_value);
        // Normalise advantages
        if advantages.len() > 1 {
            advantages = normalize(&advantages);
        }

        let n = transitions.len();
        let mut total_actor_loss  = 0.0_f64;
        let mut total_critic_loss = 0.0_f64;

        for _epoch in 0..self.epochs {
            // Shuffle indices
            let mut indices: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = self.rng.random_range(0..=i);
                indices.swap(i, j);
            }

            for chunk in indices.chunks(self.batch_size) {
                let mut actor_grads  = self.actor.zero_grads();
                let mut critic_grads = self.critic.zero_grads();
                let bs = chunk.len() as f64;

                for &idx in chunk {
                    let t = &transitions[idx];
                    let adv = advantages[idx];
                    let ret = returns[idx];

                    // ── Critic update (MSE on value targets) ───────────────
                    let v_pred = self.critic.forward(&t.state)[0];
                    let v_err  = v_pred - ret;
                    total_critic_loss += v_err * v_err;

                    // dL_critic/dV_pred = 2 * v_err / bs
                    let grad_v = vec![2.0 * v_err / bs * self.value_coef];
                    let cg     = self.critic.backward(&grad_v);
                    Network::add_grads(&mut critic_grads, &cg);

                    // ── Actor update (PPO clipped surrogate) ───────────────
                    let logits_new = self.actor.forward(&t.state);
                    let probs_new  = softmax(&logits_new);
                    let log_new    = probs_new[t.action].max(1e-10).ln();
                    let log_old    = t.log_prob;

                    let ratio   = (log_new - log_old).exp();
                    let clip_r  = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps);

                    let surr1 = ratio  * adv;
                    let surr2 = clip_r * adv;
                    let loss_clip = -surr1.min(surr2);   // we minimise

                    // Entropy bonus: H = -Σ p*log(p)
                    let entropy: f64 = probs_new.iter().map(|&p| -p * p.max(1e-10).ln()).sum();
                    let loss_actor = loss_clip - self.entropy_coef * entropy;
                    total_actor_loss += loss_actor;

                    // Gradient of (loss_clip - entropy_coef * entropy) w.r.t. logits:
                    // If not clipped: ∂(-ratio*adv)/∂z_j = -adv * ratio * (δ_{aj} - π_j)
                    // Entropy gradient: ∂(-H)/∂z_j = π_j*(log(π_j)+1) - Σ π_k*log(π_k)*π_j ≈ π_j*(log(π_j) - H)
                    let is_clipped = (adv >= 0.0 && ratio > 1.0 + self.clip_eps)
                                  || (adv <  0.0 && ratio < 1.0 - self.clip_eps);

                    let grad_clip: Vec<f64> = if is_clipped {
                        vec![0.0; self.action_size]
                    } else {
                        probs_new.iter().enumerate().map(|(j, &p)| {
                            let ind = if j == t.action { 1.0 } else { 0.0 };
                            -adv * ratio * (ind - p) / bs
                        }).collect()
                    };

                    let log_probs: Vec<f64> = probs_new.iter().map(|&p| p.max(1e-10).ln()).collect();
                    let h_val: f64          = log_probs.iter().zip(probs_new.iter()).map(|(lp,&p)| -p*lp).sum();
                    let grad_entropy: Vec<f64> = probs_new.iter().zip(log_probs.iter()).map(|(&p, &lp)| {
                        // ∂H/∂z_j = π_j*(H - log(π_j) - 1) ... gradient of entropy w.r.t. logits
                        p * (h_val - lp - 1.0) / bs
                    }).collect();

                    let grad_out: Vec<f64> = grad_clip.iter().zip(grad_entropy.iter())
                        .map(|(gc, ge)| gc - self.entropy_coef * ge)
                        .collect();

                    let ag = self.actor.backward(&grad_out);
                    Network::add_grads(&mut actor_grads, &ag);
                }

                self.actor_opt.step(&mut self.actor, &actor_grads);
                self.critic_opt.step(&mut self.critic, &critic_grads);
            }
        }

        let n_updates = (self.epochs * n) as f64;
        (total_actor_loss / n_updates, total_critic_loss / n_updates)
    }

    pub fn save(&self, path: &str, episodes: usize, best_avg: f64) -> anyhow::Result<()> {
        let snapshot = ModelSnapshot {
            algorithm:         "ppo".to_string(),
            environment:       self.env_name.clone(),
            state_size:        self.state_size,
            action_size:       self.action_size,
            training_episodes: episodes,
            best_avg_reward:   best_avg,
            policy_network:    self.actor.clone(),
            value_network:     Some(self.critic.clone()),
            metadata: json!({
                "clip_eps":   self.clip_eps,
                "epochs":     self.epochs,
                "gae_lambda": self.gae_lambda,
            }),
        };
        snapshot.save(path)
    }
}
