use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────
// Top-level config
// ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub environment: EnvironmentConfig,
    pub network: NetworkConfig,
    pub algorithm: AlgorithmConfig,
    pub training: TrainingConfig,
    pub output: OutputConfig,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        let valid_envs = ["cartpole", "mountain_car", "gridworld"];
        if !valid_envs.contains(&self.environment.name.as_str()) {
            bail!(
                "Unknown environment '{}'. Valid options: {}",
                self.environment.name,
                valid_envs.join(", ")
            );
        }
        let valid_algos = ["dqn", "reinforce", "ppo"];
        if !valid_algos.contains(&self.algorithm.name.as_str()) {
            bail!(
                "Unknown algorithm '{}'. Valid options: {}",
                self.algorithm.name,
                valid_algos.join(", ")
            );
        }
        let valid_acts = ["relu", "tanh", "sigmoid", "linear"];
        if !valid_acts.contains(&self.network.activation.as_str()) {
            bail!(
                "Unknown activation '{}'. Valid options: {}",
                self.network.activation,
                valid_acts.join(", ")
            );
        }
        if self.network.hidden_layers.is_empty() {
            bail!("network.hidden_layers must have at least one element");
        }
        if self.algorithm.gamma <= 0.0 || self.algorithm.gamma > 1.0 {
            bail!("algorithm.gamma must be in (0, 1]");
        }
        if self.algorithm.learning_rate <= 0.0 {
            bail!("algorithm.learning_rate must be positive");
        }
        if self.training.episodes == 0 {
            bail!("training.episodes must be > 0");
        }
        if self.environment.max_steps == 0 {
            bail!("environment.max_steps must be > 0");
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────
// Sub-configs
// ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    /// One of: cartpole, mountain_car, gridworld
    pub name: String,
    /// Maximum steps per episode
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    /// Random seed for reproducibility
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// GridWorld specific: grid dimensions (e.g. [5, 5])
    #[serde(default = "default_grid_size")]
    pub grid_size: [usize; 2],
    /// GridWorld specific: obstacle fraction
    #[serde(default = "default_obstacle_frac")]
    pub obstacle_fraction: f64,
}

fn default_max_steps() -> usize { 500 }
fn default_seed() -> u64 { 42 }
fn default_grid_size() -> [usize; 2] { [5, 5] }
fn default_obstacle_frac() -> f64 { 0.1 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// List of hidden layer sizes, e.g. [128, 64]
    pub hidden_layers: Vec<usize>,
    /// Hidden layer activation: relu | tanh | sigmoid | linear
    #[serde(default = "default_activation")]
    pub activation: String,
}

fn default_activation() -> String { "relu".to_string() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    /// One of: dqn, reinforce, ppo
    pub name: String,
    /// Discount factor γ
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    /// Learning rate for optimizer
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    /// Optimizer: adam | sgd
    #[serde(default = "default_optimizer")]
    pub optimizer: String,

    // Algorithm-specific sub-sections
    #[serde(default)]
    pub dqn: DqnConfig,
    #[serde(default)]
    pub ppo: PpoConfig,
    #[serde(default)]
    pub reinforce: ReinforceConfig,
}

fn default_gamma() -> f64 { 0.99 }
fn default_lr() -> f64 { 1e-3 }
fn default_optimizer() -> String { "adam".to_string() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DqnConfig {
    /// Starting exploration rate
    #[serde(default = "dqn_eps_start")]
    pub epsilon_start: f64,
    /// Minimum exploration rate
    #[serde(default = "dqn_eps_end")]
    pub epsilon_end: f64,
    /// Epsilon decay multiplier per step
    #[serde(default = "dqn_eps_decay")]
    pub epsilon_decay: f64,
    /// Replay buffer capacity
    #[serde(default = "dqn_buf")]
    pub buffer_size: usize,
    /// Mini-batch size for updates
    #[serde(default = "dqn_batch")]
    pub batch_size: usize,
    /// Frequency (steps) to sync target network
    #[serde(default = "dqn_target_freq")]
    pub target_update_freq: usize,
    /// Steps before training starts
    #[serde(default = "dqn_warmup")]
    pub warmup_steps: usize,
}

fn dqn_eps_start() -> f64 { 1.0 }
fn dqn_eps_end() -> f64 { 0.01 }
fn dqn_eps_decay() -> f64 { 0.995 }
fn dqn_buf() -> usize { 10_000 }
fn dqn_batch() -> usize { 64 }
fn dqn_target_freq() -> usize { 100 }
fn dqn_warmup() -> usize { 500 }

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            epsilon_start: dqn_eps_start(),
            epsilon_end: dqn_eps_end(),
            epsilon_decay: dqn_eps_decay(),
            buffer_size: dqn_buf(),
            batch_size: dqn_batch(),
            target_update_freq: dqn_target_freq(),
            warmup_steps: dqn_warmup(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpoConfig {
    /// PPO clipping parameter ε
    #[serde(default = "ppo_clip")]
    pub clip_epsilon: f64,
    /// Number of update epochs per batch
    #[serde(default = "ppo_epochs")]
    pub epochs: usize,
    /// Mini-batch size
    #[serde(default = "ppo_batch")]
    pub batch_size: usize,
    /// Steps to collect per update
    #[serde(default = "ppo_steps")]
    pub steps_per_update: usize,
    /// Learning rate for optimizer
    #[serde(default = "ppo_lr")]
    pub learning_rate: f64,
    /// GAE lambda
    #[serde(default = "ppo_lambda")]
    pub gae_lambda: f64,
    /// Entropy bonus coefficient
    #[serde(default = "ppo_entropy")]
    pub entropy_coef: f64,
    /// Value function loss coefficient
    #[serde(default = "ppo_vf")]
    pub value_coef: f64,

}

fn ppo_clip() -> f64 { 0.2 }
fn ppo_epochs() -> usize { 4 }
fn ppo_batch() -> usize { 64 }
fn ppo_steps() -> usize { 512 }
fn ppo_lr() -> f64 { 1e-3 }
fn ppo_lambda() -> f64 { 0.95 }
fn ppo_entropy() -> f64 { 0.01 }
fn ppo_vf() -> f64 { 0.5 }

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_epsilon: ppo_clip(),
            epochs: ppo_epochs(),
            batch_size: ppo_batch(),
            steps_per_update: ppo_steps(),
            learning_rate: ppo_lr(),
            gae_lambda: ppo_lambda(),
            entropy_coef: ppo_entropy(),
            value_coef: ppo_vf(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReinforceConfig {
    /// Normalize returns for stability
    #[serde(default = "rf_normalize")]
    pub normalize_returns: bool,
    /// Optional baseline (none | mean)
    #[serde(default = "rf_baseline")]
    pub baseline: String,
}

fn rf_normalize() -> bool { true }
fn rf_baseline() -> String { "mean".to_string() }

impl Default for ReinforceConfig {
    fn default() -> Self {
        Self {
            normalize_returns: rf_normalize(),
            baseline: rf_baseline(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Total number of training episodes
    pub episodes: usize,
    /// Log metrics every N episodes
    #[serde(default = "tr_log_interval")]
    pub log_interval: usize,
    /// Evaluate every N episodes
    #[serde(default = "tr_eval_interval")]
    pub eval_interval: usize,
    /// Number of evaluation episodes
    #[serde(default = "tr_eval_eps")]
    pub eval_episodes: usize,
    /// Save a checkpoint when a new best is found
    #[serde(default = "tr_checkpoint")]
    pub save_best: bool,
    /// Target reward — stop early if achieved
    #[serde(default)]
    pub target_reward: Option<f64>,
}

fn tr_log_interval() -> usize { 10 }
fn tr_eval_interval() -> usize { 100 }
fn tr_eval_eps() -> usize { 5 }
fn tr_checkpoint() -> bool { true }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Directory / base path to save the model
    pub model_path: String,
    /// Format: json (default)
    #[serde(default = "out_format")]
    pub format: String,
    /// Print training curve at the end
    #[serde(default = "out_curve")]
    pub print_curve: bool,
}

fn out_format() -> String { "json".to_string() }
fn out_curve() -> bool { true }

// ─────────────────────────────────────────────────────────
// Sample config generator
// ─────────────────────────────────────────────────────────

pub fn generate_sample_config(algorithm: &str, env: &str) -> String {
    let (episodes, hidden, max_steps) = match env {
        "mountain_car" => (500, "[128, 128]", 200),
        "gridworld"    => (300, "[64, 64]", 200),
        _              => (600, "[128, 64]", 500),
    };

    match algorithm {
        "ppo" => format!(
r#"[environment]
name          = "{env}"
max_steps     = {max_steps}
seed          = 42

[network]
hidden_layers = {hidden}
activation    = "tanh"

[algorithm]
name          = "ppo"
gamma         = 0.99
learning_rate = 3e-4
optimizer     = "adam"

[algorithm.ppo]
clip_epsilon    = 0.2
epochs          = 4
batch_size      = 64
steps_per_update = 512
gae_lambda      = 0.95
entropy_coef    = 0.01
value_coef      = 0.5

[training]
episodes       = {episodes}
log_interval   = 10
eval_interval  = 50
eval_episodes  = 5
save_best      = true

[output]
model_path  = "./models/{env}_ppo"
format      = "json"
print_curve = true
"#),
        "reinforce" => format!(
r#"[environment]
name          = "{env}"
max_steps     = {max_steps}
seed          = 42

[network]
hidden_layers = {hidden}
activation    = "relu"

[algorithm]
name          = "reinforce"
gamma         = 0.99
learning_rate = 1e-3
optimizer     = "adam"

[algorithm.reinforce]
normalize_returns = true
baseline          = "mean"

[training]
episodes       = {episodes}
log_interval   = 10
eval_interval  = 50
eval_episodes  = 5
save_best      = true

[output]
model_path  = "./models/{env}_reinforce"
format      = "json"
print_curve = true
"#),
        _ => format!(   // dqn default
r#"[environment]
name          = "{env}"
max_steps     = {max_steps}
seed          = 42

[network]
hidden_layers = {hidden}
activation    = "relu"

[algorithm]
name          = "dqn"
gamma         = 0.99
learning_rate = 1e-3
optimizer     = "adam"

[algorithm.dqn]
epsilon_start    = 1.0
epsilon_end      = 0.01
epsilon_decay    = 0.995
buffer_size      = 10000
batch_size       = 64
target_update_freq = 100
warmup_steps     = 500

[training]
episodes       = {episodes}
log_interval   = 10
eval_interval  = 50
eval_episodes  = 5
save_best      = true

[output]
model_path  = "./models/{env}_dqn"
format      = "json"
print_curve = true
"#),
    }
}
