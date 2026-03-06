pub mod dqn;
pub mod ppo;
pub mod reinforce;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::nn::Network;

/// Serializable model snapshot (saved to disk)
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelSnapshot {
    pub algorithm:        String,
    pub environment:      String,
    pub state_size:       usize,
    pub action_size:      usize,
    pub training_episodes: usize,
    pub best_avg_reward:  f64,
    pub policy_network:   Network,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_network:    Option<Network>,
    pub metadata:         serde_json::Value,
}

impl ModelSnapshot {
    pub fn save(&self, path: &str) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let full_path = if path.ends_with(".json") {
            path.to_string()
        } else {
            format!("{}.json", path)
        };
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&full_path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let full_path = if path.ends_with(".json") {
            path.to_string()
        } else {
            format!("{}.json", path)
        };
        let json = std::fs::read_to_string(&full_path)?;
        Ok(serde_json::from_str(&json)?)
    }
}

/// Minimal wrapper returned from `select_action` during evaluation
pub struct ActionResult {
    pub action: usize,
    pub value:  Option<f64>,   // critic estimate (if available)
}

/// Episode step data used by trajectory-based agents
#[derive(Clone)]
pub struct Transition {
    pub state:      Vec<f64>,
    pub action:     usize,
    pub reward:     f64,
    pub next_state: Vec<f64>,
    pub done:       bool,
    /// Log-probability of the taken action (for PPO)
    pub log_prob:   f64,
    /// Value estimate at state (for PPO / actor-critic)
    pub value:      f64,
}
