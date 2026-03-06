use anyhow::{bail, Result};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::agents::dqn::DqnAgent;
use crate::agents::ppo::PpoAgent;
use crate::agents::reinforce::ReinforceAgent;
use crate::agents::{ModelSnapshot, Transition};
use crate::config::Config;
use crate::env::{create_env, Environment};
use crate::nn::softmax;

// ─────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────

pub fn train(cfg: &Config, verbose: bool) -> Result<()> {
    match cfg.algorithm.name.as_str() {
        "dqn" => train_dqn(cfg, verbose),
        "reinforce" => train_reinforce(cfg, verbose),
        "ppo" => train_ppo(cfg, verbose),
        other => bail!("Unknown algorithm: '{}'", other),
    }
}

pub fn evaluate(cfg: &Config, model_path: &str, n_episodes: usize) -> Result<()> {
    println!(
        "{}",
        "\n── Evaluation ──────────────────────────────"
            .cyan()
            .bold()
    );

    let snapshot = ModelSnapshot::load(model_path)?;
    println!("  Model     : {}", model_path.bright_white());
    println!("  Algorithm : {}", snapshot.algorithm.yellow());
    println!("  Env       : {}", snapshot.environment.yellow());
    println!("  Trained   : {} episodes", snapshot.training_episodes);
    println!("  Best avg  : {:.2}\n", snapshot.best_avg_reward);

    let mut env = create_env(&cfg.environment)?;
    let mut rewards = Vec::with_capacity(n_episodes);

    for ep in 0..n_episodes {
        let mut state = env.reset();
        let mut total = 0.0_f64;
        for _ in 0..cfg.environment.max_steps {
            let logits = snapshot.policy_network.forward_no_grad(&state);
            let probs = softmax(&logits);
            let action = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let result = env.step(action);
            total += result.reward;
            state = result.next_state;
            if result.done {
                break;
            }
        }
        println!("  Episode {:>3}: reward = {:.2}", ep + 1, total);
        rewards.push(total);
    }

    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let max = rewards.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = rewards.iter().copied().fold(f64::INFINITY, f64::min);

    println!(
        "\n{}",
        "── Results ─────────────────────────────────".cyan().bold()
    );
    println!("  Mean reward : {:.2}", mean);
    println!("  Max  reward : {:.2}", max);
    println!("  Min  reward : {:.2}", min);
    Ok(())
}

// ─────────────────────────────────────────────────────────
// DQN training
// ─────────────────────────────────────────────────────────

fn train_dqn(cfg: &Config, verbose: bool) -> Result<()> {
    let mut env = create_env(&cfg.environment)?;
    let mut rng = StdRng::seed_from_u64(cfg.environment.seed);
    let mut agent = DqnAgent::new(env.state_size(), env.action_size(), cfg, &mut rng);

    let pb = make_progress_bar(cfg.training.episodes as u64, "DQN");
    let mut ep_rewards: Vec<f64> = Vec::new();
    let mut losses: Vec<f64> = Vec::new();
    let mut best_avg = f64::NEG_INFINITY;

    for episode in 0..cfg.training.episodes {
        let mut state = env.reset();
        let mut ep_reward = 0.0_f64;

        loop {
            let action = agent.select_action(&state, true);
            let result = env.step(action);
            agent.store_transition(
                &state,
                action,
                result.reward,
                &result.next_state,
                result.done,
            );

            if let Some(loss) = agent.update() {
                losses.push(loss);
            }

            ep_reward += result.reward;
            state = result.next_state;
            if result.done {
                break;
            }
        }

        ep_rewards.push(ep_reward);

        // Logging
        if episode % cfg.training.log_interval == 0 {
            let avg100 = sliding_avg(&ep_rewards, 100);
            let avg_loss = if losses.is_empty() {
                0.0
            } else {
                losses.iter().rev().take(100).sum::<f64>() / losses.len().min(100) as f64
            };
            pb.set_message(format!(
                "ep {:>5} | reward {:>7.1} | avg100 {:>7.1} | ε {:.3} | loss {:.4}",
                episode, ep_reward, avg100, agent.epsilon, avg_loss
            ));

            if verbose {
                pb.println(format!(
                    "  [ep {:>5}] reward={:.1}  avg100={:.1}  ε={:.3}  loss={:.4}",
                    episode, ep_reward, avg100, agent.epsilon, avg_loss
                ));
            }

            // Save best checkpoint
            if cfg.training.save_best && avg100 > best_avg {
                best_avg = avg100;
                agent.save(&cfg.output.model_path, episode, best_avg)?;
            }

            // Early stop
            if let Some(target) = cfg.training.target_reward {
                if avg100 >= target {
                    pb.println(format!(
                        "{}",
                        format!(
                            "  🎯 Target reward {:.1} reached at episode {}!",
                            target, episode
                        )
                        .green()
                        .bold()
                    ));
                    break;
                }
            }
        }

        // Periodic eval
        if episode % cfg.training.eval_interval == 0 && episode > 0 {
            let eval_r = eval_agent_dqn(&mut agent, &mut *env, cfg);
            pb.println(format!(
                "{}",
                format!("  ✔ eval (ep {:>5}): avg_reward={:.2}", episode, eval_r).cyan()
            ));
        }

        pb.inc(1);
    }

    pb.finish_with_message("Training complete!");

    // Final save
    agent.save(&cfg.output.model_path, cfg.training.episodes, best_avg)?;
    print_summary(&ep_rewards, cfg);

    if cfg.output.print_curve {
        print_reward_curve(&ep_rewards);
    }
    Ok(())
}

fn eval_agent_dqn(agent: &mut DqnAgent, env: &mut dyn Environment, cfg: &Config) -> f64 {
    let mut total = 0.0_f64;
    for _ in 0..cfg.training.eval_episodes {
        let mut s = env.reset();
        loop {
            let a = agent.evaluate(&s);
            let r = env.step(a);
            total += r.reward;
            s = r.next_state;
            if r.done {
                break;
            }
        }
    }
    total / cfg.training.eval_episodes as f64
}

// ─────────────────────────────────────────────────────────
// REINFORCE training
// ─────────────────────────────────────────────────────────

fn train_reinforce(cfg: &Config, verbose: bool) -> Result<()> {
    let mut env = create_env(&cfg.environment)?;
    let mut rng = StdRng::seed_from_u64(cfg.environment.seed);
    let mut agent = ReinforceAgent::new(env.state_size(), env.action_size(), cfg, &mut rng);

    let pb = make_progress_bar(cfg.training.episodes as u64, "REINFORCE");
    let mut ep_rewards: Vec<f64> = Vec::new();
    let mut best_avg = f64::NEG_INFINITY;

    for episode in 0..cfg.training.episodes {
        let mut state = env.reset();
        let mut states: Vec<Vec<f64>> = Vec::new();
        let mut actions: Vec<usize> = Vec::new();
        let mut rewards: Vec<f64> = Vec::new();

        // Collect episode
        loop {
            let (action, _log_prob) = agent.select_action(&state, &mut rng);
            let result = env.step(action);
            states.push(state.clone());
            actions.push(action);
            rewards.push(result.reward);
            state = result.next_state;
            if result.done {
                break;
            }
        }

        let ep_reward = rewards.iter().sum();
        ep_rewards.push(ep_reward);

        // Policy gradient update
        let loss = agent.update(&states, &actions, &rewards);

        if episode % cfg.training.log_interval == 0 {
            let avg100 = sliding_avg(&ep_rewards, 100);
            pb.set_message(format!(
                "ep {:>5} | reward {:>7.1} | avg100 {:>7.1} | loss {:.4}",
                episode, ep_reward, avg100, loss
            ));

            if verbose {
                pb.println(format!(
                    "  [ep {:>5}] reward={:.1}  avg100={:.1}  loss={:.4}",
                    episode, ep_reward, avg100, loss
                ));
            }

            if cfg.training.save_best && avg100 > best_avg {
                best_avg = avg100;
                agent.save(&cfg.output.model_path, episode, best_avg)?;
            }

            if let Some(target) = cfg.training.target_reward {
                if avg100 >= target {
                    pb.println(format!(
                        "{}",
                        format!("  🎯 Target reached at episode {}!", episode)
                            .green()
                            .bold()
                    ));
                    break;
                }
            }
        }

        if episode % cfg.training.eval_interval == 0 && episode > 0 {
            let eval_r = eval_agent_reinforce(&agent, &mut *env, cfg);
            pb.println(format!(
                "{}",
                format!("  ✔ eval (ep {:>5}): avg_reward={:.2}", episode, eval_r).cyan()
            ));
        }

        pb.inc(1);
    }

    pb.finish_with_message("Training complete!");
    agent.save(&cfg.output.model_path, cfg.training.episodes, best_avg)?;
    print_summary(&ep_rewards, cfg);
    if cfg.output.print_curve {
        print_reward_curve(&ep_rewards);
    }
    Ok(())
}

fn eval_agent_reinforce(agent: &ReinforceAgent, env: &mut dyn Environment, cfg: &Config) -> f64 {
    let mut total = 0.0_f64;
    for _ in 0..cfg.training.eval_episodes {
        let mut s = env.reset();
        loop {
            let a = agent.greedy_action(&s);
            let r = env.step(a);
            total += r.reward;
            s = r.next_state;
            if r.done {
                break;
            }
        }
    }
    total / cfg.training.eval_episodes as f64
}

// ─────────────────────────────────────────────────────────
// PPO training
// ─────────────────────────────────────────────────────────

fn train_ppo(cfg: &Config, verbose: bool) -> Result<()> {
    let mut env = create_env(&cfg.environment)?;
    let mut rng = StdRng::seed_from_u64(cfg.environment.seed);
    let mut agent = PpoAgent::new(env.state_size(), env.action_size(), cfg, &mut rng);

    let pb = make_progress_bar(cfg.training.episodes as u64, "PPO");
    let mut ep_rewards: Vec<f64> = Vec::new();
    let mut best_avg = f64::NEG_INFINITY;
    let steps_per_update = cfg.algorithm.ppo.steps_per_update;

    let mut trajectory: Vec<Transition> = Vec::new();
    let mut current_ep_reward = 0.0_f64;
    let mut completed_episodes = 0usize;
    let mut state = env.reset();

    loop {
        // Collect `steps_per_update` transitions
        trajectory.clear();
        for _ in 0..steps_per_update {
            let (action, log_prob, value) = agent.select_action(&state);
            let result = env.step(action);
            let done = result.done;

            trajectory.push(Transition {
                state: state.clone(),
                action,
                reward: result.reward,
                next_state: result.next_state.clone(),
                done,
                log_prob,
                value,
            });

            current_ep_reward += result.reward;
            state = result.next_state;

            if done {
                ep_rewards.push(current_ep_reward);
                current_ep_reward = 0.0;
                completed_episodes += 1;
                pb.inc(1);
                state = env.reset();

                if completed_episodes >= cfg.training.episodes {
                    break;
                }
            }
        }

        // Compute bootstrap value for the last state
        let last_value = if trajectory.last().map(|t| t.done).unwrap_or(false) {
            0.0
        } else {
            agent.value_estimate(&state)
        };

        // PPO update
        let (actor_loss, critic_loss) = agent.update(&trajectory, last_value);

        let episode = completed_episodes;
        if episode % cfg.training.log_interval == 0 && episode > 0 {
            let avg100 = sliding_avg(&ep_rewards, 100);
            let ep_reward = ep_rewards.last().cloned().unwrap_or(0.0);
            pb.set_message(format!(
                "ep {:>5} | reward {:>7.1} | avg100 {:>7.1} | actor {:.4} | critic {:.4}",
                episode, ep_reward, avg100, actor_loss, critic_loss
            ));

            if verbose {
                pb.println(format!(
                    "  [ep {:>5}] reward={:.1}  avg100={:.1}  a_loss={:.4}  c_loss={:.4}",
                    episode, ep_reward, avg100, actor_loss, critic_loss
                ));
            }

            if cfg.training.save_best && avg100 > best_avg {
                best_avg = avg100;
                agent.save(&cfg.output.model_path, episode, best_avg)?;
            }

            if let Some(target) = cfg.training.target_reward {
                if avg100 >= target {
                    pb.println(format!(
                        "{}",
                        format!("  🎯 Target reached at episode {}!", episode)
                            .green()
                            .bold()
                    ));
                    break;
                }
            }
        }

        if episode % cfg.training.eval_interval == 0 && episode > 0 {
            let eval_r = eval_agent_ppo(&agent, &mut *env, cfg);
            pb.println(format!(
                "{}",
                format!("  ✔ eval (ep {:>5}): avg_reward={:.2}", episode, eval_r).cyan()
            ));
        }

        if completed_episodes >= cfg.training.episodes {
            break;
        }
    }

    pb.finish_with_message("Training complete!");
    agent.save(&cfg.output.model_path, cfg.training.episodes, best_avg)?;
    print_summary(&ep_rewards, cfg);
    if cfg.output.print_curve {
        print_reward_curve(&ep_rewards);
    }
    Ok(())
}

fn eval_agent_ppo(agent: &PpoAgent, env: &mut dyn Environment, cfg: &Config) -> f64 {
    let mut total = 0.0_f64;
    for _ in 0..cfg.training.eval_episodes {
        let mut s = env.reset();
        loop {
            let a = agent.greedy_action(&s);
            let r = env.step(a);
            total += r.reward;
            s = r.next_state;
            if r.done {
                break;
            }
        }
    }
    total / cfg.training.eval_episodes as f64
}

// ─────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────

fn make_progress_bar(total: u64, algo: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            &format!("{{spinner:.green}} [{{elapsed_precise}}] {{bar:40.cyan/blue}} {{pos:>7}}/{{len:7}} {algo} | {{msg}}")
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb
}

fn sliding_avg(v: &[f64], window: usize) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let take = v.len().min(window);
    v.iter().rev().take(take).sum::<f64>() / take as f64
}

fn print_summary(rewards: &[f64], cfg: &Config) {
    println!(
        "\n{}",
        "── Training Summary ────────────────────────".cyan().bold()
    );

    let mean = sliding_avg(rewards, rewards.len());
    let last = sliding_avg(rewards, 100);
    let best = rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("  Total episodes   : {}", rewards.len());
    println!("  Overall mean     : {:.2}", mean);
    println!("  Last-100 avg     : {:.2}", last);
    println!("  Best episode     : {:.2}", best);
    println!(
        "  Model saved to   : {}",
        format!("{}.json", cfg.output.model_path).bright_white()
    );
}

fn print_reward_curve(rewards: &[f64]) {
    const WIDTH: usize = 60;
    const HEIGHT: usize = 10;

    if rewards.is_empty() {
        return;
    }

    // Down-sample to WIDTH points
    let step = (rewards.len() as f64 / WIDTH as f64).ceil() as usize;
    let sampled: Vec<f64> = rewards
        .chunks(step)
        .map(|c| c.iter().sum::<f64>() / c.len() as f64)
        .collect();

    let min = sampled.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = sampled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1.0);

    println!(
        "\n{}",
        "── Reward Curve ─────────────────────────────"
            .cyan()
            .bold()
    );
    println!("  Max: {:.1}", max);

    for row in (0..HEIGHT).rev() {
        let threshold = min + (row as f64 / (HEIGHT - 1) as f64) * range;
        let line: String = sampled
            .iter()
            .map(|&v| if v >= threshold { '█' } else { ' ' })
            .collect();
        let label = if row == HEIGHT - 1 {
            format!("{:>7.1} ┤", max)
        } else if row == 0 {
            format!("{:>7.1} ┤", min)
        } else {
            "        │".to_string()
        };
        println!("  {}{}", label, line);
    }
    println!("          └{}┘", "─".repeat(sampled.len()));
    println!(
        "           0{}{}episodes",
        " ".repeat(sampled.len() / 2 - 4),
        rewards.len()
    );
    println!();
}
