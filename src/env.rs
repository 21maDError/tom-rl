use anyhow::{bail, Result};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::config::EnvironmentConfig;

// ─────────────────────────────────────────────────────────
// Shared types
// ─────────────────────────────────────────────────────────

pub struct StepResult {
    pub next_state: Vec<f64>,
    pub reward:     f64,
    pub done:       bool,
    pub info:       String,
}

pub trait Environment {
    fn reset(&mut self) -> Vec<f64>;
    fn step(&mut self, action: usize) -> StepResult;
    fn state_size(&self) -> usize;
    fn action_size(&self) -> usize;
    fn name(&self) -> &'static str;
    /// Human-readable summary of current state
    fn render(&self) -> String;
}

pub fn create_env(cfg: &EnvironmentConfig) -> Result<Box<dyn Environment>> {
    match cfg.name.as_str() {
        "cartpole"     => Ok(Box::new(CartPole::new(cfg.max_steps, cfg.seed))),
        "mountain_car" => Ok(Box::new(MountainCar::new(cfg.max_steps, cfg.seed))),
        "gridworld"    => Ok(Box::new(GridWorld::new(
            cfg.grid_size[0],
            cfg.grid_size[1],
            cfg.obstacle_fraction,
            cfg.max_steps,
            cfg.seed,
        ))),
        other => bail!("Unknown environment: '{}'", other),
    }
}

// ─────────────────────────────────────────────────────────
// CartPole
// Matches OpenAI Gym's CartPole-v1 physics exactly.
// ─────────────────────────────────────────────────────────

pub struct CartPole {
    state:     [f64; 4],
    steps:     usize,
    max_steps: usize,
    rng:       StdRng,
}

impl CartPole {
    const GRAVITY:         f64 = 9.8;
    const MASSCART:        f64 = 1.0;
    const MASSPOLE:        f64 = 0.1;
    const TOTAL_MASS:      f64 = Self::MASSCART + Self::MASSPOLE;
    const LENGTH:          f64 = 0.5;
    const POLEMASS_LENGTH: f64 = Self::MASSPOLE * Self::LENGTH;
    const FORCE_MAG:       f64 = 10.0;
    const TAU:             f64 = 0.02;
    const THETA_THRESH:    f64 = 12.0 * std::f64::consts::PI / 180.0;
    const X_THRESH:        f64 = 2.4;

    pub fn new(max_steps: usize, seed: u64) -> Self {
        CartPole { state: [0.0; 4], steps: 0, max_steps, rng: StdRng::seed_from_u64(seed) }
    }
}

impl Environment for CartPole {
    fn reset(&mut self) -> Vec<f64> {
        for v in self.state.iter_mut() {
            *v = self.rng.random_range(-0.05_f64..0.05_f64);
        }
        self.steps = 0;
        self.state.to_vec()
    }

    fn step(&mut self, action: usize) -> StepResult {
        let force = if action == 1 { Self::FORCE_MAG } else { -Self::FORCE_MAG };
        let [x, xd, th, thd] = self.state;

        let cos_th = th.cos();
        let sin_th = th.sin();
        let tmp = (force + Self::POLEMASS_LENGTH * thd * thd * sin_th) / Self::TOTAL_MASS;
        let th_acc = (Self::GRAVITY * sin_th - cos_th * tmp)
            / (Self::LENGTH * (4.0 / 3.0 - Self::MASSPOLE * cos_th * cos_th / Self::TOTAL_MASS));
        let x_acc = tmp - Self::POLEMASS_LENGTH * th_acc * cos_th / Self::TOTAL_MASS;

        self.state = [
            x   + Self::TAU * xd,
            xd  + Self::TAU * x_acc,
            th  + Self::TAU * thd,
            thd + Self::TAU * th_acc,
        ];
        self.steps += 1;

        let [nx, _, nth, _] = self.state;
        let done = nx.abs() > Self::X_THRESH
            || nth.abs() > Self::THETA_THRESH
            || self.steps >= self.max_steps;

        StepResult { next_state: self.state.to_vec(), reward: 1.0, done, info: String::new() }
    }

    fn state_size(&self)  -> usize { 4 }
    fn action_size(&self) -> usize { 2 }
    fn name(&self) -> &'static str { "CartPole" }

    fn render(&self) -> String {
        let [x, _, th, _] = self.state;
        format!("cart_pos={:.3}  pole_angle={:.2}°", x, th.to_degrees())
    }
}

// ─────────────────────────────────────────────────────────
// MountainCar
// Matches OpenAI Gym's MountainCar-v0 physics.
// ─────────────────────────────────────────────────────────

pub struct MountainCar {
    pos:       f64,
    vel:       f64,
    steps:     usize,
    max_steps: usize,
    rng:       StdRng,
}

impl MountainCar {
    const POWER:        f64 = 0.001;
    const GRAVITY:      f64 = 0.0025;
    const MIN_POS:      f64 = -1.2;
    const MAX_POS:      f64 = 0.6;
    const MAX_SPEED:    f64 = 0.07;
    const GOAL_POS:     f64 = 0.5;

    pub fn new(max_steps: usize, seed: u64) -> Self {
        MountainCar { pos: -0.6, vel: 0.0, steps: 0, max_steps, rng: StdRng::seed_from_u64(seed) }
    }
}

impl Environment for MountainCar {
    fn reset(&mut self) -> Vec<f64> {
        self.pos = self.rng.random_range(-0.6_f64..-0.4_f64);
        self.vel = 0.0;
        self.steps = 0;
        vec![self.pos, self.vel]
    }

    fn step(&mut self, action: usize) -> StepResult {
        // 0=push left, 1=no push, 2=push right
        let force: f64 = match action { 0 => -1.0, 2 => 1.0, _ => 0.0 };

        self.vel += force * Self::POWER - Self::GRAVITY * (3.0 * self.pos).cos();
        self.vel  = self.vel.clamp(-Self::MAX_SPEED, Self::MAX_SPEED);
        self.pos += self.vel;
        self.pos  = self.pos.clamp(Self::MIN_POS, Self::MAX_POS);

        if self.pos == Self::MIN_POS && self.vel < 0.0 {
            self.vel = 0.0;
        }
        self.steps += 1;

        let reached_goal = self.pos >= Self::GOAL_POS;
        let done = reached_goal || self.steps >= self.max_steps;

        // Reward shaping: +100 on goal, else -1
        let reward = if reached_goal { 100.0 } else { -1.0 };

        StepResult {
            next_state: vec![self.pos, self.vel],
            reward,
            done,
            info: if reached_goal { "Goal reached!".to_string() } else { String::new() },
        }
    }

    fn state_size(&self)  -> usize { 2 }
    fn action_size(&self) -> usize { 3 }
    fn name(&self) -> &'static str { "MountainCar" }

    fn render(&self) -> String {
        let bar_len = 40usize;
        let frac = (self.pos - Self::MIN_POS) / (Self::MAX_POS - Self::MIN_POS);
        let car = (frac * bar_len as f64) as usize;
        let bar: String = (0..bar_len).map(|i| if i == car { '▲' } else { '─' }).collect();
        format!("[{}] pos={:.3}  vel={:.4}", bar, self.pos, self.vel)
    }
}

// ─────────────────────────────────────────────────────────
// GridWorld
// A configurable N×M grid with random obstacles.
// State: one-hot encoding of (row, col).
// Actions: 0=up 1=down 2=left 3=right
// ─────────────────────────────────────────────────────────

pub struct GridWorld {
    rows:       usize,
    cols:       usize,
    obstacles:  Vec<(usize, usize)>,
    agent_r:    usize,
    agent_c:    usize,
    steps:      usize,
    max_steps:  usize,
    rng:        StdRng,
    start:      (usize, usize),
    goal:       (usize, usize),
}

impl GridWorld {
    pub fn new(rows: usize, cols: usize, obstacle_frac: f64, max_steps: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let start = (0, 0);
        let goal  = (rows - 1, cols - 1);

        // Random obstacles (avoid start/goal)
        let n_obstacles = ((rows * cols) as f64 * obstacle_frac) as usize;
        let mut obstacles = Vec::new();
        while obstacles.len() < n_obstacles {
            let c = rng.random_range(0..cols);
            let r = rng.random_range(0..rows);
            let pos = (r, c);
            if pos != start && pos != goal && !obstacles.contains(&pos) {
                obstacles.push(pos);
            }
        }

        GridWorld {
            rows, cols, obstacles,
            agent_r: 0, agent_c: 0,
            steps: 0, max_steps,
            rng, start, goal,
        }
    }

    fn encode_state(&self) -> Vec<f64> {
        let mut s = vec![0.0; self.rows * self.cols];
        s[self.agent_r * self.cols + self.agent_c] = 1.0;
        s
    }

    fn is_obstacle(&self, r: usize, c: usize) -> bool {
        self.obstacles.contains(&(r, c))
    }
}

impl Environment for GridWorld {
    fn reset(&mut self) -> Vec<f64> {
        self.agent_r = self.start.0;
        self.agent_c = self.start.1;
        self.steps   = 0;
        self.encode_state()
    }

    fn step(&mut self, action: usize) -> StepResult {
        let (dr, dc): (i64, i64) = match action { 0 => (-1,0), 1 => (1,0), 2 => (0,-1), _ => (0,1) };
        let nr = (self.agent_r as i64 + dr).clamp(0, self.rows as i64 - 1) as usize;
        let nc = (self.agent_c as i64 + dc).clamp(0, self.cols as i64 - 1) as usize;

        let blocked = self.is_obstacle(nr, nc);
        if !blocked {
            self.agent_r = nr;
            self.agent_c = nc;
        }
        self.steps += 1;

        let at_goal = (self.agent_r, self.agent_c) == self.goal;
        let done = at_goal || self.steps >= self.max_steps;

        let reward = if at_goal { 10.0 } else if blocked { -0.5 } else { -0.1 };

        StepResult {
            next_state: self.encode_state(),
            reward,
            done,
            info: if at_goal { "Goal!".into() } else { String::new() },
        }
    }

    fn state_size(&self)  -> usize { self.rows * self.cols }
    fn action_size(&self) -> usize { 4 }
    fn name(&self) -> &'static str { "GridWorld" }

    fn render(&self) -> String {
        let mut grid = String::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                let ch = if (r, c) == (self.agent_r, self.agent_c) { 'A' }
                    else if (r, c) == self.goal               { 'G' }
                    else if (r, c) == self.start              { 'S' }
                    else if self.is_obstacle(r, c)            { '█' }
                    else                                       { '·' };
                grid.push(ch);
                grid.push(' ');
            }
            grid.push('\n');
        }
        grid
    }
}
