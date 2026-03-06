use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────
// Activation functions
// ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Activation {
    Relu,
    Tanh,
    Sigmoid,
    Linear,
}

impl Activation {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "relu"    => Activation::Relu,
            "tanh"    => Activation::Tanh,
            "sigmoid" => Activation::Sigmoid,
            _         => Activation::Linear,
        }
    }

    pub fn apply(&self, x: &[f64]) -> Vec<f64> {
        match self {
            Activation::Relu    => x.iter().map(|&v| v.max(0.0)).collect(),
            Activation::Tanh    => x.iter().map(|&v| v.tanh()).collect(),
            Activation::Sigmoid => x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(),
            Activation::Linear  => x.to_vec(),
        }
    }

    /// Element-wise activation gradient: dL/dz = dL/da * da/dz
    pub fn backward(&self, grad_out: &[f64], pre_act: &[f64]) -> Vec<f64> {
        match self {
            Activation::Relu => grad_out.iter().zip(pre_act.iter())
                .map(|(&g, &z)| if z > 0.0 { g } else { 0.0 })
                .collect(),
            Activation::Tanh => grad_out.iter().zip(pre_act.iter())
                .map(|(&g, &z)| g * (1.0 - z.tanh().powi(2)))
                .collect(),
            Activation::Sigmoid => grad_out.iter().zip(pre_act.iter())
                .map(|(&g, &z)| { let s = 1.0 / (1.0 + (-z).exp()); g * s * (1.0 - s) })
                .collect(),
            Activation::Linear => grad_out.to_vec(),
        }
    }
}

// ─────────────────────────────────────────────────────────
// Layer
// ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// weights[out][in]
    pub weights: Vec<Vec<f64>>,
    pub biases:  Vec<f64>,
    pub activation: Activation,

    // Cached values for backpropagation (not serialized)
    #[serde(skip)]
    input_cache:   Vec<f64>,
    #[serde(skip)]
    pre_act_cache: Vec<f64>,
}

impl Layer {
    pub fn new(in_size: usize, out_size: usize, activation: Activation, rng: &mut impl Rng) -> Self {
        // Xavier / Glorot initialization
        let scale = (2.0 / (in_size + out_size) as f64).sqrt();
        let normal = Normal::new(0.0, scale).unwrap();
        let weights = (0..out_size)
            .map(|_| (0..in_size).map(|_| normal.sample(rng)).collect())
            .collect();
        Layer {
            weights,
            biases: vec![0.0; out_size],
            activation,
            input_cache:   vec![0.0; in_size],
            pre_act_cache: vec![0.0; out_size],
        }
    }

    /// Forward pass — caches input and pre-activation for backprop.
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        self.input_cache = input.to_vec();
        let n_out = self.weights.len();
        let mut pre_act = vec![0.0; n_out];
        for i in 0..n_out {
            pre_act[i] = self.biases[i]
                + self.weights[i].iter().zip(input).map(|(w, x)| w * x).sum::<f64>();
        }
        self.pre_act_cache = pre_act.clone();
        self.activation.apply(&pre_act)
    }

    /// Forward pass without caching (used for target / eval inference).
    pub fn forward_no_grad(weights: &[Vec<f64>], biases: &[f64], activation: &Activation, input: &[f64]) -> Vec<f64> {
        let n_out = weights.len();
        let mut pre_act = vec![0.0; n_out];
        for i in 0..n_out {
            pre_act[i] = biases[i]
                + weights[i].iter().zip(input).map(|(w, x)| w * x).sum::<f64>();
        }
        activation.apply(&pre_act)
    }

    /// Backward pass.
    /// Returns (grad_weights, grad_biases, grad_input)
    pub fn backward(&self, grad_out: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let grad_pre = self.activation.backward(grad_out, &self.pre_act_cache);
        let n_out = self.weights.len();
        let n_in  = self.input_cache.len();

        let mut grad_w = vec![vec![0.0; n_in]; n_out];
        for i in 0..n_out {
            for j in 0..n_in {
                grad_w[i][j] = grad_pre[i] * self.input_cache[j];
            }
        }

        let mut grad_in = vec![0.0; n_in];
        for j in 0..n_in {
            for i in 0..n_out {
                grad_in[j] += self.weights[i][j] * grad_pre[i];
            }
        }

        (grad_w, grad_pre, grad_in) // grad_biases == grad_pre
    }
}

// ─────────────────────────────────────────────────────────
// Network
// ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub layers:   Vec<Layer>,
    pub in_size:  usize,
    pub out_size: usize,
}

impl Network {
    /// `layer_sizes`: [input, hidden..., output]
    /// `activations`: one per layer transition (len = layer_sizes.len() - 1)
    pub fn new(layer_sizes: &[usize], activations: &[Activation], rng: &mut impl Rng) -> Self {
        assert!(layer_sizes.len() >= 2);
        assert_eq!(layer_sizes.len() - 1, activations.len());
        let layers = (0..layer_sizes.len() - 1)
            .map(|i| Layer::new(layer_sizes[i], layer_sizes[i + 1], activations[i].clone(), rng))
            .collect();
        Network {
            layers,
            in_size:  layer_sizes[0],
            out_size: *layer_sizes.last().unwrap(),
        }
    }

    /// Forward with caching (for training).
    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut x = input.to_vec();
        for layer in &mut self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Forward without caching (for inference / target network).
    pub fn forward_no_grad(&self, input: &[f64]) -> Vec<f64> {
        let mut x = input.to_vec();
        for layer in &self.layers {
            x = Layer::forward_no_grad(&layer.weights, &layer.biases, &layer.activation, &x);
        }
        x
    }

    /// Backward pass given loss gradient w.r.t. network output.
    /// Returns gradients for every layer: Vec<(grad_weights, grad_biases)>
    pub fn backward(&self, grad_output: &[f64]) -> Vec<(Vec<Vec<f64>>, Vec<f64>)> {
        let n = self.layers.len();
        let mut all_grads = vec![(vec![], vec![]); n];
        let mut grad = grad_output.to_vec();
        for i in (0..n).rev() {
            let (gw, gb, gin) = self.layers[i].backward(&grad);
            all_grads[i] = (gw, gb);
            grad = gin;
        }
        all_grads
    }

    /// Copy weights from another network (used for target network sync).
    pub fn copy_weights_from(&mut self, other: &Network) {
        for (sl, ol) in self.layers.iter_mut().zip(other.layers.iter()) {
            sl.weights = ol.weights.clone();
            sl.biases  = ol.biases.clone();
        }
    }

    /// Zero-initialise a gradient structure matching this network.
    pub fn zero_grads(&self) -> Vec<(Vec<Vec<f64>>, Vec<f64>)> {
        self.layers.iter().map(|l| {
            let gw = vec![vec![0.0; l.weights[0].len()]; l.weights.len()];
            let gb = vec![0.0; l.biases.len()];
            (gw, gb)
        }).collect()
    }

    /// Add `other` gradients into `acc` (in-place).
    pub fn add_grads(
        acc: &mut [(Vec<Vec<f64>>, Vec<f64>)],
        other: &[(Vec<Vec<f64>>, Vec<f64>)],
    ) {
        for (a, o) in acc.iter_mut().zip(other.iter()) {
            for (ar, or_) in a.0.iter_mut().zip(o.0.iter()) {
                for (av, ov) in ar.iter_mut().zip(or_.iter()) {
                    *av += ov;
                }
            }
            for (ab, ob) in a.1.iter_mut().zip(o.1.iter()) {
                *ab += ob;
            }
        }
    }

    /// Scale all gradients by a scalar.
    pub fn scale_grads(grads: &mut [(Vec<Vec<f64>>, Vec<f64>)], scale: f64) {
        for (gw, gb) in grads.iter_mut() {
            for row in gw.iter_mut() {
                for v in row.iter_mut() { *v *= scale; }
            }
            for v in gb.iter_mut() { *v *= scale; }
        }
    }

    /// Apply gradients with a plain learning rate (SGD step).
    pub fn apply_grads_sgd(&mut self, grads: &[(Vec<Vec<f64>>, Vec<f64>)], lr: f64) {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            for (r, gr) in layer.weights.iter_mut().zip(grads[i].0.iter()) {
                for (w, gw) in r.iter_mut().zip(gr.iter()) {
                    *w -= lr * gw;
                }
            }
            for (b, gb) in layer.biases.iter_mut().zip(grads[i].1.iter()) {
                *b -= lr * gb;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────
// Adam optimizer
// ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Adam {
    pub lr:    f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps:   f64,
    t: usize,
    // First & second moment buffers: [layer][row][col] for weights, [layer][i] for biases
    mw: Vec<Vec<Vec<f64>>>,
    vw: Vec<Vec<Vec<f64>>>,
    mb: Vec<Vec<f64>>,
    vb: Vec<Vec<f64>>,
}

impl Adam {
    pub fn new(net: &Network, lr: f64) -> Self {
        let mw: Vec<_> = net.layers.iter().map(|l| vec![vec![0.0; l.weights[0].len()]; l.weights.len()]).collect();
        let vw = mw.clone();
        let mb: Vec<_> = net.layers.iter().map(|l| vec![0.0; l.biases.len()]).collect();
        let vb = mb.clone();
        Adam { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, mw, vw, mb, vb }
    }

    pub fn step(&mut self, net: &mut Network, grads: &[(Vec<Vec<f64>>, Vec<f64>)]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, layer) in net.layers.iter_mut().enumerate() {
            // Weights
            for r in 0..layer.weights.len() {
                for c in 0..layer.weights[r].len() {
                    let g = grads[i].0[r][c];
                    self.mw[i][r][c] = self.beta1 * self.mw[i][r][c] + (1.0 - self.beta1) * g;
                    self.vw[i][r][c] = self.beta2 * self.vw[i][r][c] + (1.0 - self.beta2) * g * g;
                    let m_hat = self.mw[i][r][c] / bc1;
                    let v_hat = self.vw[i][r][c] / bc2;
                    layer.weights[r][c] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
                }
            }
            // Biases
            for j in 0..layer.biases.len() {
                let g = grads[i].1[j];
                self.mb[i][j] = self.beta1 * self.mb[i][j] + (1.0 - self.beta1) * g;
                self.vb[i][j] = self.beta2 * self.vb[i][j] + (1.0 - self.beta2) * g * g;
                let m_hat = self.mb[i][j] / bc1;
                let v_hat = self.vb[i][j] / bc2;
                layer.biases[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────

/// Numerically stable softmax
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_l).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// log-sum-exp for numerical stability
pub fn log_prob_softmax(logits: &[f64], action: usize) -> f64 {
    let probs = softmax(logits);
    probs[action].max(1e-10).ln()
}

/// Build layer sizes: [state_size, hidden..., action_size]
pub fn build_layer_sizes(state: usize, hidden: &[usize], actions: usize) -> Vec<usize> {
    let mut sizes = vec![state];
    sizes.extend_from_slice(hidden);
    sizes.push(actions);
    sizes
}

/// Build activation list: hidden activations + final Linear
pub fn build_activations(act_str: &str, n_layers: usize) -> Vec<Activation> {
    let hidden_act = Activation::from_str(act_str);
    let mut acts: Vec<Activation> = (0..n_layers - 1).map(|_| hidden_act.clone()).collect();
    acts.push(Activation::Linear); // output layer always linear
    acts
}

/// Compute discounted returns for a reward sequence
pub fn compute_returns(rewards: &[f64], gamma: f64) -> Vec<f64> {
    let mut returns = vec![0.0; rewards.len()];
    let mut running = 0.0;
    for i in (0..rewards.len()).rev() {
        running = rewards[i] + gamma * running;
        returns[i] = running;
    }
    returns
}

/// Normalise a vector to zero mean, unit variance
pub fn normalize(v: &[f64]) -> Vec<f64> {
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let var  = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
    let std  = var.sqrt().max(1e-8);
    v.iter().map(|x| (x - mean) / std).collect()
}
