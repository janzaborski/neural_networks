use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::data::{DataSpec, TaskKind};
use crate::math_utils::matrix::{MatRef, MatRefMut};
use crate::math_utils::{Real, Scalar};
use crate::neural_network::activation::{self as act, Activation, ElementWise};
use crate::neural_network::cost::{Mae, Mse};
use crate::neural_network::head::{Composed, Head, SoftmaxCrossEntropy};
use crate::training::TrainConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub name: String,
    #[serde(default = "default_out_dir")]
    pub out_dir: PathBuf,
    #[serde(default)]
    pub dtype: Dtype,
    pub seeds: Vec<u64>,
    pub data: DataSpec,
    pub net: NetConfig,
    pub train: TrainSettings,
    #[serde(default)]
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    F32,
    #[default]
    F64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetConfig {
    pub hidden: Vec<usize>,
    #[serde(default = "default_true")]
    pub bias: bool,
    pub activation: HiddenKind,
    pub head: HeadKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainSettings {
    pub epochs: usize,
    pub batch: usize,
    pub lr: f64,
    #[serde(default = "default_log_every")]
    pub log_every: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    pub weights_every: usize,
    pub grid_steps: usize,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            weights_every: 0,
            grid_steps: 200,
        }
    }
}

fn default_out_dir() -> PathBuf {
    PathBuf::from("out")
}

fn default_true() -> bool {
    true
}

fn default_log_every() -> usize {
    1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HiddenKind {
    Sigmoid,
    Tanh,
    Relu,
    Identity,
}

impl HiddenKind {
    fn with<T: Real, R>(self, f: impl FnOnce(&dyn Activation<T>) -> R) -> R {
        match self {
            Self::Sigmoid => f(&ElementWise(act::Sigmoid)),
            Self::Tanh => f(&ElementWise(act::Tanh)),
            Self::Relu => f(&ElementWise(act::ReLU)),
            Self::Identity => f(&act::Identity),
        }
    }
}

impl<T: Real> Activation<T> for HiddenKind {
    fn forward(&self, z: MatRef<'_, T>, a: MatRefMut<'_, T>) {
        self.with(|h| h.forward(z, a))
    }

    fn backward(&self, z: MatRef<'_, T>, a: MatRef<'_, T>, grad: MatRefMut<'_, T>) {
        self.with(|h| h.backward(z, a, grad))
    }

    fn init_std(&self, fan_in: usize, fan_out: usize) -> f64 {
        self.with::<T, _>(|h| h.init_std(fan_in, fan_out))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadKind {
    IdentityMse,
    IdentityMae,
    SoftmaxCrossEntropy,
    SoftmaxMse,
}

impl HeadKind {
    fn with<T: Real, R>(self, f: impl FnOnce(&dyn Head<T>) -> R) -> R {
        match self {
            Self::IdentityMse => f(&Composed::new(act::Identity, Mse)),
            Self::IdentityMae => f(&Composed::new(act::Identity, Mae)),
            Self::SoftmaxCrossEntropy => f(&SoftmaxCrossEntropy),
            Self::SoftmaxMse => f(&Composed::new(act::Softmax, Mse)),
        }
    }

    pub fn task(self) -> TaskKind {
        match self {
            Self::IdentityMse | Self::IdentityMae => TaskKind::Regression,
            Self::SoftmaxCrossEntropy | Self::SoftmaxMse => TaskKind::Classification,
        }
    }
}

impl<T: Real> Head<T> for HeadKind {
    fn forward(&self, z: MatRef<'_, T>, a: MatRefMut<'_, T>) {
        self.with(|h| h.forward(z, a))
    }

    fn cost(&self, a: MatRef<'_, T>, y: MatRef<'_, T>) -> f64 {
        self.with(|h| h.cost(a, y))
    }

    fn delta(&self, z: MatRef<'_, T>, a: MatRef<'_, T>, y: MatRef<'_, T>, delta: MatRefMut<'_, T>) {
        self.with(|h| h.delta(z, a, y, delta))
    }

    fn init_std(&self, fan_in: usize, fan_out: usize) -> f64 {
        self.with::<T, _>(|h| h.init_std(fan_in, fan_out))
    }
}

impl TrainSettings {
    pub fn to_train_config<T: Scalar>(&self) -> TrainConfig<T> {
        TrainConfig {
            epochs: self.epochs,
            batch: self.batch,
            lr: T::from_f64(self.lr),
            log_every: self.log_every,
        }
    }
}

impl Config {
    pub fn dir(&self) -> PathBuf {
        self.out_dir.join(&self.name)
    }

    pub fn resolve(&mut self) -> Result<(), String> {
        self.data
            .resolve()
            .map_err(|e| format!("config {:?}: {e}", self.name))
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut errs = Vec::new();

        if self.name.is_empty()
            || self.name == "."
            || self.name == ".."
            || self.name.contains(['/', '\\'])
        {
            errs.push(format!(
                "invalid name {:?} (must be a plain directory name)",
                self.name
            ));
        }
        if self.seeds.is_empty() {
            errs.push("seeds is empty".to_string());
        }
        if self.seeds.iter().collect::<HashSet<_>>().len() != self.seeds.len() {
            errs.push(format!("duplicate seeds in {:?}", self.seeds));
        }
        if self.net.hidden.contains(&0) {
            errs.push(format!("hidden layer of size 0 in {:?}", self.net.hidden));
        }
        if self.net.head.task() != self.data.task {
            errs.push(format!(
                "head {:?} cannot be used for {:?}",
                self.net.head, self.data.task
            ));
        }
        if self.train.epochs == 0 {
            errs.push("epochs must be > 0".to_string());
        }
        if self.train.batch == 0 {
            errs.push("batch must be > 0".to_string());
        }
        if !(self.train.lr.is_finite() && self.train.lr > 0.0) {
            errs.push(format!("lr must be finite and > 0, got {}", self.train.lr));
        }
        if self.train.log_every == 0 {
            errs.push("log_every must be > 0".to_string());
        }
        if self.train.log_every > 0
            && !self
                .output
                .weights_every
                .is_multiple_of(self.train.log_every)
        {
            errs.push(format!(
                "weights_every ({}) must be 0 or a multiple of log_every ({})",
                self.output.weights_every, self.train.log_every
            ));
        }
        if !(0.0..1.0).contains(&self.data.validation) {
            errs.push(format!(
                "validation must be in [0, 1), got {}",
                self.data.validation
            ));
        }
        if self.output.grid_steps < 2 {
            errs.push("grid_steps must be >= 2".to_string());
        }

        if errs.is_empty() {
            Ok(())
        } else {
            Err(format!("config {:?}: {}", self.name, errs.join("; ")))
        }
    }
}

pub fn load_configs(path: &Path) -> Result<Vec<Config>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;

    match value {
        serde_json::Value::Array(items) => items
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                serde_json::from_value(v).map_err(|e| format!("{}[{i}]: {e}", path.display()))
            })
            .collect(),
        v => serde_json::from_value(v)
            .map(|c| vec![c])
            .map_err(|e| format!("{}: {e}", path.display())),
    }
}

pub fn validate_all(configs: &[Config]) -> Result<(), String> {
    let mut errs: Vec<String> = configs.iter().filter_map(|c| c.validate().err()).collect();

    let mut seen = HashSet::new();
    for c in configs {
        if !seen.insert(c.dir()) {
            errs.push(format!(
                "output directory {} used by more than one config",
                c.dir().display()
            ));
        }
    }

    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs.join("\n"))
    }
}
