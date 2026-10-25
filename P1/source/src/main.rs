use std::collections::HashMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::{Value, json};

use neural_net::config::{Config, Dtype, HeadKind, HiddenKind, load_configs, validate_all};
use neural_net::data::{DataSpec, Dataset, Normalization, Task, TaskKind};
use neural_net::eval::{self, History, WeightLog};
use neural_net::math_utils::Real;
use neural_net::neural_network::NeuralNetwork;
use neural_net::training::train;

type Res<T> = Result<T, Box<dyn Error + Send + Sync>>;
type Net<T> = NeuralNetwork<T, HiddenKind, HeadKind>;

struct Prepared<T> {
    raw_train: Option<Dataset<T>>,
    train: Dataset<T>,
    val: Option<Dataset<T>>,
    test: Option<Dataset<T>>,
    evals: Vec<(String, Dataset<T>)>,
    norm: Normalization<T>,
}

impl<T> Prepared<T> {
    fn splits(&self) -> Vec<(&str, &Dataset<T>)> {
        let mut s = vec![("train", &self.train)];
        if let Some(v) = &self.val {
            s.push(("val", v));
        }
        if let Some(t) = &self.test {
            s.push(("test", t));
        }
        s.extend(self.evals.iter().map(|(name, d)| (name.as_str(), d)));
        s
    }
}

type Cache<T> = HashMap<String, Prepared<T>>;

const PLOT_MAX_INPUTS: usize = 2;

fn usage() -> String {
    "usage: nn [--overwrite] <config.json>...\n  each file holds one config object or an array of them".to_string()
}

fn main() -> ExitCode {
    match run() {
        Ok(0) => ExitCode::SUCCESS,
        Ok(failed) => {
            eprintln!("{failed} run(s) failed");
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Res<usize> {
    let mut overwrite = false;
    let mut files = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--overwrite" => overwrite = true,
            "-h" | "--help" => return Err(usage().into()),
            s if s.starts_with("--") => return Err(format!("unknown flag {s}\n{}", usage()).into()),
            _ => files.push(PathBuf::from(arg)),
        }
    }
    if files.is_empty() {
        return Err(usage().into());
    }

    let mut configs = Vec::new();
    for f in &files {
        configs.extend(load_configs(f)?);
    }
    let resolve_errors: Vec<String> = configs
        .iter_mut()
        .filter_map(|c| c.resolve().err())
        .collect();
    if !resolve_errors.is_empty() {
        return Err(resolve_errors.join("\n").into());
    }
    validate_all(&configs)?;

    let existing: Vec<String> = configs
        .iter()
        .map(Config::dir)
        .filter(|d| d.exists())
        .map(|d| d.display().to_string())
        .collect();
    if !existing.is_empty() && !overwrite {
        return Err(format!(
            "output already exists (use --overwrite):\n  {}",
            existing.join("\n  ")
        )
        .into());
    }

    let c64: Vec<&Config> = configs.iter().filter(|c| c.dtype == Dtype::F64).collect();
    let c32: Vec<&Config> = configs.iter().filter(|c| c.dtype == Dtype::F32).collect();
    let d64 = load_cache::<f64>(&c64)?;
    let d32 = load_cache::<f32>(&c32)?;

    for c in &configs {
        let dir = c.dir();
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        fs::create_dir_all(&dir)?;
        write_json(dir.join("config.json"), c)?;
    }
    write_points(&c64, &d64)?;
    write_points(&c32, &d32)?;

    let jobs: usize = configs.iter().map(|c| c.seeds.len()).sum();
    println!(
        "{} config(s), {jobs} run(s), {} thread(s)",
        configs.len(),
        rayon::current_num_threads()
    );

    let mut failures = run_jobs(&c64, &d64);
    failures.extend(run_jobs(&c32, &d32));
    for f in &failures {
        eprintln!("FAILED {f}");
    }
    Ok(failures.len())
}

fn load_cache<T: Real>(configs: &[&Config]) -> Res<Cache<T>> {
    let mut cache = Cache::new();
    for c in configs {
        let key = c.data.key();
        if !cache.contains_key(&key) {
            let p = prepare::<T>(&c.data)?;
            cache.insert(key.clone(), p);
        }
        let d = &cache[&key];
        if c.train.batch > d.train.len() {
            return Err(format!(
                "config {:?}: batch {} > {} training samples",
                c.name,
                c.train.batch,
                d.train.len()
            )
            .into());
        }
    }
    Ok(cache)
}

fn prepare<T: Real>(d: &DataSpec) -> Res<Prepared<T>> {
    let full = Dataset::<T>::load(d, &d.train, None)?;
    let k = match full.task() {
        Task::Classification { n_classes } => Some(n_classes),
        Task::Regression => None,
    };
    let mut test = d
        .test
        .as_ref()
        .map(|s| Dataset::<T>::load(d, s, k))
        .transpose()?;

    let mut evals = d
        .eval
        .iter()
        .map(|(name, s)| Ok((name.clone(), Dataset::<T>::load(d, s, k)?)))
        .collect::<Res<Vec<_>>>()?;

    for (name, other) in test
        .iter()
        .map(|t| ("test", t))
        .chain(evals.iter().map(|(n, e)| (n.as_str(), e)))
    {
        if full.x().shape().cols() != other.x().shape().cols()
            || full.y().shape().cols() != other.y().shape().cols()
        {
            return Err(format!("train and {name} input/output dimensions differ").into());
        }
    }

    let (mut train, mut val) = if d.validation > 0.0 {
        let n = full.len();
        let n_val = (n as f64 * d.validation).round() as usize;
        if n_val == 0 || n_val >= n {
            return Err(format!(
                "validation = {} leaves {n_val} of {n} samples for validation",
                d.validation
            )
            .into());
        }
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut ChaCha8Rng::seed_from_u64(d.split_seed));
        (full.select(&idx[n_val..]), Some(full.select(&idx[..n_val])))
    } else {
        (full, None)
    };

    let raw_train = (train.x().shape().cols() <= PLOT_MAX_INPUTS).then(|| train.duplicate());
    let norm = train.fit_normalization(d.normalize);
    if let Some(test) = &mut test {
        test.apply_normalization(&norm);
    }
    if let Some(v) = val.as_mut() {
        v.apply_normalization(&norm);
    }
    for (_, e) in &mut evals {
        e.apply_normalization(&norm);
    }
    Ok(Prepared {
        raw_train,
        train,
        val,
        test,
        evals,
        norm,
    })
}

fn write_points<T: Real>(configs: &[&Config], cache: &Cache<T>) -> Res<()> {
    for c in configs {
        if let Some(raw) = &cache[&c.data.key()].raw_train {
            eval::write_points(c.dir().join("points_train.csv"), raw)?;
        }
    }
    Ok(())
}

fn run_jobs<T: Real>(configs: &[&Config], cache: &Cache<T>) -> Vec<String> {
    let jobs: Vec<(&Config, u64)> = configs
        .iter()
        .flat_map(|c| c.seeds.iter().map(move |&s| (*c, s)))
        .collect();

    jobs.par_iter()
        .filter_map(|&(c, seed)| {
            run_seed(c, &cache[&c.data.key()], seed)
                .err()
                .map(|e| format!("[{} seed={seed}] {e}", c.name))
        })
        .collect()
}

fn final_metrics<T: Real>(net: &Net<T>, data: &Dataset<T>, norm: &Normalization<T>) -> Value {
    let (pred, cost) = eval::predict_with_cost(net, data);
    match data.task() {
        Task::Regression => {
            let m = eval::regression_metrics_of(pred, data, norm);
            json!({ "cost": cost, "mse": m.mse, "mae": m.mae })
        }
        Task::Classification { .. } => {
            let m = eval::classification_metrics_of(pred.to_ref(), data);
            json!({ "cost": cost, "accuracy": m.accuracy, "macro_f1": m.macro_f1, "confusion": m.confusion })
        }
    }
}

fn run_seed<T: Real>(c: &Config, d: &Prepared<T>, seed: u64) -> Res<()> {
    let start = Instant::now();
    let dir = c.dir().join(format!("seed_{seed}"));
    fs::create_dir_all(&dir)?;

    let mut sizes = vec![d.train.x().shape().cols()];
    sizes.extend_from_slice(&c.net.hidden);
    sizes.push(d.train.y().shape().cols());

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut net: Net<T> =
        NeuralNetwork::new(&sizes, c.net.bias, c.net.activation, c.net.head, &mut rng);
    let cfg = c.train.to_train_config::<T>();
    let splits = d.splits();
    let weights_every = c.output.weights_every;

    let mut history = History::default();
    let mut weights = match weights_every {
        0 => None,
        _ => Some(WeightLog::create(&dir, &net)?),
    };
    // Observe the same training example each epoch, without SGD or RNG calls.
    let mut delta_probe = c.output.log_deltas.then(|| {
        (
            net.workspace(1),
            net.zero_grads(),
            net.layers()
                .iter()
                .map(|_| History::default())
                .collect::<Vec<_>>(),
        )
    });

    let outcome = train(&mut net, &d.train, &cfg, &mut rng, |epoch, net| {
        let mut row = vec![("epoch".to_string(), epoch as f64)];
        for (name, data) in &splits {
            for (metric, v) in eval::evaluate(net, data, &d.norm) {
                row.push((format!("{name}_{metric}"), v));
            }
        }
        let train_cost = row[1].1;
        history.push(row);
        if !train_cost.is_finite() {
            return Err(format!(
                "diverged at epoch {epoch} (train cost {train_cost})"
            ));
        }
        if let Some(w) = weights
            .as_mut()
            .filter(|_| epoch % weights_every == 0 || epoch == cfg.epochs)
        {
            w.record(epoch, net).map_err(|e| e.to_string())?;
        }
        if let Some((ws, grads, histories)) = delta_probe.as_mut() {
            let x = d.train.x().subrows(0..1);
            let y = d.train.y().subrows(0..1);
            net.forward(x, ws);
            net.backward(x, y, ws, grads);
            for (layer, history) in histories.iter_mut().enumerate() {
                let mut row = vec![("epoch".to_string(), epoch as f64)];
                for (neuron, delta) in ws.delta(layer).row(0).iter().enumerate() {
                    row.push((format!("delta_{neuron}"), delta.to_f64()));
                }
                history.push(row);
            }
        }
        Ok(())
    });

    if let Some(w) = weights.as_mut() {
        w.flush()?;
    }
    history.write_csv(dir.join("history.csv"))?;
    if let Some((_, _, histories)) = delta_probe {
        for (layer, history) in histories.iter().enumerate() {
            history.write_csv(dir.join(format!("deltas_layer{layer}.csv")))?;
        }
    }
    let seconds = start.elapsed().as_secs_f64();

    if let Err(e) = outcome {
        write_json(
            dir.join("metrics.json"),
            &json!({ "seed": seed, "seconds": seconds, "error": e }),
        )?;
        return Err(e.into());
    }

    let mut metrics = json!({ "seed": seed, "seconds": seconds, "epochs": cfg.epochs });
    for (name, data) in &splits {
        metrics[*name] = final_metrics(&net, data, &d.norm);
    }
    let evaluated = if d.test.is_some() { "test" } else { "val" };

    let summary = match c.data.task {
        TaskKind::Regression => {
            if let Some(test) = d.test.as_ref().filter(|_| d.train.x().shape().cols() == 1) {
                eval::write_regression(dir.join("pred.csv"), &net, test, &d.norm)?;
            }
            format!(
                "{evaluated} MSE {:.4} MAE {:.4}",
                metrics[evaluated]["mse"].as_f64().unwrap_or(f64::NAN),
                metrics[evaluated]["mae"].as_f64().unwrap_or(f64::NAN)
            )
        }
        TaskKind::Classification => {
            if let Some(test) = d
                .test
                .as_ref()
                .filter(|t| t.x().shape().cols() <= PLOT_MAX_INPUTS)
            {
                eval::write_classification(dir.join("pred.csv"), &net, test, &d.norm)?;
            }
            if let Some(raw) = d.raw_train.as_ref().filter(|r| r.x().shape().cols() == 2) {
                eval::write_decision_grid(
                    dir.join("grid.csv"),
                    &net,
                    raw,
                    &d.norm,
                    c.output.grid_steps,
                )?;
            }
            format!(
                "{evaluated} acc {:.4} F1 {:.4}",
                metrics[evaluated]["accuracy"].as_f64().unwrap_or(f64::NAN),
                metrics[evaluated]["macro_f1"].as_f64().unwrap_or(f64::NAN)
            )
        }
    };

    // Written last: its presence means predictions and weight logs also finished.
    write_json(dir.join("metrics.json"), &metrics)?;
    println!("[{} seed={seed}] {summary} ({seconds:.1}s)", c.name);
    Ok(())
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Res<()> {
    let w = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(w, value)?;
    Ok(())
}
