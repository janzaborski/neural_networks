use std::collections::HashMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::{Value, json};

use neural_net::config::{
    Config, DataConfig, Dtype, HeadKind, HiddenKind, TaskKind, load_configs, validate_all,
};
use neural_net::data::{Dataset, Normalization, Task};
use neural_net::eval::{self, History, WeightLog};
use neural_net::math_utils::Real;
use neural_net::neural_network::NeuralNetwork;
use neural_net::training::train;

type Res<T> = Result<T, Box<dyn Error + Send + Sync>>;
type Net<T> = NeuralNetwork<T, HiddenKind, HeadKind>;

struct Prepared<T> {
    raw_train: Dataset<T>,
    train: Dataset<T>,
    test: Dataset<T>,
    norm: Normalization<T>,
}

type Cache<T> = HashMap<DataConfig, Prepared<T>>;

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
        if !cache.contains_key(&c.data) {
            let p = prepare::<T>(&c.data)?;
            cache.insert(c.data.clone(), p);
        }
        let d = &cache[&c.data];
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

fn prepare<T: Real>(d: &DataConfig) -> Res<Prepared<T>> {
    let load = |path: &Path, k: Option<usize>| -> Res<Dataset<T>> {
        let r = match d.task {
            TaskKind::Regression => Dataset::load_regression(path),
            TaskKind::Classification => Dataset::load_classification(path, k),
        };
        r.map_err(|e| format!("{}: {e}", path.display()).into())
    };

    let raw_train = load(&d.train, None)?;
    let k = match raw_train.task() {
        Task::Classification { n_classes } => Some(n_classes),
        Task::Regression => None,
    };
    let mut train = load(&d.train, k)?;
    let mut test = load(&d.test, k)?;

    if train.x().shape().cols() != test.x().shape().cols() {
        return Err(format!(
            "{} has {} input column(s), {} has {}",
            d.train.display(),
            train.x().shape().cols(),
            d.test.display(),
            test.x().shape().cols()
        )
        .into());
    }

    let norm = train.fit_normalization();
    test.apply_normalization(&norm);
    Ok(Prepared {
        raw_train,
        train,
        test,
        norm,
    })
}

fn write_points<T: Real>(configs: &[&Config], cache: &Cache<T>) -> Res<()> {
    for c in configs {
        eval::write_points(c.dir().join("points_train.csv"), &cache[&c.data].raw_train)?;
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
            run_seed(c, &cache[&c.data], seed)
                .err()
                .map(|e| format!("[{} seed={seed}] {e}", c.name))
        })
        .collect()
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

    let mut history = History::default();
    let mut weights = match c.output.log_weights {
        true => Some(WeightLog::create(dir.join("weights.csv"))?),
        false => None,
    };

    let (tr, te) = (&d.train, &d.test);
    let outcome = train(&mut net, tr, &cfg, &mut rng, |epoch, net| {
        let c_tr = eval::cost(net, tr);
        let c_te = eval::cost(net, te);
        history.push(epoch, c_tr, c_te);
        if !c_tr.is_finite() {
            return Err(format!("diverged at epoch {epoch} (train cost {c_tr})"));
        }
        if let Some(w) = weights.as_mut() {
            w.record(epoch, net).map_err(|e| e.to_string())?;
        }
        Ok(())
    });

    if let Some(w) = weights.as_mut() {
        w.flush()?;
    }
    history.write_csv(dir.join("loss.csv"))?;
    let seconds = start.elapsed().as_secs_f64();

    if let Err(e) = outcome {
        write_json(
            dir.join("metrics.json"),
            &json!({ "seed": seed, "seconds": seconds, "error": e }),
        )?;
        return Err(e.into());
    }

    let (m_train, m_test, summary) = match c.data.task {
        TaskKind::Regression => {
            let m_tr = eval::regression_metrics(&net, tr, &d.norm);
            let m_te = eval::regression_metrics(&net, te, &d.norm);
            if tr.x().shape().cols() == 1 {
                eval::write_regression(dir.join("pred.csv"), &net, te, &d.norm)?;
            }
            (
                json!({ "mse": m_tr.mse, "mae": m_tr.mae }),
                json!({ "mse": m_te.mse, "mae": m_te.mae }),
                format!("test MSE {:.4} MAE {:.4}", m_te.mse, m_te.mae),
            )
        }
        TaskKind::Classification => {
            let m_tr = eval::classification_metrics(&net, tr);
            let m_te = eval::classification_metrics(&net, te);
            if tr.x().shape().cols() == 2 {
                eval::write_decision_grid(
                    dir.join("grid.csv"),
                    &net,
                    &d.raw_train,
                    &d.norm,
                    c.output.grid_steps,
                )?;
            }
            (
                json!({ "accuracy": m_tr.accuracy, "macro_f1": m_tr.macro_f1, "confusion": m_tr.confusion }),
                json!({ "accuracy": m_te.accuracy, "macro_f1": m_te.macro_f1, "confusion": m_te.confusion }),
                format!("test acc {:.4} F1 {:.4}", m_te.accuracy, m_te.macro_f1),
            )
        }
    };

    let with_cost = |mut m: Value, cost: Option<&f64>| {
        m["cost"] = json!(cost);
        m
    };
    write_json(
        dir.join("metrics.json"),
        &json!({
            "seed": seed,
            "seconds": seconds,
            "train": with_cost(m_train, history.train.last()),
            "test": with_cost(m_test, history.test.last()),
        }),
    )?;

    println!("[{} seed={seed}] {summary} ({seconds:.1}s)", c.name);
    Ok(())
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Res<()> {
    let w = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(w, value)?;
    Ok(())
}
