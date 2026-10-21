use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::ops::Range;
use std::path::Path;

use crate::data::{Dataset, Normalization, Task};
use crate::math_utils::Scalar;
use crate::math_utils::matrix::layout::Shape;
use crate::math_utils::matrix::{Mat, MatRef};
use crate::neural_network::activation::Activation;
use crate::neural_network::head::Head;
use crate::neural_network::{NeuralNetwork, Workspace};

const EVAL_CHUNK: usize = 2048;

fn forward_chunks<T, H, O>(
    net: &NeuralNetwork<T, H, O>,
    x: MatRef<'_, T>,
    mut f: impl FnMut(Range<usize>, &Workspace<T>),
) where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    let n = x.shape().rows();
    let chunk = n.clamp(1, EVAL_CHUNK);
    let mut ws = net.workspace(chunk);
    let mut start = 0;
    while start < n {
        let end = (start + chunk).min(n);
        if end - start != chunk {
            ws = net.workspace(end - start);
        }
        net.forward(x.subrows(start..end), &mut ws);
        f(start..end, &ws);
        start = end;
    }
}

pub fn predict<T, H, O>(net: &NeuralNetwork<T, H, O>, x: MatRef<'_, T>) -> Mat<T>
where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    let n_out = net.layers().last().map_or(0, |l| l.params().fan_out());
    let mut pred = Mat::zeros(Shape::new(x.shape().rows(), n_out));
    let mut p = pred.to_ref_mut();
    forward_chunks(net, x, |range, ws| {
        for (i, row) in range.zip(net.output(ws).row_iterator()) {
            p.row_mut(i).copy_from_slice(row);
        }
    });
    pred
}

/// Cost of the head (normalized units for regression) on a whole dataset.
pub fn cost<T, H, O>(net: &NeuralNetwork<T, H, O>, data: &Dataset<T>) -> f64
where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    let mut total = 0.0;
    forward_chunks(net, data.x(), |range, ws| {
        let rows = range.len() as f64;
        total += net.cost(ws, data.y().subrows(range)) * rows;
    });
    total / data.len().max(1) as f64
}

pub fn argmax_rows<T: Scalar>(m: MatRef<'_, T>) -> Vec<usize> {
    m.row_iterator()
        .map(|r| {
            r.iter()
                .enumerate()
                .fold(
                    (0, r[0]),
                    |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
                )
                .0
        })
        .collect()
}

pub struct RegressionMetrics {
    pub mse: f64,
    pub mae: f64,
}

pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub macro_f1: f64,
    /// confusion[true][predicted]
    pub confusion: Vec<Vec<usize>>,
}

/// Predictions and head cost in a single chunked pass.
pub fn predict_with_cost<T, H, O>(net: &NeuralNetwork<T, H, O>, data: &Dataset<T>) -> (Mat<T>, f64)
where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    let n_out = net.layers().last().map_or(0, |l| l.params().fan_out());
    let mut pred = Mat::zeros(Shape::new(data.len(), n_out));
    let mut total = 0.0;
    {
        let mut p = pred.to_ref_mut();
        forward_chunks(net, data.x(), |range, ws| {
            total += net.cost(ws, data.y().subrows(range.clone())) * range.len() as f64;
            for (i, row) in range.zip(net.output(ws).row_iterator()) {
                p.row_mut(i).copy_from_slice(row);
            }
        });
    }
    (pred, total / data.len().max(1) as f64)
}

/// MSE and MAE in original units: predictions and targets are de-normalized first.
pub fn regression_metrics_of<T: Scalar>(
    mut pred: Mat<T>,
    data: &Dataset<T>,
    norm: &Normalization<T>,
) -> RegressionMetrics {
    let mut truth = Mat::zeros(data.y().shape());
    data.y().copy_into(truth.to_ref_mut());
    norm.invert_y(pred.to_ref_mut());
    norm.invert_y(truth.to_ref_mut());

    let n = (truth.shape().rows() * truth.shape().cols()).max(1) as f64;
    let (se, ae) = pred
        .to_ref()
        .zip_fold(truth.to_ref(), (0.0, 0.0), |(se, ae), p, t| {
            let d = p.to_f64() - t.to_f64();
            (se + d * d, ae + d.abs())
        });

    RegressionMetrics {
        mse: se / n,
        mae: ae / n,
    }
}

pub fn classification_metrics_of<T: Scalar>(
    pred: MatRef<'_, T>,
    data: &Dataset<T>,
) -> ClassificationMetrics {
    let k = data.y().shape().cols();
    let pred = argmax_rows(pred);
    let truth = argmax_rows(data.y());

    let mut confusion = vec![vec![0usize; k]; k];
    for (&t, &p) in truth.iter().zip(&pred) {
        confusion[t][p] += 1;
    }

    let correct: usize = (0..k).map(|c| confusion[c][c]).sum();
    let f1_sum: f64 = (0..k)
        .map(|c| {
            let tp = confusion[c][c] as f64;
            let predicted: usize = (0..k).map(|t| confusion[t][c]).sum();
            let actual: usize = confusion[c].iter().sum();
            if predicted == 0 || actual == 0 {
                return 0.0;
            }
            let precision = tp / predicted as f64;
            let recall = tp / actual as f64;
            if precision + recall == 0.0 {
                0.0
            } else {
                2.0 * precision * recall / (precision + recall)
            }
        })
        .sum();

    ClassificationMetrics {
        accuracy: correct as f64 / truth.len().max(1) as f64,
        macro_f1: f1_sum / k as f64,
        confusion,
    }
}

pub fn regression_metrics<T, H, O>(
    net: &NeuralNetwork<T, H, O>,
    data: &Dataset<T>,
    norm: &Normalization<T>,
) -> RegressionMetrics
where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    regression_metrics_of(predict(net, data.x()), data, norm)
}

pub fn classification_metrics<T, H, O>(
    net: &NeuralNetwork<T, H, O>,
    data: &Dataset<T>,
) -> ClassificationMetrics
where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    classification_metrics_of(predict(net, data.x()).to_ref(), data)
}

/// Cost plus task metrics (regression in original units) from one forward pass.
pub fn evaluate<T, H, O>(
    net: &NeuralNetwork<T, H, O>,
    data: &Dataset<T>,
    norm: &Normalization<T>,
) -> Vec<(&'static str, f64)>
where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    let (pred, cost) = predict_with_cost(net, data);
    match data.task() {
        Task::Regression => {
            let m = regression_metrics_of(pred, data, norm);
            vec![("cost", cost), ("mse", m.mse), ("mae", m.mae)]
        }
        Task::Classification { .. } => {
            let m = classification_metrics_of(pred.to_ref(), data);
            vec![
                ("cost", cost),
                ("accuracy", m.accuracy),
                ("macro_f1", m.macro_f1),
            ]
        }
    }
}

/// Table with one row per log point; the first row fixes the columns.
#[derive(Default)]
pub struct History {
    columns: Vec<String>,
    rows: Vec<Vec<f64>>,
}

impl History {
    pub fn push(&mut self, row: Vec<(String, f64)>) {
        if self.columns.is_empty() {
            self.columns = row.iter().map(|(k, _)| k.clone()).collect();
        }
        debug_assert!(
            row.iter().map(|(k, _)| k).eq(self.columns.iter()),
            "history columns changed"
        );
        self.rows.push(row.into_iter().map(|(_, v)| v).collect());
    }

    pub fn last(&self, column: &str) -> Option<f64> {
        let i = self.columns.iter().position(|c| c == column)?;
        self.rows.last().map(|r| r[i])
    }

    pub fn write_csv(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "{}", self.columns.join(","))?;
        for r in &self.rows {
            let line: Vec<String> = r.iter().map(|v| v.to_string()).collect();
            writeln!(w, "{}", line.join(","))?;
        }
        w.flush()
    }
}

/// One file per layer, `weights_layer{l}.csv`: a row per recorded epoch with
/// `epoch` followed by W row-major (`w{i}_{j}`), then the bias as row i = fan_in.
pub struct WeightLog {
    files: Vec<BufWriter<File>>,
}

impl WeightLog {
    pub fn create<T, H, O>(dir: impl AsRef<Path>, net: &NeuralNetwork<T, H, O>) -> io::Result<Self>
    where
        T: Scalar,
        H: Activation<T>,
        O: Head<T>,
    {
        let files = net
            .layers()
            .iter()
            .enumerate()
            .map(|(l, layer)| {
                let p = layer.params();
                let rows = p.fan_in() + usize::from(p.b().is_some());
                let mut w = BufWriter::new(File::create(
                    dir.as_ref().join(format!("weights_layer{l}.csv")),
                )?);
                write!(w, "epoch")?;
                for i in 0..rows {
                    for j in 0..p.fan_out() {
                        write!(w, ",w{i}_{j}")?;
                    }
                }
                writeln!(w)?;
                Ok(w)
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self { files })
    }

    pub fn record<T, H, O>(&mut self, epoch: usize, net: &NeuralNetwork<T, H, O>) -> io::Result<()>
    where
        T: Scalar,
        H: Activation<T>,
        O: Head<T>,
    {
        for (w, layer) in self.files.iter_mut().zip(net.layers()) {
            let p = layer.params();
            write!(w, "{epoch}")?;
            for row in p.w().row_iterator().chain(p.b().map(|b| b.row(0))) {
                for v in row {
                    write!(w, ",{}", v.to_f64())?;
                }
            }
            writeln!(w)?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.files.iter_mut().try_for_each(|w| w.flush())
    }
}

/// Regression: rows `x,y_true,y_pred` in original units (single input column).
pub fn write_regression<T, H, O>(
    path: impl AsRef<Path>,
    net: &NeuralNetwork<T, H, O>,
    data: &Dataset<T>,
    norm: &Normalization<T>,
) -> io::Result<()>
where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    let mut pred = predict(net, data.x());
    let mut truth = Mat::zeros(data.y().shape());
    data.y().copy_into(truth.to_ref_mut());
    let mut x = Mat::zeros(data.x().shape());
    data.x().copy_into(x.to_ref_mut());
    norm.invert_y(pred.to_ref_mut());
    norm.invert_y(truth.to_ref_mut());
    norm.invert_x(x.to_ref_mut());

    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "x,y_true,y_pred")?;
    for ((xr, tr), pr) in x
        .row_iterator()
        .zip(truth.row_iterator())
        .zip(pred.row_iterator())
    {
        writeln!(
            w,
            "{},{},{}",
            xr[0].to_f64(),
            tr[0].to_f64(),
            pr[0].to_f64()
        )?;
    }
    w.flush()
}

/// Classification in 2D: predicted class (1-based) on a `steps × steps` grid
/// covering the data range (original units) with a 10% margin.
pub fn write_decision_grid<T, H, O>(
    path: impl AsRef<Path>,
    net: &NeuralNetwork<T, H, O>,
    raw: &Dataset<T>,
    norm: &Normalization<T>,
    steps: usize,
) -> io::Result<()>
where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    assert!(matches!(raw.task(), Task::Classification { .. }));
    assert_eq!(raw.x().shape().cols(), 2, "decision grid needs 2D input");

    let range = |c: usize| {
        let (lo, hi) = raw
            .x()
            .row_iterator()
            .fold((f64::MAX, f64::MIN), |(lo, hi), r| {
                let v = r[c].to_f64();
                (lo.min(v), hi.max(v))
            });
        let m = 0.1 * (hi - lo);
        (lo - m, hi + m)
    };
    let (x0, x1) = range(0);
    let (y0, y1) = range(1);
    let at = |lo: f64, hi: f64, i: usize| lo + (hi - lo) * i as f64 / (steps - 1) as f64;

    let mut grid = Mat::<T>::zeros(Shape::new(steps * steps, 2));
    {
        let mut g = grid.to_ref_mut();
        for i in 0..steps {
            for j in 0..steps {
                let r = g.row_mut(i * steps + j);
                r[0] = T::from_f64(at(x0, x1, j));
                r[1] = T::from_f64(at(y0, y1, i));
            }
        }
    }
    let mut norm_grid = Mat::zeros(grid.shape());
    grid.to_ref().copy_into(norm_grid.to_ref_mut());
    norm.apply_x(norm_grid.to_ref_mut());

    let classes = argmax_rows(predict(net, norm_grid.to_ref()).to_ref());

    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "x1,x2,class")?;
    for (r, c) in grid.row_iterator().zip(classes) {
        writeln!(w, "{},{},{}", r[0].to_f64(), r[1].to_f64(), c + 1)?;
    }
    w.flush()
}

/// Raw points (original units) for scatter plots: `x1,..,xn,label`.
pub fn write_points<T: Scalar>(path: impl AsRef<Path>, raw: &Dataset<T>) -> io::Result<()> {
    let n_in = raw.x().shape().cols();
    let labels: Vec<f64> = match raw.task() {
        Task::Regression => raw.y().row_iterator().map(|r| r[0].to_f64()).collect(),
        Task::Classification { .. } => argmax_rows(raw.y())
            .into_iter()
            .map(|c| (c + 1) as f64)
            .collect(),
    };
    let mut w = BufWriter::new(File::create(path)?);
    let header: Vec<String> = (1..=n_in).map(|i| format!("x{i}")).collect();
    writeln!(w, "{},label", header.join(","))?;
    for (r, l) in raw.x().row_iterator().zip(labels) {
        let xs: Vec<String> = r.iter().map(|v| v.to_f64().to_string()).collect();
        writeln!(w, "{},{l}", xs.join(","))?;
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HeadKind, HiddenKind};
    use crate::data::{DataSpec, LabelColumn, Normalize, Source, TaskKind};
    use rand::SeedableRng;

    #[test]
    fn chunked_eval_matches_single_pass() {
        let n = 2 * EVAL_CHUNK + 37;
        let body: String = (0..n)
            .map(|i| {
                format!(
                    "{},{},{}\n",
                    (i as f64).sin(),
                    (i as f64 * 0.3).cos(),
                    i % 3 + 1
                )
            })
            .collect();
        let p = std::env::temp_dir().join("nn_chunk.csv");
        std::fs::write(&p, body).unwrap();
        let spec = DataSpec {
            task: TaskKind::Classification,
            train: Source::File(p.clone()),
            test: Some(Source::File(p.clone())),
            format: None,
            normalize: Normalize::None,
            first_label: Some(1),
            label_column: LabelColumn::Last,
            validation: 0.0,
            split_seed: 0,
        };
        let data = Dataset::<f64>::load(&spec, &Source::File(p), None).unwrap();

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let net = NeuralNetwork::new(
            &[2, 7, 3],
            true,
            HiddenKind::Tanh,
            HeadKind::SoftmaxCrossEntropy,
            &mut rng,
        );

        let mut ws = net.workspace(n);
        net.forward(data.x(), &mut ws);
        let full = net.cost(&ws, data.y());
        assert!((cost(&net, &data) - full).abs() < 1e-12);
        let (_, c2) = predict_with_cost(&net, &data);
        assert!((c2 - full).abs() < 1e-12);

        let pred = predict(&net, data.x());
        let diff = pred
            .to_ref()
            .zip_fold(net.output(&ws), 0f64, |m, a, b| m.max((a - b).abs()));
        assert_eq!(diff, 0.0);
    }
}
