use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::data::{Dataset, Normalization, Task};
use crate::math_utils::Scalar;
use crate::math_utils::matrix::layout::Shape;
use crate::math_utils::matrix::{Mat, MatRef};
use crate::neural_network::NeuralNetwork;
use crate::neural_network::activation::Activation;
use crate::neural_network::head::Head;

pub fn predict<T, H, O>(net: &NeuralNetwork<T, H, O>, x: MatRef<'_, T>) -> Mat<T>
where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    let mut ws = net.workspace(x.shape().rows());
    net.forward(x, &mut ws);
    let out = net.output(&ws);
    let mut pred = Mat::zeros(out.shape());
    out.copy_into(pred.to_ref_mut());
    pred
}

pub fn cost<T, H, O>(net: &NeuralNetwork<T, H, O>, data: &Dataset<T>) -> f64
where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    let mut ws = net.workspace(data.len());
    net.forward(data.x(), &mut ws);
    net.cost(&ws, data.y())
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
    pub confusion: Vec<Vec<usize>>,
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
    let mut pred = predict(net, data.x());
    let mut truth = Mat::zeros(data.y().shape());
    data.y().copy_into(truth.to_ref_mut());
    norm.invert_y(pred.to_ref_mut());
    norm.invert_y(truth.to_ref_mut());

    let n = (truth.shape().rows() * truth.shape().cols()) as f64;
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

pub fn classification_metrics<T, H, O>(
    net: &NeuralNetwork<T, H, O>,
    data: &Dataset<T>,
) -> ClassificationMetrics
where
    T: Scalar,
    H: Activation<T>,
    O: Head<T>,
{
    let k = data.y().shape().cols();
    let pred = argmax_rows(predict(net, data.x()).to_ref());
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

#[derive(Default)]
pub struct History {
    pub epoch: Vec<usize>,
    pub train: Vec<f64>,
    pub test: Vec<f64>,
}

impl History {
    pub fn push(&mut self, epoch: usize, train: f64, test: f64) {
        self.epoch.push(epoch);
        self.train.push(train);
        self.test.push(test);
    }

    pub fn write_csv(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "epoch,train,test")?;
        for i in 0..self.epoch.len() {
            writeln!(w, "{},{},{}", self.epoch[i], self.train[i], self.test[i])?;
        }
        w.flush()
    }
}

pub struct WeightLog {
    w: BufWriter<File>,
}

impl WeightLog {
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "epoch,layer,i,j,w")?;
        Ok(Self { w })
    }

    pub fn record<T, H, O>(&mut self, epoch: usize, net: &NeuralNetwork<T, H, O>) -> io::Result<()>
    where
        T: Scalar,
        H: Activation<T>,
        O: Head<T>,
    {
        for (l, layer) in net.layers().iter().enumerate() {
            let p = layer.params();
            for (i, row) in p.w().row_iterator().enumerate() {
                for (j, v) in row.iter().enumerate() {
                    writeln!(self.w, "{epoch},{l},{i},{j},{}", v.to_f64())?;
                }
            }
            if let Some(b) = p.b() {
                let i = p.fan_in();
                for (j, v) in b.row(0).iter().enumerate() {
                    writeln!(self.w, "{epoch},{l},{i},{j},{}", v.to_f64())?;
                }
            }
        }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.w.flush()
    }
}

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
