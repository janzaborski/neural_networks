use std::error::Error;
use std::fs;
use std::path::Path;

use crate::math_utils::matrix::layout::Shape;
use crate::math_utils::matrix::{Mat, MatRef, MatRefMut};
use crate::math_utils::{Real, Scalar};

type Rows = Vec<Vec<f64>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Regression,
    Classification { n_classes: usize },
}

pub struct Dataset<T> {
    x: Mat<T>,
    y: Mat<T>,
    task: Task,
}

impl<T: Scalar> Dataset<T> {
    pub fn load_regression(path: impl AsRef<Path>) -> Result<Self, Box<dyn Error>> {
        let (rows, n_cols) = read_csv(path.as_ref())?;
        let n_in = n_cols - 1;

        let mut x = Vec::with_capacity(rows.len() * n_in);
        let mut y = Vec::with_capacity(rows.len());
        for r in &rows {
            x.extend(r[..n_in].iter().map(|&v| T::from_f64(v)));
            y.push(T::from_f64(r[n_in]));
        }

        Ok(Self {
            x: Mat::from_vec(x, Shape::new(rows.len(), n_in)),
            y: Mat::from_vec(y, Shape::new(rows.len(), 1)),
            task: Task::Regression,
        })
    }

    pub fn load_classification(
        path: impl AsRef<Path>,
        n_classes: Option<usize>,
    ) -> Result<Self, Box<dyn Error>> {
        let (rows, n_cols) = read_csv(path.as_ref())?;
        let n_in = n_cols - 1;

        let labels = rows
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let c = r[n_in];
                if c < 1.0 || c.fract() != 0.0 {
                    return Err(format!("row {}: invalid class label {c}", i + 2));
                }
                Ok(c as usize - 1)
            })
            .collect::<Result<Vec<usize>, String>>()?;

        let found = labels.iter().max().map_or(0, |&m| m + 1);
        let k = n_classes.unwrap_or(found);
        if found > k {
            return Err(format!("found class {found}, but n_classes = {k}").into());
        }

        let mut x = Vec::with_capacity(rows.len() * n_in);
        let mut y = vec![T::ZERO; rows.len() * k];
        for (i, (r, &c)) in rows.iter().zip(&labels).enumerate() {
            x.extend(r[..n_in].iter().map(|&v| T::from_f64(v)));
            y[i * k + c] = T::ONE;
        }

        Ok(Self {
            x: Mat::from_vec(x, Shape::new(rows.len(), n_in)),
            y: Mat::from_vec(y, Shape::new(rows.len(), k)),
            task: Task::Classification { n_classes: k },
        })
    }

    pub fn x(&self) -> MatRef<'_, T> {
        self.x.to_ref()
    }

    pub fn y(&self) -> MatRef<'_, T> {
        self.y.to_ref()
    }

    pub fn x_mut(&mut self) -> &mut Mat<T> {
        &mut self.x
    }

    pub fn y_mut(&mut self) -> &mut Mat<T> {
        &mut self.y
    }

    pub fn len(&self) -> usize {
        self.x.shape().rows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn task(&self) -> Task {
        self.task
    }

    pub fn apply_normalization(&mut self, n: &Normalization<T>) {
        debug_assert!(
            self.task == n.task,
            "normalization fitted on a different task"
        );

        self.x.to_ref_mut().apply_normalization(&n.x_mean, &n.x_std);
        if let Some((mean, std)) = &n.y {
            self.y.to_ref_mut().apply_normalization(mean, std);
        }
    }
}

impl<T: Real> Dataset<T> {
    pub fn fit_normalization(&mut self) -> Normalization<T> {
        let (x_mean, x_std) = column_stats(self.x.to_ref());
        let y = match self.task {
            Task::Regression => Some(column_stats(self.y.to_ref())),
            Task::Classification { .. } => None,
        };

        let n = Normalization {
            task: self.task,
            x_mean,
            x_std,
            y,
        };
        self.apply_normalization(&n);
        n
    }
}

pub struct Normalization<T> {
    task: Task,
    x_mean: Vec<T>,
    x_std: Vec<T>,
    y: Option<(Vec<T>, Vec<T>)>,
}

impl<T: Scalar> Normalization<T> {
    pub fn apply_x(&self, mut x: MatRefMut<'_, T>) {
        x.apply_normalization(&self.x_mean, &self.x_std);
    }

    pub fn invert_x(&self, mut x: MatRefMut<'_, T>) {
        x.invert_normalization(&self.x_mean, &self.x_std);
    }

    pub fn invert_y(&self, mut y: MatRefMut<'_, T>) {
        if let Some((mean, std)) = &self.y {
            y.invert_normalization(mean, std);
        }
    }

    pub fn y_std(&self) -> Option<&[T]> {
        self.y.as_ref().map(|(_, std)| std.as_slice())
    }
}

fn column_stats<T: Real>(m: MatRef<'_, T>) -> (Vec<T>, Vec<T>) {
    let (mean, mut std) = m.get_means_stdevs();
    let eps = T::from_f64(1e-12);
    std.iter_mut()
        .filter(|s| **s < eps)
        .for_each(|s| *s = T::ONE);
    (mean, std)
}

fn read_csv(path: &Path) -> Result<(Rows, usize), Box<dyn Error>> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = text.lines();

    let header = lines
        .next()
        .ok_or_else(|| format!("{}: empty file", path.display()))?;
    let n_cols = header.split(',').count();
    if n_cols < 2 {
        return Err(format!("{}: expected at least 2 columns", path.display()).into());
    }

    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row = line
            .split(',')
            .map(|v| v.trim().parse::<f64>())
            .collect::<Result<Vec<f64>, _>>()
            .map_err(|e| format!("{}:{}: {e}", path.display(), i + 2))?;
        if row.len() != n_cols {
            return Err(format!(
                "{}:{}: expected {n_cols} columns, got {}",
                path.display(),
                i + 2,
                row.len()
            )
            .into());
        }
        rows.push(row);
    }

    Ok((rows, n_cols))
}
