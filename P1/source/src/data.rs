use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::math_utils::matrix::layout::Shape;
use crate::math_utils::matrix::{Mat, MatRef, MatRefMut};
use crate::math_utils::{Real, Scalar};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    Regression,
    Classification { n_classes: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskKind {
    Regression,
    Classification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileFormat {
    Csv,
    Idx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Normalize {
    None,
    #[default]
    PerColumn,
    Global,
    MinMax,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Source {
    File(PathBuf),
    Split(SplitSource),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitSource {
    pub x: PathBuf,
    pub y: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LabelColumn {
    First,
    #[default]
    Last,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataSpec {
    pub task: TaskKind,
    pub train: Source,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test: Option<Source>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<FileFormat>,
    #[serde(default)]
    pub normalize: Normalize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_label: Option<u32>,
    #[serde(default)]
    pub label_column: LabelColumn,
    #[serde(default)]
    pub validation: f64,
    #[serde(default)]
    pub split_seed: u64,
}

impl FileFormat {
    pub fn detect(path: &Path) -> Result<Self, String> {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if name.ends_with(".gz") {
            return Err(format!(
                "{}: compressed file, decompress it first (gunzip)",
                path.display()
            ));
        }
        if name.ends_with(".csv") {
            Ok(Self::Csv)
        } else if name.ends_with("-ubyte") || name.ends_with(".idx") || name.contains(".idx") {
            Ok(Self::Idx)
        } else {
            Err(format!(
                "{}: cannot detect format from file name, set \"format\": \"csv\" | \"idx\"",
                path.display()
            ))
        }
    }
}

impl Source {
    pub fn paths(&self) -> Vec<&Path> {
        match self {
            Self::File(p) => vec![p],
            Self::Split(s) => vec![&s.x, &s.y],
        }
    }

    fn target_path(&self) -> &Path {
        match self {
            Self::File(p) => p,
            Self::Split(s) => &s.y,
        }
    }
}

impl DataSpec {
    pub fn format_of(&self, path: &Path) -> Result<FileFormat, String> {
        match self.format {
            Some(f) => Ok(f),
            None => FileFormat::detect(path),
        }
    }

    pub fn key(&self) -> String {
        serde_json::to_string(self).expect("DataSpec is always serializable")
    }

    pub fn resolve(&mut self) -> Result<(), String> {
        for p in self
            .train
            .paths()
            .into_iter()
            .chain(self.test.iter().flat_map(Source::paths))
        {
            self.format_of(p)?;
        }
        if self.first_label.is_none() && self.task == TaskKind::Classification {
            let f = self.format_of(self.train.target_path())?;
            self.first_label = Some(match f {
                FileFormat::Idx => 0,
                FileFormat::Csv => 1,
            });
        }
        Ok(())
    }
}

pub struct Dataset<T> {
    x: Mat<T>,
    y: Mat<T>,
    task: Task,
}

impl<T: Scalar> Dataset<T> {
    pub fn load(
        spec: &DataSpec,
        source: &Source,
        n_classes: Option<usize>,
    ) -> Result<Self, String> {
        let (x, targets) = match source {
            Source::File(p) => {
                split_label_column(read_matrix(p, spec.format_of(p)?)?, spec.label_column, p)?
            }
            Source::Split(s) => (
                read_matrix(&s.x, spec.format_of(&s.x)?)?,
                read_matrix(&s.y, spec.format_of(&s.y)?)?,
            ),
        };

        if x.shape().rows() != targets.shape().rows() {
            return Err(format!(
                "{source:?}: {} input rows but {} target rows",
                x.shape().rows(),
                targets.shape().rows()
            ));
        }

        match spec.task {
            TaskKind::Regression => Ok(Self {
                x,
                y: targets,
                task: Task::Regression,
            }),
            TaskKind::Classification => {
                let first = spec.first_label.unwrap_or(1);
                let (y, k) = one_hot(targets.to_ref(), first, n_classes)
                    .map_err(|e| format!("{source:?}: {e}"))?;
                Ok(Self {
                    x,
                    y,
                    task: Task::Classification { n_classes: k },
                })
            }
        }
    }

    pub fn select(&self, rows: &[usize]) -> Self {
        let pick = |m: &Mat<T>| {
            let cols = m.shape().cols();
            let mut data = Vec::with_capacity(rows.len() * cols);
            rows.iter()
                .for_each(|&i| data.extend_from_slice(m.to_ref().row(i)));
            Mat::from_vec(data, Shape::new(rows.len(), cols))
        };
        Self {
            x: pick(&self.x),
            y: pick(&self.y),
            task: self.task,
        }
    }

    pub fn duplicate(&self) -> Self {
        let copy = |m: &Mat<T>| {
            let mut c = Mat::zeros(m.shape());
            m.to_ref().copy_into(c.to_ref_mut());
            c
        };
        Self {
            x: copy(&self.x),
            y: copy(&self.y),
            task: self.task,
        }
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
    pub fn fit_normalization(&mut self, mode: Normalize) -> Normalization<T> {
        let cols = self.x.shape().cols();
        let (x_mean, x_std) = match mode {
            Normalize::None => (vec![T::ZERO; cols], vec![T::ONE; cols]),
            Normalize::PerColumn => column_stats(self.x.to_ref()),
            Normalize::Global => {
                let (m, s) = global_mean_std(self.x.to_ref());
                (vec![m; cols], vec![s; cols])
            }
            Normalize::MinMax => {
                let (lo, range) = global_min_range(self.x.to_ref());
                (vec![lo; cols], vec![range; cols])
            }
        };
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

const STD_EPS: f64 = 1e-12;

fn column_stats<T: Real>(m: MatRef<'_, T>) -> (Vec<T>, Vec<T>) {
    let (mean, mut std) = m.get_means_stdevs();
    let eps = T::from_f64(STD_EPS);
    std.iter_mut()
        .filter(|s| **s < eps)
        .for_each(|s| *s = T::ONE);
    (mean, std)
}

fn global_mean_std<T: Scalar>(m: MatRef<'_, T>) -> (T, T) {
    let n = (m.shape().rows() * m.shape().cols()).max(1) as f64;
    let mean = m.fold(0f64, |s, x| s + x.to_f64()) / n;
    let var = m.fold(0f64, |s, x| s + (x.to_f64() - mean).powi(2)) / n;
    let std = var.sqrt();
    (
        T::from_f64(mean),
        T::from_f64(if std < STD_EPS { 1.0 } else { std }),
    )
}

fn global_min_range<T: Scalar>(m: MatRef<'_, T>) -> (T, T) {
    let (lo, hi) = m.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
        let v = x.to_f64();
        (lo.min(v), hi.max(v))
    });
    if !lo.is_finite() {
        return (T::ZERO, T::ONE);
    }
    let range = hi - lo;
    (
        T::from_f64(lo),
        T::from_f64(if range < STD_EPS { 1.0 } else { range }),
    )
}

fn one_hot<T: Scalar>(
    labels: MatRef<'_, T>,
    first: u32,
    n_classes: Option<usize>,
) -> Result<(Mat<T>, usize), String> {
    if labels.shape().cols() != 1 {
        return Err(format!(
            "classification needs exactly 1 label column, got {}",
            labels.shape().cols()
        ));
    }

    let idx = labels
        .row_iterator()
        .enumerate()
        .map(|(i, r)| {
            let v = r[0].to_f64() - first as f64;
            if v < 0.0 || v.fract() != 0.0 {
                return Err(format!(
                    "row {i}: invalid label {} (first_label = {first})",
                    r[0].to_f64()
                ));
            }
            Ok(v as usize)
        })
        .collect::<Result<Vec<usize>, String>>()?;

    let found = idx.iter().max().map_or(0, |&m| m + 1);
    let k = n_classes.unwrap_or(found);
    if found > k {
        return Err(format!("found {found} classes, expected {k}"));
    }

    let mut y = vec![T::ZERO; idx.len() * k];
    for (i, &c) in idx.iter().enumerate() {
        y[i * k + c] = T::ONE;
    }
    Ok((Mat::from_vec(y, Shape::new(idx.len(), k)), k))
}

fn split_label_column<T: Scalar>(
    m: Mat<T>,
    column: LabelColumn,
    path: &Path,
) -> Result<(Mat<T>, Mat<T>), String> {
    let (rows, cols) = (m.shape().rows(), m.shape().cols());
    if cols < 2 {
        return Err(format!(
            "{}: expected at least 2 columns (inputs + target), got {cols}",
            path.display()
        ));
    }
    let (label, inputs) = match column {
        LabelColumn::First => (0, 1..cols),
        LabelColumn::Last => (cols - 1, 0..cols - 1),
    };
    let mut x = Vec::with_capacity(rows * (cols - 1));
    let mut y = Vec::with_capacity(rows);
    for r in m.row_iterator() {
        x.extend_from_slice(&r[inputs.clone()]);
        y.push(r[label]);
    }
    Ok((
        Mat::from_vec(x, Shape::new(rows, cols - 1)),
        Mat::from_vec(y, Shape::new(rows, 1)),
    ))
}

pub fn read_matrix<T: Scalar>(path: &Path, format: FileFormat) -> Result<Mat<T>, String> {
    match format {
        FileFormat::Csv => read_csv(path),
        FileFormat::Idx => read_idx(path),
    }
}

fn read_csv<T: Scalar>(path: &Path) -> Result<Mat<T>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let parse = |line: &str| -> Result<Vec<T>, std::num::ParseFloatError> {
        line.split(',')
            .map(|v| v.trim().parse::<f64>().map(T::from_f64))
            .collect()
    };

    let mut lines = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .peekable();
    let (_, first) = *lines
        .peek()
        .ok_or_else(|| format!("{}: empty file", path.display()))?;
    let cols = first.split(',').count();
    if parse(first).is_err() {
        lines.next();
    }

    let mut data = Vec::new();
    for (i, line) in lines {
        let row = parse(line).map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
        if row.len() != cols {
            return Err(format!(
                "{}:{}: expected {cols} columns, got {}",
                path.display(),
                i + 1,
                row.len()
            ));
        }
        data.extend(row);
    }

    let rows = data.len() / cols;
    if rows == 0 {
        return Err(format!("{}: no data rows", path.display()));
    }
    Ok(Mat::from_vec(data, Shape::new(rows, cols)))
}

fn read_idx<T: Scalar>(path: &Path) -> Result<Mat<T>, String> {
    let err = |msg: String| format!("{}: {msg}", path.display());
    let bytes = fs::read(path).map_err(|e| err(e.to_string()))?;

    if bytes.len() < 4 || bytes[0] != 0 || bytes[1] != 0 {
        return Err(err("not an IDX file (bad magic number)".into()));
    }
    let (size, decode): (usize, fn(&[u8]) -> f64) = match bytes[2] {
        0x08 => (1, |b| b[0] as f64),
        0x09 => (1, |b| b[0] as i8 as f64),
        0x0B => (2, |b| i16::from_be_bytes([b[0], b[1]]) as f64),
        0x0C => (4, |b| i32::from_be_bytes([b[0], b[1], b[2], b[3]]) as f64),
        0x0D => (4, |b| f32::from_be_bytes([b[0], b[1], b[2], b[3]]) as f64),
        0x0E => (8, |b| f64::from_be_bytes(b.try_into().unwrap())),
        t => return Err(err(format!("unknown IDX data type 0x{t:02X}"))),
    };
    let ndims = bytes[3] as usize;
    let header = 4 + 4 * ndims;
    if ndims == 0 || bytes.len() < header {
        return Err(err("truncated IDX header".into()));
    }

    let dims: Vec<usize> = bytes[4..header]
        .chunks_exact(4)
        .map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect();
    let rows = dims[0];
    let cols: usize = dims[1..].iter().product();
    let expected = header + rows * cols * size;
    if bytes.len() != expected {
        return Err(err(format!(
            "dims {dims:?} need {expected} bytes, file has {}",
            bytes.len()
        )));
    }

    let data = bytes[header..]
        .chunks_exact(size)
        .map(|b| T::from_f64(decode(b)))
        .collect();
    Ok(Mat::from_vec(data, Shape::new(rows, cols.max(1))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str, body: &[u8]) -> PathBuf {
        let p = std::env::temp_dir().join(name);
        fs::write(&p, body).unwrap();
        p
    }

    fn spec(task: TaskKind, train: Source, normalize: Normalize) -> DataSpec {
        let mut s = DataSpec {
            task,
            test: Some(train.clone()),
            train,
            format: None,
            normalize,
            first_label: None,
            label_column: LabelColumn::Last,
            validation: 0.0,
            split_seed: 0,
        };
        s.resolve().unwrap();
        s
    }

    fn idx(dtype: u8, dims: &[u32], payload: &[u8]) -> Vec<u8> {
        let mut b = vec![0, 0, dtype, dims.len() as u8];
        dims.iter().for_each(|d| b.extend(d.to_be_bytes()));
        b.extend(payload);
        b
    }

    #[test]
    fn regression_normalization_roundtrip() {
        let tr = tmp("nn_n_tr.csv", b"x,y\n-1,-28\n0,-8\n1,7\n2,0\n");
        let te = tmp("nn_n_te.csv", b"x,y\n-5,-1253\n5,87\n");
        let s = spec(
            TaskKind::Regression,
            Source::File(tr.clone()),
            Normalize::PerColumn,
        );
        let mut train = Dataset::<f64>::load(&s, &Source::File(tr), None).unwrap();
        let mut test = Dataset::<f64>::load(&s, &Source::File(te), None).unwrap();

        let n = train.fit_normalization(Normalize::PerColumn);
        let (mx, sx) = train.x().get_means_stdevs();
        let (my, sy) = train.y().get_means_stdevs();
        assert!(mx[0].abs() < 1e-12 && (sx[0] - 1.0).abs() < 1e-12);
        assert!(my[0].abs() < 1e-12 && (sy[0] - 1.0).abs() < 1e-12);

        test.apply_normalization(&n);
        let mut pred = Mat::from_vec(
            test.y().row_iterator().map(|r| r[0]).collect(),
            Shape::new(2, 1),
        );
        n.invert_y(pred.to_ref_mut());
        let p = pred.as_contiguous_slice().unwrap();
        assert!((p[0] + 1253.0).abs() < 1e-9 && (p[1] - 87.0).abs() < 1e-9);
    }

    #[test]
    fn classification_csv_header_optional() {
        let with = tmp("nn_c_h.csv", b"x,y,cls\n0.5,3,1\n-0.5,3,2\n1.5,3,1\n");
        let without = tmp("nn_c_nh.csv", b"0.5,3,1\n-0.5,3,2\n1.5,3,1\n");
        for p in [with, without] {
            let s = spec(
                TaskKind::Classification,
                Source::File(p.clone()),
                Normalize::PerColumn,
            );
            let mut d = Dataset::<f64>::load(&s, &Source::File(p), None).unwrap();
            d.fit_normalization(Normalize::PerColumn);
            assert_eq!(d.task(), Task::Classification { n_classes: 2 });
            assert_eq!(d.y().row(1), &[0.0, 1.0]);
            assert_eq!(d.x().row(0)[1], 0.0);
        }
    }

    #[test]
    fn idx_split_files() {
        let pixels: Vec<u8> = (0..3 * 2 * 2).map(|i| (i * 20) as u8).collect();
        let xi = tmp("nn-images-idx3-ubyte", &idx(0x08, &[3, 2, 2], &pixels));
        let yi = tmp("nn-labels-idx1-ubyte", &idx(0x08, &[3], &[0, 9, 4]));
        let src = Source::Split(SplitSource { x: xi, y: yi });
        let s = spec(TaskKind::Classification, src.clone(), Normalize::MinMax);
        assert_eq!(s.first_label, Some(0));

        let mut d = Dataset::<f32>::load(&s, &src, None).unwrap();
        assert_eq!(d.x().shape(), Shape::new(3, 4));
        assert_eq!(d.task(), Task::Classification { n_classes: 10 });
        assert_eq!(d.y().row(1)[9], 1.0);
        d.fit_normalization(Normalize::MinMax);
        assert_eq!(d.x().row(0)[0], 0.0);
        assert_eq!(d.x().row(2)[3], 1.0);
    }

    #[test]
    fn kaggle_csv_label_first_zero_based() {
        let p = tmp("nn_kaggle.csv", b"label,pixel0,pixel1\n0,0,255\n9,128,0\n");
        let mut s = spec(
            TaskKind::Classification,
            Source::File(p.clone()),
            Normalize::MinMax,
        );
        s.label_column = LabelColumn::First;
        s.first_label = Some(0);
        let d = Dataset::<f64>::load(&s, &Source::File(p), None).unwrap();
        assert_eq!(d.x().row(1), &[128.0, 0.0]);
        assert_eq!(d.task(), Task::Classification { n_classes: 10 });
        assert_eq!(d.y().row(0)[0], 1.0);
        let sub = d.select(&[1]);
        assert_eq!(sub.x().row(0), &[128.0, 0.0]);
        assert_eq!(sub.y().row(0)[9], 1.0);
    }

    #[test]
    fn global_standardization() {
        let p = tmp("nn_g.csv", b"a,b,c\n0,2,1\n4,6,1\n");
        let s = spec(
            TaskKind::Classification,
            Source::File(p.clone()),
            Normalize::Global,
        );
        let mut d = Dataset::<f64>::load(&s, &Source::File(p), None).unwrap();
        d.fit_normalization(Normalize::Global);
        let (sum, sq) = d.x().fold((0.0, 0.0), |(s, q), v| (s + v, q + v * v));
        assert!(sum.abs() < 1e-12 && (sq / 4.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn errors_are_readable() {
        assert!(
            FileFormat::detect(Path::new("a.gz"))
                .unwrap_err()
                .contains("gunzip")
        );
        assert!(
            FileFormat::detect(Path::new("data.bin"))
                .unwrap_err()
                .contains("format")
        );
        let bad = tmp("nn_bad-ubyte", &[0, 0, 8, 1, 0, 0, 0, 5, 1, 2]);
        assert!(read_idx::<f64>(&bad).err().unwrap().contains("need"));
    }
}
