use crate::math_utils::matrix::layout::Shape;
use crate::math_utils::matrix::{Mat, MatRef, MatRefMut};
use crate::math_utils::{Real, Scalar};

impl<T: Scalar> Mat<T> {
    pub fn scale_by(&mut self, a: T) {
        self.to_ref_mut().scale_by(a);
    }

    pub fn fill(&mut self, a: T) {
        self.to_ref_mut().fill(a);
    }

    pub fn add_to_every_row(&mut self, row_vec: MatRef<'_, T>) {
        self.to_ref_mut().add_to_every_row(row_vec);
    }

    pub fn sum_of_rows(&self, alpha: T, out: MatRefMut<'_, T>) {
        self.to_ref().sum_of_rows(alpha, out);
    }

    pub fn map<F>(&self, f: F, out: MatRefMut<'_, T>)
    where
        F: Fn(T) -> T + Sync,
    {
        self.to_ref().map(f, out);
    }

    pub fn map_in_place<F>(&mut self, f: F)
    where
        F: Fn(T) -> T + Sync,
    {
        self.to_ref_mut().map_in_place(f);
    }

    pub fn zip_map<F>(&self, other: MatRef<'_, T>, f: F, out: MatRefMut<'_, T>)
    where
        F: Fn(T, T) -> T + Sync,
    {
        self.to_ref().zip_map(other, f, out);
    }

    pub fn zip_map_in_place<F>(&mut self, other: MatRef<'_, T>, f: F)
    where
        F: Fn(T, T) -> T + Sync,
    {
        self.to_ref_mut().zip_map_in_place(other, f);
    }

    pub fn zip2_map_in_place<F>(&mut self, other1: MatRef<'_, T>, other2: MatRef<'_, T>, f: F)
    where
        F: Fn(T, T, T) -> T + Sync,
    {
        self.to_ref_mut().zip2_map_in_place(other1, other2, f);
    }

    pub fn fold<A, F>(&self, init: A, f: F) -> A
    where
        F: Fn(A, T) -> A,
    {
        self.to_ref().fold(init, f)
    }

    pub fn sum(&self) -> T {
        self.to_ref().sum()
    }

    pub fn zip_fold<A, F>(&self, other: MatRef<'_, T>, init: A, f: F) -> A
    where
        F: Fn(A, T, T) -> A,
    {
        self.to_ref().zip_fold(other, init, f)
    }

    pub fn dot(&self, other: MatRef<'_, T>) -> T {
        self.to_ref().dot(other)
    }
}

impl<'a, T: Scalar> MatRef<'a, T> {
    pub fn sum_of_rows(&self, alpha: T, mut out: MatRefMut<'_, T>) {
        debug_assert!(self.shape().cols() == out.shape().cols() && out.shape().rows() == 1);
        let acc = out.row_mut(0);
        acc.fill(T::ZERO);
        self.row_iterator().for_each(|r| {
            acc.iter_mut().zip(r.iter()).for_each(|(ac, &e)| *ac += e);
        });
        if alpha != T::ONE {
            out.scale_by(alpha);
        }
    }

    pub fn map<F>(&self, f: F, mut out: MatRefMut<'_, T>)
    where
        F: Fn(T) -> T + Sync,
    {
        debug_assert!(self.shape() == out.shape());
        self.row_iterator()
            .zip(out.row_iterator_mut())
            .for_each(|(rs, ro)| {
                rs.iter().zip(ro.iter_mut()).for_each(|(&s, o)| *o = f(s));
            });
    }

    pub fn zip_map<F>(&self, other: MatRef<'_, T>, f: F, mut out: MatRefMut<'_, T>)
    where
        F: Fn(T, T) -> T + Sync,
    {
        debug_assert!(self.shape() == other.shape() && self.shape() == out.shape());
        self.row_iterator()
            .zip(other.row_iterator())
            .zip(out.row_iterator_mut())
            .for_each(|((rs, ro), rr)| {
                rs.iter()
                    .zip(ro.iter())
                    .zip(rr.iter_mut())
                    .for_each(|((&s, &o), r)| *r = f(s, o));
            });
    }

    pub fn fold<A, F>(&self, init: A, f: F) -> A
    where
        F: Fn(A, T) -> A,
    {
        self.row_iterator()
            .fold(init, |acc, r| r.iter().fold(acc, |acc, &e| f(acc, e)))
    }

    pub fn sum(&self) -> T {
        self.fold(T::ZERO, |x, y| x + y)
    }

    pub fn zip_fold<A, F>(&self, other: MatRef<'_, T>, init: A, f: F) -> A
    where
        F: Fn(A, T, T) -> A,
    {
        debug_assert!(self.shape() == other.shape());
        self.row_iterator()
            .zip(other.row_iterator())
            .fold(init, |acc, (rs, ro)| {
                rs.iter()
                    .zip(ro.iter())
                    .fold(acc, |acc, (&es, &eo)| f(acc, es, eo))
            })
    }

    pub fn dot(&self, other: MatRef<'_, T>) -> T {
        self.zip_fold(other, T::ZERO, |a, x, y| a + x * y)
    }
}

impl<'a, T: Real> MatRef<'a, T> {
    pub fn norm(&self) -> T {
        self.dot(*self).sqrt()
    }

    pub fn get_means_stdevs(&self) -> (Vec<T>, Vec<T>) {
        let mut means = vec![T::ZERO; self.shape().cols()];

        self.row_iterator()
            .for_each(|r| r.iter().zip(means.iter_mut()).for_each(|(&x, s)| *s += x));
        let n_rows = self.shape().rows();
        let n_rows = T::from_f64(n_rows as f64);
        means.iter_mut().for_each(|x| *x = *x / n_rows);

        let mut stds = vec![T::ZERO; self.shape().cols()];
        self.row_iterator().for_each(|r| {
            r.iter()
                .zip(means.iter())
                .zip(stds.iter_mut())
                .for_each(|((&x, &s), st)| *st += (x - s) * (x - s))
        });
        stds.iter_mut().for_each(|st| *st = T::sqrt(*st / n_rows));

        stds.iter_mut().for_each(|s| {
            if *s < T::from_f64(1e-12) {
                *s = T::ONE
            }
        });

        (means, stds)
    }
}

impl<'a, T: Scalar> MatRefMut<'a, T> {
    pub fn scale_by(&mut self, a: T) {
        if a == T::ZERO {
            self.fill(a);
        } else if a != T::ONE {
            self.row_iterator_mut()
                .for_each(|r| r.iter_mut().for_each(|x| *x *= a));
        }
    }

    pub fn fill(&mut self, a: T) {
        self.row_iterator_mut().for_each(|r| r.fill(a));
    }

    pub fn add_to_every_row(&mut self, row_vec: MatRef<'_, T>) {
        debug_assert!(row_vec.shape() == Shape::new(1, self.shape().cols()));
        let row = row_vec.row(0);
        self.row_iterator_mut()
            .for_each(|r| r.iter_mut().zip(row.iter()).for_each(|(s, &o)| *s += o));
    }

    pub fn sum_of_rows(&self, alpha: T, out: MatRefMut<'_, T>) {
        self.reborrow().sum_of_rows(alpha, out);
    }

    pub fn map<F>(&self, f: F, out: MatRefMut<'_, T>)
    where
        F: Fn(T) -> T + Sync,
    {
        self.reborrow().map(f, out);
    }

    pub fn map_in_place<F>(&mut self, f: F)
    where
        F: Fn(T) -> T + Sync,
    {
        self.row_iterator_mut()
            .for_each(|r| r.iter_mut().for_each(|x| *x = f(*x)));
    }

    pub fn zip_map<F>(&self, other: MatRef<'_, T>, f: F, out: MatRefMut<'_, T>)
    where
        F: Fn(T, T) -> T + Sync,
    {
        self.reborrow().zip_map(other, f, out);
    }

    pub fn zip_map_in_place<F>(&mut self, other: MatRef<'_, T>, f: F)
    where
        F: Fn(T, T) -> T + Sync,
    {
        debug_assert!(self.shape() == other.shape());
        self.row_iterator_mut()
            .zip(other.row_iterator())
            .for_each(|(rs, ro)| {
                rs.iter_mut()
                    .zip(ro.iter())
                    .for_each(|(s, &o)| *s = f(*s, o));
            });
    }

    pub fn zip2_map_in_place<F>(&mut self, other1: MatRef<'_, T>, other2: MatRef<'_, T>, f: F)
    where
        F: Fn(T, T, T) -> T + Sync,
    {
        debug_assert!(self.shape() == other1.shape() && self.shape() == other2.shape());
        self.row_iterator_mut()
            .zip(other1.row_iterator())
            .zip(other2.row_iterator())
            .for_each(|((rs, ro1), ro2)| {
                rs.iter_mut()
                    .zip(ro1.iter())
                    .zip(ro2.iter())
                    .for_each(|((s, &o1), &o2)| *s = f(*s, o1, o2));
            });
    }

    pub fn fold<A, F>(&self, init: A, f: F) -> A
    where
        F: Fn(A, T) -> A,
    {
        self.reborrow().fold(init, f)
    }

    pub fn sum(&self) -> T {
        self.reborrow().sum()
    }

    pub fn zip_fold<A, F>(&self, other: MatRef<'_, T>, init: A, f: F) -> A
    where
        F: Fn(A, T, T) -> A,
    {
        self.reborrow().zip_fold(other, init, f)
    }

    pub fn dot(&self, other: MatRef<'_, T>) -> T {
        self.reborrow().dot(other)
    }

    pub fn apply_normalization(&mut self, means: &[T], stdevs: &[T]) {
        debug_assert!(means.len() == self.shape().cols() && stdevs.len() == means.len());
        self.row_iterator_mut().for_each(|r| {
            r.iter_mut()
                .zip(means.iter())
                .zip(stdevs.iter())
                .for_each(|((x, &m), &s)| *x = (*x - m) / s)
        });
    }

    pub fn invert_normalization(&mut self, means: &[T], stds: &[T]) {
        self.row_iterator_mut().for_each(|r| {
            r.iter_mut()
                .zip(means.iter())
                .zip(stds.iter())
                .for_each(|((x, &m), &s)| *x = *x * s + m)
        });
    }
}

impl<'a, T: Real> MatRefMut<'a, T> {
    pub fn normalize(&mut self) -> (Vec<T>, Vec<T>) {
        let (means, stdevs) = self.reborrow().get_means_stdevs();
        self.apply_normalization(&means, &stdevs);
        (means, stdevs)
    }
}

pub fn gemm<T: Scalar>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatRefMut<'_, T>,
) {
    debug_assert!(
        a.shape().cols() == b.shape().rows()
            && a.shape().rows() == c.shape().rows()
            && b.shape().cols() == c.shape().cols()
    );

    c.scale_by(beta);

    if alpha == T::ZERO {
        return;
    }

    a.into_row_iterator()
        .zip(c.row_iterator_mut())
        .for_each(|(r_a, r_c)| {
            b.into_row_iterator()
                .zip(r_a.iter())
                .for_each(|(r_b, &e_a)| {
                    let x = alpha * e_a;
                    r_b.iter()
                        .zip(r_c.iter_mut())
                        .for_each(|(&e_b, e_c)| *e_c += x * e_b);
                });
        });
}

pub fn gemm_at<T: Scalar>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatRefMut<'_, T>,
) {
    debug_assert!(
        a.shape().rows() == b.shape().rows()
            && a.shape().cols() == c.shape().rows()
            && b.shape().cols() == c.shape().cols()
    );

    c.scale_by(beta);

    if alpha == T::ZERO {
        return;
    }

    a.into_row_iterator()
        .zip(b.into_row_iterator())
        .for_each(|(r_a, r_b)| {
            r_a.iter()
                .zip(c.row_iterator_mut())
                .for_each(|(&e_a, r_c)| {
                    let x = alpha * e_a;
                    r_b.iter()
                        .zip(r_c.iter_mut())
                        .for_each(|(&e_b, e_c)| *e_c += x * e_b);
                });
        });
}

pub fn gemm_bt<T: Scalar>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatRefMut<'_, T>,
) {
    debug_assert!(
        a.shape().cols() == b.shape().cols()
            && a.shape().rows() == c.shape().rows()
            && b.shape().rows() == c.shape().cols()
    );

    c.scale_by(beta);

    if alpha == T::ZERO {
        return;
    }

    a.into_row_iterator()
        .zip(c.into_row_iterator_mut())
        .for_each(|(r_a, r_c)| {
            r_c.iter_mut()
                .zip(b.into_row_iterator())
                .for_each(|(e_c, r_b)| {
                    let acc = r_a
                        .iter()
                        .zip(r_b.iter())
                        .fold(T::ZERO, |s, (&x, &y)| s + x * y);
                    *e_c += alpha * acc;
                });
        });
}
