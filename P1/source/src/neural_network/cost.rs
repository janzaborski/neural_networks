use crate::math_utils::matrix::{MatRef, MatRefMut};
use crate::math_utils::{Real, Scalar};

pub trait Cost<T: Scalar> {
    fn cost(&self, a: MatRef<'_, T>, y: MatRef<'_, T>) -> f64;
    fn grad(&self, a: MatRef<'_, T>, y: MatRef<'_, T>, out: MatRefMut<'_, T>);
}

pub struct Mse;
impl<T: Scalar> Cost<T> for Mse {
    fn cost(&self, a: MatRef<'_, T>, y: MatRef<'_, T>) -> f64 {
        let n = a.shape().rows();
        let c = a.zip_fold(y, 0f64, |acc, a, y| acc + (a.to_f64() - y.to_f64()).powi(2));

        c / (2f64 * n as f64)
    }

    fn grad(&self, a: MatRef<'_, T>, y: MatRef<'_, T>, out: MatRefMut<'_, T>) {
        let n = T::from_f64(a.shape().rows() as f64);
        let n_inv = T::ONE / n;

        a.zip_map(y, |a, y| (a - y) * n_inv, out);
    }
}

pub struct CrossEntropy;
impl<T: Real> Cost<T> for CrossEntropy {
    fn cost(&self, a: MatRef<'_, T>, y: MatRef<'_, T>) -> f64 {
        let n = a.shape().rows();
        let c = -a.zip_fold(y, 0f64, |acc, a, y| {
            acc + y.to_f64() * a.to_f64().max(1e-12).ln()
        });

        c / (n as f64)
    }

    fn grad(&self, a: MatRef<'_, T>, y: MatRef<'_, T>, out: MatRefMut<'_, T>) {
        let n = T::from_f64(a.shape().rows() as f64);
        let n_inv = T::ONE / n;

        a.zip_map(y, |a, y| -y / a.max(T::from_f64(1e-12)) * n_inv, out);
    }
}

pub struct Mae;
impl<T: Scalar> Cost<T> for Mae {
    fn cost(&self, a: MatRef<'_, T>, y: MatRef<'_, T>) -> f64 {
        let n = a.shape().rows();
        let c = a.zip_fold(y, 0f64, |acc, a, y| acc + (a - y).abs().to_f64());

        c / (n as f64)
    }

    fn grad(&self, a: MatRef<'_, T>, y: MatRef<'_, T>, out: MatRefMut<'_, T>) {
        let n = T::from_f64(a.shape().rows() as f64);
        let n_inv = T::ONE / n;

        a.zip_map(y, |a, y| (a - y).sign() * n_inv, out);
    }
}
