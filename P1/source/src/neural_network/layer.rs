use crate::math_utils::Scalar;
use crate::math_utils::matrix::layout::Shape;
use crate::math_utils::matrix::ops::{gemm, gemm_at, gemm_bt};
use crate::math_utils::matrix::{Mat, MatRef, MatRefMut};
use crate::neural_network::gradient::Gradient;
use crate::neural_network::params::Params;

pub struct Layer<T> {
    p: Params<T>,
}

impl<T: Scalar> Layer<T> {
    pub fn new(
        fan_in: usize,
        fan_out: usize,
        bias: bool,
        weights: impl Iterator<Item = f64>,
    ) -> Self {
        let shape_w = Shape::new(fan_in, fan_out);
        let data_w: Vec<T> = weights
            .take(shape_w.logical_count())
            .map(T::from_f64)
            .collect();

        Self {
            p: Params {
                w: Mat::from_vec(data_w, shape_w),
                b: bias.then(|| Mat::zeros(Shape::new(1, fan_out))),
            },
        }
    }

    pub fn forward(&self, x: MatRef<'_, T>, mut z: MatRefMut<'_, T>) {
        gemm(T::ONE, x, self.p.w(), T::ZERO, z.reborrow_mut());
        if let Some(b) = self.p.b() {
            z.add_to_every_row(b);
        }
    }

    pub fn backward(
        &self,
        x: MatRef<'_, T>,
        delta: MatRef<'_, T>,
        grad: &mut Gradient<T>,
        delta_prev: Option<MatRefMut<'_, T>>,
    ) {
        gemm_at(T::ONE, x, delta, T::ZERO, grad.p.w_mut());
        if let Some(db) = grad.p.b_mut() {
            delta.sum_of_rows(T::ONE, db);
        }

        if let Some(d) = delta_prev {
            gemm_bt(T::ONE, delta, self.p.w(), T::ZERO, d);
        }
    }

    pub fn zero_grad(&self) -> Gradient<T> {
        Gradient {
            p: self.p.zeros_like(),
        }
    }

    pub fn sgd_step(&mut self, grad: &Gradient<T>, lr: T) {
        self.p.zip_map_in_place(&grad.p, |w, g| w - lr * g);
    }

    pub fn params(&self) -> &Params<T> {
        &self.p
    }

    pub fn params_mut(&mut self) -> &mut Params<T> {
        &mut self.p
    }
}
