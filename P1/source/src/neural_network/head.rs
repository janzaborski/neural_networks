use crate::math_utils::matrix::{MatRef, MatRefMut};
use crate::math_utils::{Real, Scalar};
use crate::neural_network::activation::{Activation, Softmax};
use crate::neural_network::cost::{Cost, CrossEntropy};

pub trait Head<T: Scalar> {
    fn forward(&self, z: MatRef<'_, T>, a: MatRefMut<'_, T>);
    fn cost(&self, a: MatRef<'_, T>, y: MatRef<'_, T>) -> f64;
    fn delta(&self, z: MatRef<'_, T>, a: MatRef<'_, T>, y: MatRef<'_, T>, delta: MatRefMut<'_, T>);
    fn init_std(&self, fan_in: usize, fan_out: usize) -> f64;
}

pub struct Composed<A, C> {
    act: A,
    cost: C,
}

impl<T: Scalar, A: Activation<T>, C: Cost<T>> Head<T> for Composed<A, C> {
    fn forward(&self, z: MatRef<'_, T>, a: MatRefMut<'_, T>) {
        self.act.forward(z, a);
    }

    fn cost(&self, a: MatRef<'_, T>, y: MatRef<'_, T>) -> f64 {
        self.cost.cost(a, y)
    }

    fn delta(
        &self,
        z: MatRef<'_, T>,
        a: MatRef<'_, T>,
        y: MatRef<'_, T>,
        mut delta: MatRefMut<'_, T>,
    ) {
        self.cost.grad(a, y, delta.reborrow_mut());
        self.act.backward(z, a, delta);
    }

    fn init_std(&self, fan_in: usize, fan_out: usize) -> f64 {
        self.act.init_std(fan_in, fan_out)
    }
}

impl<A, C> Composed<A, C> {
    pub fn new(act: A, cost: C) -> Self {
        Self { act, cost }
    }
}

pub struct SoftmaxCrossEntropy;

impl<T: Real> Head<T> for SoftmaxCrossEntropy {
    fn forward(&self, z: MatRef<'_, T>, a: MatRefMut<'_, T>) {
        Softmax.forward(z, a);
    }

    fn cost(&self, a: MatRef<'_, T>, y: MatRef<'_, T>) -> f64 {
        CrossEntropy.cost(a, y)
    }

    fn delta(&self, _: MatRef<'_, T>, a: MatRef<'_, T>, y: MatRef<'_, T>, delta: MatRefMut<'_, T>) {
        let n = a.shape().rows();
        let n_inv = T::from_f64(1f64 / n as f64);

        a.zip_map(y, |a, y| (a - y) * n_inv, delta);
    }

    fn init_std(&self, fan_in: usize, fan_out: usize) -> f64 {
        Activation::<T>::init_std(&Softmax, fan_in, fan_out)
    }
}
