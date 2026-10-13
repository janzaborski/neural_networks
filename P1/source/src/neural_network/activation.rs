use crate::math_utils::matrix::{MatRef, MatRefMut};
use crate::math_utils::{Real, Scalar};

pub trait Activation<T: Scalar> {
    fn forward(&self, z: MatRef<'_, T>, a: MatRefMut<'_, T>);
    fn backward(&self, z: MatRef<'_, T>, a: MatRef<'_, T>, grad: MatRefMut<'_, T>);

    fn init_std(&self, fan_in: usize, fan_out: usize) -> f64 {
        debug_assert!(fan_in > 0 && fan_out > 0);
        (2f64 / (fan_in + fan_out) as f64).sqrt()
    }
}

pub trait Pointwise<T: Scalar> {
    fn f(&self, z: T) -> T;
    fn df(&self, z: T, a: T) -> T;

    fn init_std(&self, fan_in: usize, fan_out: usize) -> f64 {
        debug_assert!(fan_in > 0 && fan_out > 0);
        (2f64 / (fan_in + fan_out) as f64).sqrt()
    }
}

pub struct ElementWise<P>(pub P);
impl<T: Scalar, P: Pointwise<T> + Sync> Activation<T> for ElementWise<P> {
    fn forward(&self, z: MatRef<'_, T>, a: MatRefMut<'_, T>) {
        z.map(|x| self.0.f(x), a);
    }

    fn backward(&self, z: MatRef<'_, T>, a: MatRef<'_, T>, mut grad: MatRefMut<'_, T>) {
        grad.zip2_map_in_place(z, a, |g, z, a| g * self.0.df(z, a));
    }

    fn init_std(&self, fan_in: usize, fan_out: usize) -> f64 {
        self.0.init_std(fan_in, fan_out)
    }
}

pub struct Sigmoid;
impl<T: Real> Pointwise<T> for Sigmoid {
    fn f(&self, z: T) -> T {
        T::ONE / (T::ONE + T::exp(-z))
    }

    fn df(&self, _: T, a: T) -> T {
        a * (T::ONE - a)
    }
}

pub struct Tanh;
impl<T: Real> Pointwise<T> for Tanh {
    fn f(&self, z: T) -> T {
        T::tanh(z)
    }

    fn df(&self, _: T, a: T) -> T {
        T::ONE - a * a
    }
}

pub struct ReLU;
impl<T: Scalar> Pointwise<T> for ReLU {
    fn f(&self, z: T) -> T {
        T::max(z, T::ZERO)
    }

    fn df(&self, z: T, _: T) -> T {
        if z > T::ZERO { T::ONE } else { T::ZERO }
    }

    fn init_std(&self, fan_in: usize, _fan_out: usize) -> f64 {
        debug_assert!(fan_in > 0);
        (2.0 / fan_in as f64).sqrt()
    }
}

pub struct Identity;
impl<T: Scalar> Activation<T> for Identity {
    fn forward(&self, z: MatRef<'_, T>, a: MatRefMut<'_, T>) {
        z.copy_into(a);
    }

    fn backward(&self, _: MatRef<'_, T>, _: MatRef<'_, T>, _: MatRefMut<'_, T>) {}
}

pub struct Softmax;
impl<T: Real> Activation<T> for Softmax {
    fn forward(&self, z: MatRef<'_, T>, mut a: MatRefMut<'_, T>) {
        debug_assert!(z.shape() == a.shape());

        z.row_iterator()
            .zip(a.row_iterator_mut())
            .for_each(|(r_z, r_a)| {
                let m = r_z.iter().fold(r_z[0], |m, &x| m.max(x));

                let mut s = T::ZERO;
                r_z.iter().zip(r_a.iter_mut()).for_each(|(&e_z, e_a)| {
                    *e_a = T::exp(e_z - m);
                    s += *e_a;
                });

                let s_inv = T::ONE / s;
                r_a.iter_mut().for_each(|e_a| *e_a *= s_inv);
            });
    }

    fn backward(&self, _: MatRef<'_, T>, a: MatRef<'_, T>, mut grad: MatRefMut<'_, T>) {
        debug_assert!(a.shape() == grad.shape());

        grad.row_iterator_mut()
            .zip(a.row_iterator())
            .for_each(|(r_g, r_a)| {
                let dot = r_g
                    .iter()
                    .zip(r_a.iter())
                    .fold(T::ZERO, |s, (&g, &a)| s + g * a);

                r_g.iter_mut()
                    .zip(r_a.iter())
                    .for_each(|(e_g, &e_a)| *e_g = e_a * (*e_g - dot));
            });
    }
}
