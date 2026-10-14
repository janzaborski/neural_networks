use crate::math_utils::{Real, Scalar};
use crate::neural_network::params::Params;

pub struct Gradient<T> {
    pub p: Params<T>,
}

impl<T: Scalar> Gradient<T> {
    pub fn params(&self) -> &Params<T> {
        &self.p
    }
}

impl<T: Real> Gradient<T> {
    pub fn norm(&self) -> T {
        let sq = self.p.w().dot(self.p.w()) + self.p.b().map_or(T::ZERO, |b| b.dot(b));
        sq.sqrt()
    }
}
