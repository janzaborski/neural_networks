use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub mod matrix;

pub trait Scalar:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + MulAssign
    + SubAssign
    + Debug
    + Send
    + Sync
{
    const ZERO: Self;
    const ONE: Self;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn abs(self) -> Self;
    fn is_finite(self) -> bool;

    fn sign(self) -> Self {
        if self > Self::ZERO {
            Self::ONE
        } else if self < Self::ZERO {
            -Self::ONE
        } else {
            Self::ZERO
        }
    }
}

pub trait Real: Scalar {
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn tanh(self) -> Self;
}
