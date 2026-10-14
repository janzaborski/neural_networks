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

macro_rules! impl_native_float {
    ($($t:ty),* $(,)?) => {$(
        impl Scalar for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            #[inline] fn from_f64(x: f64) -> Self { x as $t }
            #[inline] #[allow(clippy::unnecessary_cast)] fn to_f64(self) -> f64 { self as f64 }
            #[inline] fn max(self, other: Self) -> Self { <$t>::max(self, other) }
            #[inline] fn min(self, other: Self) -> Self { <$t>::min(self, other) }
            #[inline] fn abs(self) -> Self { <$t>::abs(self) }
            #[inline] fn is_finite(self) -> bool { <$t>::is_finite(self) }
        }

        impl Real for $t {
            #[inline] fn ln(self) -> Self { <$t>::ln(self) }
            #[inline] fn sqrt(self) -> Self { <$t>::sqrt(self) }
            #[inline] fn exp(self) -> Self { <$t>::exp(self) }
            #[inline] fn tanh(self) -> Self { <$t>::tanh(self) }
        }
    )*};
}

#[allow(unused_macros)]
macro_rules! impl_float_via_f64 {
    ($t:ty, $zero:expr, $one:expr, $from:expr, $to:expr $(,)?) => {
        impl Scalar for $t {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            #[inline]
            fn from_f64(x: f64) -> Self {
                ($from)(x)
            }
            #[inline]
            fn to_f64(self) -> f64 {
                ($to)(self)
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                Self::from_f64(self.to_f64().max(other.to_f64()))
            }
            #[inline]
            fn min(self, other: Self) -> Self {
                Self::from_f64(self.to_f64().min(other.to_f64()))
            }
            #[inline]
            fn abs(self) -> Self {
                Self::from_f64(self.to_f64().abs())
            }
            #[inline]
            fn is_finite(self) -> bool {
                self.to_f64().is_finite()
            }
        }

        impl Real for $t {
            #[inline]
            fn ln(self) -> Self {
                Self::from_f64(self.to_f64().ln())
            }
            #[inline]
            fn sqrt(self) -> Self {
                Self::from_f64(self.to_f64().sqrt())
            }
            #[inline]
            fn exp(self) -> Self {
                Self::from_f64(self.to_f64().exp())
            }
            #[inline]
            fn tanh(self) -> Self {
                Self::from_f64(self.to_f64().tanh())
            }
        }
    };
}

impl_native_float!(f32, f64);
