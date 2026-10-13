use crate::math_utils::matrix::{Mat, MatRef, MatRefMut};
use crate::math_utils::{Real, Scalar};
use crate::neural_network::activation::Activation;

pub struct Layer<T> {
    w: Mat<T>,
    b: Option<Mat<T>>,
}
