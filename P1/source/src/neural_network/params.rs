use crate::math_utils::Scalar;
use crate::math_utils::matrix::{Mat, MatRef, MatRefMut};

pub struct Params<T> {
    pub w: Mat<T>,
    pub b: Option<Mat<T>>,
}

impl<T: Scalar> Params<T> {
    pub fn zeros_like(&self) -> Self {
        Self {
            w: Mat::zeros(self.w.shape()),
            b: self.b.as_ref().map(|b| Mat::zeros(b.shape())),
        }
    }

    pub fn zip_map_in_place<F>(&mut self, other: &Params<T>, f: F)
    where
        F: Fn(T, T) -> T + Sync,
    {
        debug_assert!(self.b.is_some() == other.b.is_some());

        self.w.zip_map_in_place(other.w.to_ref(), &f);
        if let (Some(b), Some(ob)) = (&mut self.b, &other.b) {
            b.zip_map_in_place(ob.to_ref(), &f);
        }
    }

    pub fn w(&self) -> MatRef<'_, T> {
        self.w.to_ref()
    }

    pub fn w_mut(&mut self) -> MatRefMut<'_, T> {
        self.w.to_ref_mut()
    }

    pub fn b(&self) -> Option<MatRef<'_, T>> {
        self.b.as_ref().map(|b| b.to_ref())
    }

    pub fn b_mut(&mut self) -> Option<MatRefMut<'_, T>> {
        self.b.as_mut().map(|b| b.to_ref_mut())
    }

    pub fn fan_in(&self) -> usize {
        self.w.shape().rows()
    }

    pub fn fan_out(&self) -> usize {
        self.w.shape().cols()
    }
}
