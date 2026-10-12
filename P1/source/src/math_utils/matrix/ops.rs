use crate::math_utils::Scalar;
use crate::math_utils::matrix::layout::{Layout, Shape};
use crate::math_utils::matrix::{Mat, MatRef, MatRefMut};

impl<T: Scalar> Mat<T> {
    pub fn scale_by(&mut self, a: T) {
        self.to_ref_mut().scale_by(a);
    }

    pub fn fill(&mut self, a: T) {
        self.data.fill(a);
    }
}

impl<'a, T: Scalar> MatRef<'a, T> {}

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
        self.data.fill(a);
    }
}

pub fn gemm<T: Scalar>(a: T, A: MatRef<'_, T>, B: MatRef<'_, T>, b: T, mut C: MatRefMut<'_, T>) {
    debug_assert!(
        A.shape().cols() == B.shape().rows()
            && A.shape().rows() == C.shape().rows()
            && B.shape().cols() == C.shape().cols(),
        "A B C have wrong shapes"
    );

    C.scale_by(b);

    if a == T::ZERO {
        return;
    }

    A.into_row_iterator()
        .zip(C.row_iterator_mut())
        .for_each(|(rA, rC)| {
            B.into_row_iterator().zip(rA.iter()).for_each(|(rB, &eA)| {
                let x = a * eA;
                rB.iter()
                    .zip(rC.iter_mut())
                    .for_each(|(&eB, eC)| *eC += x * eB);
            });
        });
}

pub fn gemm_AT<T: Scalar>(a: T, A: MatRef<'_, T>, B: MatRef<'_, T>, b: T, mut C: MatRefMut<'_, T>) {
    debug_assert!(
        A.shape().rows() == B.shape().rows()
            && A.shape().cols() == C.shape().rows()
            && B.shape().cols() == C.shape().cols(),
        "A B C have wrong shapes"
    );

    C.scale_by(b);

    if a == T::ZERO {
        return;
    }

    todo!()
}
