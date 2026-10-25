use crate::math_utils::matrix::ops::{gemm, gemm_at, gemm_bt};
use crate::math_utils::matrix::{MatRef, MatRefMut};

#[inline(never)]
pub fn gemm_f64(
    alpha: f64,
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    beta: f64,
    c: MatRefMut<'_, f64>,
) {
    gemm(alpha, a, b, beta, c)
}

#[inline(never)]
pub fn gemm_at_f64(
    alpha: f64,
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    beta: f64,
    c: MatRefMut<'_, f64>,
) {
    gemm_at(alpha, a, b, beta, c)
}

#[inline(never)]
pub fn gemm_bt_f64(
    alpha: f64,
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    beta: f64,
    c: MatRefMut<'_, f64>,
) {
    gemm_bt(alpha, a, b, beta, c)
}
