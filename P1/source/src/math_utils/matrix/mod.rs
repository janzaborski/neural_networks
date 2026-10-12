pub mod layout;
pub mod ops;

use std::ops::Range;

use crate::math_utils::Scalar;
use crate::math_utils::matrix::layout::{Layout, Shape};

pub struct Mat<T> {
    data: Vec<T>,
    layout: Layout,
}

pub struct MatRef<'a, T> {
    data: &'a [T],
    layout: Layout,
}

impl<T> Clone for MatRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for MatRef<'_, T> {}

pub struct MatRefMut<'a, T> {
    data: &'a mut [T],
    layout: Layout,
}

impl<T> Mat<T> {
    pub fn from_vec(data: Vec<T>, shape: Shape) -> Self {
        let layout = Layout::contiguous(shape);
        assert_eq!(
            data.len(),
            layout.shape().logical_count(),
            "data size != {}x{}",
            shape.rows(),
            shape.cols()
        );
        Self { data, layout }
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn shape(&self) -> Shape {
        self.layout.shape()
    }

    pub fn row_iterator(&self) -> impl Iterator<Item = &[T]> {
        self.to_ref().into_row_iterator()
    }

    pub fn row_iterator_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.to_ref_mut().into_row_iterator_mut()
    }

    pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        self.layout.is_contiguous().then_some(self.data.as_slice())
    }

    pub fn as_contiguous_slice_mut(&mut self) -> Option<&mut [T]> {
        self.layout
            .is_contiguous()
            .then_some(self.data.as_mut_slice())
    }

    pub fn to_ref(&self) -> MatRef<'_, T> {
        MatRef {
            data: self.data.as_slice(),
            layout: self.layout,
        }
    }

    pub fn to_ref_mut(&mut self) -> MatRefMut<'_, T> {
        MatRefMut {
            data: self.data.as_mut_slice(),
            layout: self.layout,
        }
    }
}

impl<'a, T> MatRef<'a, T> {
    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn shape(&self) -> Shape {
        self.layout.shape()
    }

    pub fn row(&self, i: usize) -> &'a [T] {
        &self.data[self.layout.row_range(i)]
    }

    pub fn subrows(&self, r: Range<usize>) -> MatRef<'a, T> {
        let (range, layout) = self.layout.subrows(r);
        MatRef {
            data: &self.data[range],
            layout,
        }
    }

    pub fn row_iterator(&self) -> impl Iterator<Item = &'a [T]> + use<'a, T> {
        (*self).into_row_iterator()
    }

    pub fn into_row_iterator(self) -> impl Iterator<Item = &'a [T]> {
        let layout = self.layout;
        let data = self.data;
        (0..layout.rows()).map(move |i| &data[layout.row_range(i)])
    }

    pub fn as_contiguous_slice(&self) -> Option<&'a [T]> {
        self.layout.is_contiguous().then_some(self.data)
    }
}

impl<'a, T> MatRefMut<'a, T> {
    pub fn layout(&self) -> Layout {
        self.layout
    }

    pub fn shape(&self) -> Shape {
        self.layout.shape()
    }

    pub fn reborrow(&self) -> MatRef<'_, T> {
        MatRef {
            data: &*self.data,
            layout: self.layout,
        }
    }

    pub fn reborrow_mut(&mut self) -> MatRefMut<'_, T> {
        MatRefMut {
            data: &mut *self.data,
            layout: self.layout,
        }
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.data[self.layout.row_range(i)]
    }

    pub fn row_iterator(&self) -> impl Iterator<Item = &[T]> {
        self.reborrow().into_row_iterator()
    }

    pub fn row_iterator_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        self.reborrow_mut().into_row_iterator_mut()
    }

    pub fn into_row_iterator_mut(self) -> impl Iterator<Item = &'a mut [T]> {
        let cols = self.layout.cols();
        let rows = self.layout.rows();
        self.data
            .chunks_mut(self.layout.stride())
            .take(rows)
            .map(move |r| &mut r[..cols])
    }

    pub fn as_contiguous_slice(&self) -> Option<&[T]> {
        self.layout.is_contiguous().then_some(&*self.data)
    }

    pub fn as_contiguous_slice_mut(&mut self) -> Option<&mut [T]> {
        self.layout.is_contiguous().then_some(&mut *self.data)
    }

    pub fn split_at_row_mut(self, row: usize) -> (MatRefMut<'a, T>, MatRefMut<'a, T>) {
        let (mid, top, bottom) = self.layout.split_at_row(row);
        let (d1, d2) = self.data.split_at_mut(mid);
        (
            MatRefMut {
                data: d1,
                layout: top,
            },
            MatRefMut {
                data: d2,
                layout: bottom,
            },
        )
    }
}

impl<T: Scalar> Mat<T> {
    pub fn zeros(shape: Shape) -> Self {
        Self::from_vec(vec![T::ZERO; shape.logical_count()], shape)
    }
}
