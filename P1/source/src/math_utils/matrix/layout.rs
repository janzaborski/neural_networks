use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    rows: usize,
    cols: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Layout {
    shape: Shape,
    stride: usize,
}

impl Shape {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    pub fn rows(self) -> usize {
        self.rows
    }

    pub fn cols(self) -> usize {
        self.cols
    }

    pub fn logical_count(self) -> usize {
        self.rows * self.cols
    }
}

impl Layout {
    pub fn new(shape: Shape, stride: usize) -> Self {
        assert!(shape.cols > 0, "cols must be > 0");
        assert!(
            stride >= shape.cols,
            "stride ({stride}) < cols ({})",
            shape.cols
        );

        Self { shape, stride }
    }

    pub fn contiguous(shape: Shape) -> Self {
        Self::new(shape, shape.cols)
    }

    pub fn is_contiguous(self) -> bool {
        self.stride == self.shape.cols
    }

    pub fn required_len(self) -> usize {
        if self.shape.rows == 0 {
            return 0;
        }
        (self.shape.rows - 1) * self.stride + self.shape.cols
    }

    pub fn row_range(self, i: usize) -> Range<usize> {
        debug_assert!(
            i < self.shape.rows,
            "row id ({i}) >= total rows ({})",
            self.shape.rows
        );
        let start = i * self.stride;
        start..start + self.shape.cols
    }

    pub fn subrows(self, r: Range<usize>) -> (Range<usize>, Layout) {
        debug_assert!(
            r.start <= r.end && r.end <= self.shape.rows,
            "range {r:?} outside 0..{}",
            self.shape.rows
        );
        let layout = Layout {
            shape: Shape::new(r.end - r.start, self.shape.cols),
            stride: self.stride,
        };

        let start = (r.start * self.stride).min(self.required_len());
        (start..start + layout.required_len(), layout)
    }

    pub fn split_at_row(self, row: usize) -> (usize, Layout, Layout) {
        debug_assert!(
            row <= self.shape.rows,
            "split row id ({row}) > total rows ({})",
            self.shape.rows
        );
        let top = Layout {
            shape: Shape::new(row, self.shape.cols),
            stride: self.stride,
        };
        let bottom = Layout {
            shape: Shape::new(self.shape.rows - row, self.shape.cols),
            stride: self.stride,
        };
        let mid = (row * self.stride).min(self.required_len());
        (mid, top, bottom)
    }

    pub fn shape(self) -> Shape {
        self.shape
    }

    pub fn rows(self) -> usize {
        self.shape.rows()
    }

    pub fn cols(self) -> usize {
        self.shape.cols()
    }

    pub fn stride(self) -> usize {
        self.stride
    }
}
