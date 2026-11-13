module;

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <format>
#include <functional>
#include <limits>
#include <mdspan>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

export module linalg;

namespace linalg {

constexpr auto check(bool condition, const char *function,
                     std::string_view what) -> void {
  if (!condition) {
    throw std::invalid_argument(std::format("linalg::{}: {}", function, what));
  }
}

} // namespace linalg

export namespace linalg {

template <typename T>
concept Scalar = std::is_arithmetic_v<T>;

template <Scalar A, Scalar B>
using product_t = decltype(std::declval<A>() * std::declval<B>());

using extents = std::dextents<std::size_t, 2>;

template <typename T> using matrix_view = std::mdspan<T, extents>;

template <Scalar T> class Matrix {
public:
  using value_type = T;

  constexpr Matrix() = default;

  constexpr Matrix(std::size_t rows, std::size_t cols, T value = T{})
      : extents_{rows, cols}, data_(rows * cols, value) {}

  constexpr Matrix(extents shape, std::vector<T> data)
      : extents_{shape}, data_(std::move(data)) {
    check(data_.size() == rows() * cols(), "Matrix",
          "data size does not match the shape");
  }

  [[nodiscard]] constexpr auto rows() const noexcept -> std::size_t {
    return extents_.extent(0);
  }
  [[nodiscard]] constexpr auto cols() const noexcept -> std::size_t {
    return extents_.extent(1);
  }

  template <typename Self> [[nodiscard]] constexpr auto view(this Self &self) {
    using Element = std::conditional_t<std::is_const_v<Self>, const T, T>;
    return matrix_view<Element>(self.data_.data(), self.extents_);
  }

  template <typename Self>
  [[nodiscard]] constexpr auto row(this Self &self, std::size_t i) {
    using Element = std::conditional_t<std::is_const_v<Self>, const T, T>;
    check(i < self.rows(), "Matrix::row", "row index out of range");
    return std::span<Element>(self.data_.data() + i * self.cols(), self.cols());
  }

  template <typename Self> [[nodiscard]] constexpr auto data(this Self &self) {
    using Element = std::conditional_t<std::is_const_v<Self>, const T, T>;
    return std::span<Element>(self.data_);
  }

private:
  extents extents_{0, 0};
  std::vector<T> data_;
};

template <Scalar A, Scalar B>
[[nodiscard]] constexpr auto dot(std::span<const A> a, std::span<const B> b)
    -> product_t<A, B> {
  check(a.size() == b.size(), "dot", "sizes differ");
  return std::ranges::fold_left(std::views::zip(a, b), product_t<A, B>{},
                                [](product_t<A, B> acc, const auto &pair) {
                                  const auto &[x, y] = pair;
                                  return acc + x * y;
                                });
}

template <Scalar A, Scalar X, Scalar Y>
constexpr auto gemv(matrix_view<const A> a, std::span<const X> x,
                    std::span<Y> y) -> void {
  check(a.extent(1) == x.size() && a.extent(0) == y.size(), "gemv",
        "shapes differ");
  for (std::size_t i = 0; i < a.extent(0); ++i) {
    Y acc{};
    for (std::size_t j = 0; j < a.extent(1); ++j) {
      acc += static_cast<Y>(a[i, j] * x[j]);
    }
    y[i] = acc;
  }
}

template <Scalar A, Scalar X, Scalar Y>
constexpr auto gemv_t(matrix_view<const A> a, std::span<const X> x,
                      std::span<Y> y) -> void {
  check(a.extent(0) == x.size() && a.extent(1) == y.size(), "gemv_t",
        "shapes differ");
  std::ranges::fill(y, Y{});
  for (std::size_t i = 0; i < a.extent(0); ++i) {
    for (std::size_t j = 0; j < a.extent(1); ++j) {
      y[j] += static_cast<Y>(a[i, j] * x[i]);
    }
  }
}

template <Scalar Alpha, Scalar X, Scalar Y>
constexpr auto axpy(Alpha alpha, std::span<const X> x, std::span<Y> y) -> void {
  check(x.size() == y.size(), "axpy", "sizes differ");
  for (auto &&[xi, yi] : std::views::zip(x, y)) {
    yi += static_cast<Y>(alpha * xi);
  }
}

template <Scalar T>
[[nodiscard]] constexpr auto transpose(matrix_view<const T> a) -> Matrix<T> {
  Matrix<T> result(a.extent(1), a.extent(0));
  auto r = result.view();
  for (std::size_t i = 0; i < a.extent(0); ++i) {
    for (std::size_t j = 0; j < a.extent(1); ++j) {
      r[j, i] = a[i, j];
    }
  }
  return result;
}

template <Scalar R, Scalar A, Scalar B>
[[nodiscard]] constexpr auto matmul(matrix_view<const A> a,
                                    matrix_view<const B> b) -> Matrix<R> {
  check(a.extent(1) == b.extent(0), "matmul", "inner dimensions differ");
  Matrix<R> result(a.extent(0), b.extent(1));
  auto r = result.view();
  for (std::size_t i = 0; i < a.extent(0); ++i) {
    for (std::size_t k = 0; k < a.extent(1); ++k) {
      const auto aik = static_cast<R>(a[i, k]);
      for (std::size_t j = 0; j < b.extent(1); ++j) {
        r[i, j] += aik * static_cast<R>(b[k, j]);
      }
    }
  }
  return result;
}

template <std::floating_point T>
[[nodiscard]] constexpr auto inverse(Matrix<T> a) -> Matrix<T> {
  const auto n = a.rows();
  check(a.cols() == n, "inverse", "matrix is not square");
  Matrix<T> inv(n, n);
  auto m = a.view();
  auto r = inv.view();
  for (std::size_t i = 0; i < n; ++i) {
    r[i, i] = T{1};
  }

  const auto scale = std::ranges::fold_left(
      a.data() | std::views::transform([](T v) { return std::abs(v); }), T{},
      [](T x, T y) { return std::max(x, y); });
  const auto tolerance = T{1e-9} * scale;

  for (std::size_t col = 0; col < n; ++col) {
    auto pivot = col;
    for (std::size_t i = col + 1; i < n; ++i) {
      if (std::abs(m[i, col]) > std::abs(m[pivot, col])) {
        pivot = i;
      }
    }
    if (!(std::abs(m[pivot, col]) > tolerance)) {
      throw std::domain_error("linalg::inverse: matrix is singular");
    }
    if (pivot != col) {
      std::ranges::swap_ranges(a.row(col), a.row(pivot));
      std::ranges::swap_ranges(inv.row(col), inv.row(pivot));
    }
    const auto d = m[col, col];
    for (std::size_t j = 0; j < n; ++j) {
      m[col, j] /= d;
      r[col, j] /= d;
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (i == col) {
        continue;
      }
      const auto f = m[i, col];
      for (std::size_t j = 0; j < n; ++j) {
        m[i, j] -= f * m[col, j];
        r[i, j] -= f * r[col, j];
      }
    }
  }
  return inv;
}

} // namespace linalg
