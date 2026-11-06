module;
#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <mdspan>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>
export module linalg;

export namespace linalg {

template <typename T>
concept Scalar = std::floating_point<T> || std::signed_integral<T>;

using extents = std::dextents<std::size_t, 2>;

template <typename T> using matrix_view = std::mdspan<T, extents>;

template <Scalar T> class Matrix {
public:
  constexpr Matrix(std::size_t rows, std::size_t cols, T value = T{});
  constexpr Matrix() = default;
  constexpr Matrix(extents ext, std::vector<T> data);

  constexpr auto rows() const noexcept -> std::size_t;
  constexpr auto cols() const noexcept -> std::size_t;
  constexpr auto size() const noexcept -> std::size_t;

  template <typename Self>
  using element_t = std::conditional_t<std::is_const_v<Self>, const T, T>;

  template <typename Self>
  constexpr auto view(this Self &self) -> matrix_view<element_t<Self>>;

  template <typename Self>
  constexpr auto operator[](this Self &self, std::size_t i, std::size_t j)
      -> element_t<Self> &;

  template <typename Self>
  constexpr auto data(this Self &self) -> std::span<element_t<Self>>;

  template <typename Self>
  constexpr auto row(this Self &self, std::size_t i)
      -> std::span<element_t<Self>>;

  constexpr auto fill(T value) -> void;

  constexpr bool operator==(const Matrix &rhs) const = default;

private:
  extents extents_;
  std::vector<T> data_;
};

template <typename T>
concept scalar_element = Scalar<std::remove_const_t<T>>;

template <typename A, typename B>
using product_t = decltype(std::declval<A>() * std::declval<B>());

template <scalar_element A>
constexpr auto row_of(matrix_view<A> a, std::size_t i) -> std::span<A>;

template <scalar_element X, scalar_element Y>
constexpr auto dot(std::span<X> x, std::span<Y> y) -> product_t<X, Y>;

template <scalar_element A, scalar_element X, Scalar Y>
constexpr auto gemv(matrix_view<A> a, std::span<X> x, std::span<Y> y) -> void;

template <scalar_element A, scalar_element X, Scalar Y>
constexpr auto gemv_t(matrix_view<A> a, std::span<X> x, std::span<Y> y) -> void;

template <Scalar A, scalar_element X, scalar_element Y>
constexpr auto rank1_update(matrix_view<A> a, std::type_identity_t<A> alpha,
                            std::span<X> x, std::span<Y> y) -> void;

template <Scalar A> constexpr auto zero_diagonal(matrix_view<A> a) -> void;

// impls

template <Scalar T>
constexpr Matrix<T>::Matrix(std::size_t rows, std::size_t cols, T value)
    : extents_{rows, cols}, data_(rows * cols, value) {}

template <Scalar T>
constexpr Matrix<T>::Matrix(extents ext, std::vector<T> data)
    : extents_{ext}, data_{std::move(data)} {
  if (data_.size() != extents_.extent(0) * extents_.extent(1)) {
    throw std::invalid_argument("Matrix: data size does not match extents");
  }
}

template <Scalar T>
constexpr auto Matrix<T>::rows() const noexcept -> std::size_t {
  return extents_.extent(0);
}

template <Scalar T>
constexpr auto Matrix<T>::cols() const noexcept -> std::size_t {
  return extents_.extent(1);
}

template <Scalar T>
constexpr auto Matrix<T>::size() const noexcept -> std::size_t {
  return rows() * cols();
}

template <Scalar T>
template <typename Self>
constexpr auto Matrix<T>::view(this Self &self)
    -> matrix_view<element_t<Self>> {
  return matrix_view<element_t<Self>>(self.data_.data(), self.extents_);
}

template <Scalar T>
template <typename Self>
constexpr auto Matrix<T>::operator[](this Self &self, std::size_t i,
                                     std::size_t j) -> element_t<Self> & {
  assert(i < self.rows() && j < self.cols());
  return self.data_[i * self.cols() + j];
}

template <Scalar T>
template <typename Self>
constexpr auto Matrix<T>::data(this Self &self) -> std::span<element_t<Self>> {
  return std::span<element_t<Self>>(self.data_);
}

template <Scalar T>
template <typename Self>
constexpr auto Matrix<T>::row(this Self &self, std::size_t i)
    -> std::span<element_t<Self>> {
  assert(i < self.rows());
  return self.data().subspan(i * self.cols(), self.cols());
}

template <scalar_element A>
constexpr auto row_of(matrix_view<A> a, std::size_t i) -> std::span<A> {
  return std::span<A>(a.data_handle() + i * a.stride(0), a.extent(1));
}

template <Scalar T> constexpr auto Matrix<T>::fill(T value) -> void {
  std::ranges::fill(data_, value);
}

template <scalar_element X, scalar_element Y>
constexpr auto dot(std::span<X> x, std::span<Y> y) -> product_t<X, Y> {
  return std::ranges::fold_left(
      std::views::zip_transform(std::multiplies{}, x, y), product_t<X, Y>{},
      std::plus{});
}

template <scalar_element A, scalar_element X, Scalar Y>
constexpr auto gemv(matrix_view<A> a, std::span<X> x, std::span<Y> y) -> void {
  if (a.extent(1) != x.size() || a.extent(0) != y.size()) {
    throw std::invalid_argument("gemv: dimension mismatch");
  }

  for (auto i{0uz}; i < a.extent(0); ++i) {
    y[i] = static_cast<Y>(dot(row_of(a, i), x));
  }
}

template <scalar_element A, scalar_element X, Scalar Y>
constexpr auto gemv_t(matrix_view<A> a, std::span<X> x, std::span<Y> y)
    -> void {
  if (a.extent(0) != x.size() || a.extent(1) != y.size()) {
    throw std::invalid_argument("gemv_t: dimension mismatch");
  }

  std::ranges::fill(y, Y{});
  for (auto i{0uz}; i < a.extent(0); ++i) {
    const auto xi = x[i];
    for (auto [yj, aij] : std::views::zip(y, row_of(a, i))) {
      yj = static_cast<Y>(yj + aij * xi);
    }
  }
}

template <Scalar A, scalar_element X, scalar_element Y>
constexpr auto rank1_update(matrix_view<A> a, std::type_identity_t<A> alpha,
                            std::span<X> x, std::span<Y> y) -> void {
  if (a.extent(0) != x.size() || a.extent(1) != y.size()) {
    throw std::invalid_argument("rank1_update: dimension mismatch");
  }

  for (auto i{0uz}; i < a.extent(0); ++i) {
    const auto scale = alpha * x[i];
    for (auto [aij, yj] : std::views::zip(row_of(a, i), y)) {
      aij = static_cast<A>(aij + scale * yj);
    }
  }
}

template <Scalar A> constexpr auto zero_diagonal(matrix_view<A> a) -> void {
  const auto n = std::min(a.extent(0), a.extent(1));
  for (auto i{0uz}; i < n; ++i) {
    a[i, i] = A{};
  }
}

} // namespace linalg
