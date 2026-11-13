module;

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <generator>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

export module hopfield;

export import linalg;

export namespace hopfield {

using State = std::int8_t;

enum class LearningMethod { hebb, pseudoinverse };
enum class Simulation { sync, async };
enum class Status { stable, cycle, max_steps, running };

struct Step {
  std::size_t index;
  std::span<const State> state;
  std::size_t changed;
  Status status;
};

inline constexpr double tie_tolerance = 1e-9;

template <std::floating_point T>
[[nodiscard]] constexpr auto next_state(T h, State current) noexcept -> State {
  if (h > static_cast<T>(tie_tolerance)) {
    return State{1};
  }
  if (h < -static_cast<T>(tie_tolerance)) {
    return State{-1};
  }
  return current;
}

template <std::floating_point T>
class HopfieldNetwork {
public:
  HopfieldNetwork(linalg::Matrix<State> patterns, linalg::Matrix<T> coefficients,
                  std::vector<T> self_weights)
      : x_(std::move(patterns)), b_(std::move(coefficients)), d_(std::move(self_weights)) {
    if (b_.rows() != x_.rows() || b_.cols() != x_.cols() || d_.size() != x_.rows()) {
      throw std::invalid_argument("HopfieldNetwork: shapes of patterns, coefficients and self weights differ");
    }
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t { return x_.rows(); }
  [[nodiscard]] auto pattern_count() const noexcept -> std::size_t { return x_.cols(); }

  auto local_field(std::span<const State> x, std::span<T> h) const -> void {
    std::vector<T> s(pattern_count());
    overlaps(x, s);
    linalg::gemv(b_.view(), std::span<const T>(s), h);
    for (std::size_t i = 0; i < size(); ++i) {
      h[i] -= d_[i] * static_cast<T>(x[i]);
    }
  }

  [[nodiscard]] auto weights() const -> linalg::Matrix<T> {
    linalg::Matrix<T> w(size(), size());
    auto view = w.view();
    for (std::size_t i = 0; i < size(); ++i) {
      for (std::size_t j = 0; j < size(); ++j) {
        view[i, j] = i == j ? T{} : linalg::dot(b_.row(i), x_.row(j));
      }
    }
    return w;
  }

  auto retrieve_sync(std::vector<State> state, std::size_t max_steps) const -> std::generator<Step> {
    check_state(state);
    co_yield Step{0, state, 0, max_steps == 0 ? Status::max_steps : Status::running};
    if (max_steps == 0) {
      co_return;
    }

    std::vector<T> h(size());
    std::vector<State> next(size());
    std::vector<State> previous;
    for (std::size_t t = 1; t <= max_steps; ++t) {
      local_field(state, h);
      std::size_t changed = 0;
      for (std::size_t i = 0; i < size(); ++i) {
        next[i] = next_state(h[i], state[i]);
        changed += next[i] != state[i];
      }

      auto status = Status::running;
      if (changed == 0) {
        status = Status::stable;
      } else if (next == previous) {
        status = Status::cycle;
      } else if (t == max_steps) {
        status = Status::max_steps;
      }

      previous = state;
      std::swap(state, next);
      co_yield Step{t, state, changed, status};
      if (status != Status::running) {
        co_return;
      }
    }
  }

  auto retrieve_async(std::vector<State> state, std::size_t max_steps, std::uint64_t seed) const
      -> std::generator<Step> {
    check_state(state);
    co_yield Step{0, state, 0, max_steps == 0 ? Status::max_steps : Status::running};
    if (max_steps == 0) {
      co_return;
    }

    std::vector<T> s(pattern_count());
    overlaps(state, s);
    std::vector<std::size_t> order(size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::mt19937_64 rng(seed);

    for (std::size_t t = 1; t <= max_steps; ++t) {
      std::ranges::shuffle(order, rng);
      std::size_t changed = 0;
      for (const auto i : order) {
        const T h = linalg::dot(b_.row(i), std::span<const T>(s)) - d_[i] * static_cast<T>(state[i]);
        const State v = next_state(h, state[i]);
        if (v != state[i]) {
          state[i] = v;
          ++changed;
          linalg::axpy(static_cast<T>(2 * v), x_.row(i), std::span<T>(s));
        }
      }

      auto status = Status::running;
      if (changed == 0) {
        status = Status::stable;
      } else if (t == max_steps) {
        status = Status::max_steps;
      }
      co_yield Step{t, state, changed, status};
      if (status != Status::running) {
        co_return;
      }
    }
  }

private:
  auto overlaps(std::span<const State> x, std::span<T> s) const -> void {
    linalg::gemv_t(x_.view(), x, s);
  }

  auto check_state(std::span<const State> state) const -> void {
    if (state.size() != size()) {
      throw std::invalid_argument(
          std::format("HopfieldNetwork: state has {} neurons, network has {}", state.size(), size()));
    }
  }

  linalg::Matrix<State> x_;
  linalg::Matrix<T> b_;
  std::vector<T> d_;
};

}

namespace hopfield {

template <std::floating_point T>
auto from_coefficients(linalg::Matrix<State> x, const linalg::Matrix<double>& c) -> HopfieldNetwork<T> {
  auto b = linalg::matmul<T>(std::as_const(x).view(), c.view());
  std::vector<T> d(x.rows());
  for (std::size_t i = 0; i < x.rows(); ++i) {
    d[i] = linalg::dot(std::as_const(b).row(i), std::as_const(x).row(i));
  }
  return HopfieldNetwork<T>(std::move(x), std::move(b), std::move(d));
}

auto columns(const linalg::matrix_view<const State>& patterns) -> linalg::Matrix<State> {
  if (patterns.extent(0) == 0 || patterns.extent(1) == 0) {
    throw std::invalid_argument("hopfield: no patterns");
  }
  return linalg::transpose(patterns);
}

}

export namespace hopfield {

template <std::floating_point T>
[[nodiscard]] auto hebb(const linalg::matrix_view<const State>& patterns) -> HopfieldNetwork<T> {
  auto x = columns(patterns);
  const auto n = static_cast<double>(x.rows());
  linalg::Matrix<double> c(x.cols(), x.cols());
  for (std::size_t k = 0; k < x.cols(); ++k) {
    c.view()[k, k] = 1.0 / n;
  }
  return from_coefficients<T>(std::move(x), c);
}

template <std::floating_point T>
[[nodiscard]] auto pseudoinverse(const linalg::matrix_view<const State>& patterns) -> HopfieldNetwork<T> {
  auto x = columns(patterns);
  auto gram = linalg::matmul<double>(patterns, std::as_const(x).view());
  try {
    return from_coefficients<T>(std::move(x), linalg::inverse(std::move(gram)));
  } catch (const std::domain_error&) {
    throw std::domain_error("pseudoinverse: patterns are linearly dependent");
  }
}

}
