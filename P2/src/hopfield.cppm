module;
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <generator>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <vector>
export module hopfield;

export import linalg;

export namespace hopfield {

enum class LearningMethod { hebb, oja };
enum class Simulation { async, sync };
enum class Status { stable, cycle, max_steps, running };
using State = std::int8_t;

struct Step {
  std::size_t index;
  std::span<const State> state;
  std::size_t changed;
  Status status;
};

template <linalg::Scalar T> class HopfieldNetwork {
public:
  constexpr explicit HopfieldNetwork(linalg::Matrix<T> weights);

  constexpr auto size() const noexcept -> std::size_t;
  constexpr auto weights() const -> linalg::matrix_view<const T>;

  auto retrieve_async(std::vector<State> s, const std::size_t max_steps,
                      const std::uint64_t seed) const -> std::generator<Step>;
  auto retrieve_sync(std::vector<State> s, const std::size_t max_steps) const
      -> std::generator<Step>;

private:
  linalg::Matrix<T> weights_;
};

template <linalg::Scalar T>
auto hebb(const linalg::matrix_view<const State> &patterns)
    -> linalg::Matrix<T>;

// impls

template <linalg::Scalar T>
auto HopfieldNetwork<T>::retrieve_async(std::vector<State> s,
                                        const std::size_t max_steps,
                                        const std::uint64_t seed) const
    -> std::generator<Step> {
  co_yield Step{.index = 0,
                .state = s,
                .changed = 0,
                .status = max_steps == 0 ? Status::max_steps : Status::running};

  if (max_steps == 0)
    co_return;

  const auto n = size();
  std::vector<std::size_t> order(n);
  std::mt19937_64 rng{seed};
  std::ranges::iota(order, 0uz);

  for (const auto step : std::views::iota(0uz, max_steps)) {
    std::ranges::shuffle(order, rng);

    std::size_t changed{0uz};
    for (const auto i : order) {
      const auto hi = linalg::dot(weights_.row(i), std::span<const State>(s));
      const auto new_si = hi < T{} ? State{-1} : hi > T{} ? State{1} : s[i];
      if (new_si != s[i]) {
        s[i] = new_si;
        ++changed;
      }
    }

    co_yield Step{.index = step + 1,
                  .state = s,
                  .changed = changed,
                  .status = changed == 0           ? Status::stable
                            : step < max_steps - 1 ? Status::running
                                                   : Status::max_steps};
    if (changed == 0 || step == max_steps - 1)
      co_return;
  }
}

template <linalg::Scalar T>
auto HopfieldNetwork<T>::retrieve_sync(std::vector<State> s,
                                       const std::size_t max_steps) const
    -> std::generator<Step> {
  co_yield Step{.index = 0,
                .state = s,
                .changed = 0,
                .status = max_steps == 0 ? Status::max_steps : Status::running};

  if (max_steps == 0)
    co_return;

  std::vector<State> s_new(s.size()), s_prev(s.size());
  std::vector<T> h(s.size());
  const auto wv = weights_.view();

  for (auto t{0uz}; t < max_steps; ++t) {
    std::size_t changed = 0;
    bool same_as_prev = true;
    Status status = Status::running;

    linalg::gemv(wv, std::span<const State>(s), std::span<T>(h));
    for (const auto [i, hi] : std::views::enumerate(h)) {
      s_new[i] = hi < T{} ? State{-1} : hi > T{} ? State{1} : s[i];
      if (s_new[i] != s[i])
        ++changed;
      if (t > 0 && s_new[i] != s_prev[i])
        same_as_prev = false;
    }

    if (changed == 0) {
      status = Status::stable;
    } else if (t > 0 && same_as_prev) {
      status = Status::cycle;
    } else if (t == max_steps - 1) {
      status = Status::max_steps;
    }

    std::swap(s_prev, s);
    std::swap(s, s_new);

    co_yield Step{
        .index = t + 1, .state = s, .changed = changed, .status = status};
    if (status != Status::running)
      co_return;
  }
}

template <linalg::Scalar T>
auto hebb(const linalg::matrix_view<const State> &patterns)
    -> linalg::Matrix<T> {
  const auto n_patterns = patterns.extent(0);
  const auto n_neurons = patterns.extent(1);
  const T scale = T{1} / static_cast<T>(n_neurons);

  linalg::Matrix<T> w(n_neurons, n_neurons);
  auto wv = w.view();
  for (auto mu{0uz}; mu < n_patterns; ++mu) {
    const auto xi = linalg::row_of(patterns, mu);
    linalg::rank1_update(wv, scale, xi, xi);
  }
  linalg::zero_diagonal(wv);

  return w;
}

} // namespace hopfield
