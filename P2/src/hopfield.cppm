module;
#include <cstdint>
export module hopfield;

export import linalg;

export namespace hopfield {

enum class LearningMethod { hebb, oja };
enum class Simulation { async, sync };

using State = std::int8_t;

template <linalg::Scalar T> class HopfieldNetwork {
public:
  constexpr explicit HopfieldNetwork(linalg::Matrix<T> weights);

  constexpr auto size() const noexcept -> std::size_t;
  constexpr auto weights() const -> linalg::matrix_view<const T>;

private:
  linalg::Matrix<T> weights_;
};

template <linalg::Scalar T> auto hebb() -> linalg::Matrix<T>;

} // namespace hopfield
