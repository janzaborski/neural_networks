#include <charconv>
#include <chrono>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

import hopfield;
import io;

namespace {

using hopfield::State;
using Weight = float;
using Network = hopfield::HopfieldNetwork<Weight>;

constexpr std::string_view usage =
    R"(usage: hopfield_app --patterns FILE --width W --height H --out DIR [options]

required:
  --patterns FILE    training patterns: CSV, one pattern per line, values -1/1
  --width W          bitmap width (columns)
  --height H         bitmap height (rows); each line must have W*H values
  --out DIR          output directory (created if missing)

options:
  --queries FILE     inputs to retrieve from (same format); default: the patterns
  --targets FILE     expected pattern index per query line (-1 = none);
                     default: 0..p-1 when queries are the patterns, otherwise -1
  --rule NAME        hebb | oja                (default hebb)
  --mode NAME        sync | async              (default async)
  --max-steps N      steps (sync) or sweeps (async) per query (default 100)
  --seed S           RNG seed for the async update order (default 0)
  --trajectory       write every step of every query to trajectory.csv
  --weights          write the weight matrix to weights.csv
)";

struct Options {
  std::filesystem::path patterns;
  std::size_t width = 0;
  std::size_t height = 0;
  std::filesystem::path out;
  std::optional<std::filesystem::path> queries;
  std::optional<std::filesystem::path> targets;
  std::string rule = "hebb";
  std::string mode = "async";
  std::size_t max_steps = 100;
  std::uint64_t seed = 0;
  bool trajectory = false;
  bool weights = false;
};

template <typename Int>
auto parse_number(std::string_view name, std::string_view text)
    -> std::expected<Int, std::string> {
  Int value{};
  const auto *end = text.data() + text.size();
  const auto [ptr, ec] = std::from_chars(text.data(), end, value);
  if (ec != std::errc{} || ptr != end) {
    return std::unexpected(std::format("{}: invalid number '{}'", name, text));
  }
  return value;
}

auto parse_args(int argc, char **argv) -> std::expected<Options, std::string> {
  Options opts;
  bool has_width = false;
  bool has_height = false;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];

    if (arg == "--trajectory") {
      opts.trajectory = true;
      continue;
    }
    if (arg == "--weights") {
      opts.weights = true;
      continue;
    }
    if (arg == "-h" || arg == "--help") {
      return std::unexpected(std::string{});
    }
    if (i + 1 >= argc) {
      return std::unexpected(std::format("{}: missing value", arg));
    }
    const std::string_view value = argv[++i];

    if (arg == "--patterns") {
      opts.patterns = value;
    } else if (arg == "--out") {
      opts.out = value;
    } else if (arg == "--queries") {
      opts.queries = value;
    } else if (arg == "--targets") {
      opts.targets = value;
    } else if (arg == "--rule") {
      opts.rule = value;
    } else if (arg == "--mode") {
      opts.mode = value;
    } else if (arg == "--width") {
      auto n = parse_number<std::size_t>(arg, value);
      if (!n)
        return std::unexpected(n.error());
      opts.width = *n;
      has_width = true;
    } else if (arg == "--height") {
      auto n = parse_number<std::size_t>(arg, value);
      if (!n)
        return std::unexpected(n.error());
      opts.height = *n;
      has_height = true;
    } else if (arg == "--max-steps") {
      auto n = parse_number<std::size_t>(arg, value);
      if (!n)
        return std::unexpected(n.error());
      opts.max_steps = *n;
    } else if (arg == "--seed") {
      auto n = parse_number<std::uint64_t>(arg, value);
      if (!n)
        return std::unexpected(n.error());
      opts.seed = *n;
    } else {
      return std::unexpected(std::format("unknown option {}", arg));
    }
  }

  if (opts.patterns.empty() || opts.out.empty() || !has_width || !has_height) {
    return std::unexpected(
        "--patterns, --width, --height and --out are required");
  }
  if (opts.width == 0 || opts.height == 0) {
    return std::unexpected("--width and --height must be positive");
  }
  if (opts.rule != "hebb" && opts.rule != "oja") {
    return std::unexpected(std::format("--rule: unknown rule '{}'", opts.rule));
  }
  if (opts.mode != "sync" && opts.mode != "async") {
    return std::unexpected(std::format("--mode: unknown mode '{}'", opts.mode));
  }
  if (opts.targets && !opts.queries) {
    return std::unexpected("--targets requires --queries");
  }
  return opts;
}

auto status_name(hopfield::Status status) -> std::string_view {
  switch (status) {
  case hopfield::Status::stable:
    return "stable";
  case hopfield::Status::cycle:
    return "cycle";
  case hopfield::Status::max_steps:
    return "max_steps";
  case hopfield::Status::running:
    return "running";
  }
  return "unknown";
}

auto local_field(const Network &net, std::span<const State> s,
                 std::vector<Weight> &h) -> void {
  linalg::gemv(net.weights(), s, std::span<Weight>(h));
}

auto energy(const Network &net, std::span<const State> s,
            std::vector<Weight> &h) -> double {
  local_field(net, s, h);
  return -0.5 * static_cast<double>(linalg::dot(std::span<const Weight>(h), s));
}

auto unstable_neurons(const Network &net, std::span<const State> s,
                      std::vector<Weight> &h) -> std::size_t {
  local_field(net, s, h);
  std::size_t count = 0;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if ((h[i] < Weight{} && s[i] > 0) || (h[i] > Weight{} && s[i] < 0)) {
      ++count;
    }
  }
  return count;
}

struct Outcome {
  std::vector<State> final_state;
  hopfield::Status status = hopfield::Status::running;
  std::size_t steps = 0;
};

auto run_query(const Network &net, hopfield::Simulation mode,
               std::vector<State> state, std::size_t max_steps,
               std::uint64_t seed, std::size_t query, std::ofstream *trajectory,
               std::vector<Weight> &h) -> Outcome {
  Outcome outcome;
  auto consume = [&](auto &&steps) {
    for (const auto &step : steps) {
      if (trajectory != nullptr) {
        *trajectory << std::format("{},{},{},{},{},{}\n", query, step.index,
                                   status_name(step.status), step.changed,
                                   energy(net, step.state, h),
                                   io::encode_state(step.state));
      }
      outcome.status = step.status;
      outcome.steps = step.index;
      outcome.final_state.assign(step.state.begin(), step.state.end());
    }
  };

  if (mode == hopfield::Simulation::sync) {
    consume(net.retrieve_sync(std::move(state), max_steps));
  } else {
    consume(net.retrieve_async(std::move(state), max_steps, seed));
  }
  return outcome;
}

auto run(const Options &opts) -> int {
  const auto start = std::chrono::steady_clock::now();
  const auto n = opts.width * opts.height;

  auto patterns = io::load_states(opts.patterns, n);
  if (!patterns) {
    std::cerr << "error: " << patterns.error() << '\n';
    return 1;
  }
  const auto n_patterns = patterns->rows();

  auto queries = opts.queries ? io::load_states(*opts.queries, n)
                              : io::load_states(opts.patterns, n);
  if (!queries) {
    std::cerr << "error: " << queries.error() << '\n';
    return 1;
  }
  const auto n_queries = queries->rows();

  std::vector<long> targets(n_queries, -1);
  if (opts.targets) {
    auto loaded = io::load_targets(*opts.targets, n_queries);
    if (!loaded) {
      std::cerr << "error: " << loaded.error() << '\n';
      return 1;
    }
    targets = std::move(*loaded);
  } else if (!opts.queries) {
    for (std::size_t q = 0; q < n_queries; ++q) {
      targets[q] = static_cast<long>(q);
    }
  }
  for (const auto t : targets) {
    if (t >= static_cast<long>(n_patterns)) {
      std::cerr << std::format("error: target {} out of range (patterns: {})\n",
                               t, n_patterns);
      return 1;
    }
  }

  if (opts.rule == "oja") {
    std::cerr << "error: the Oja rule is not implemented yet\n";
    return 1;
  }
  const Network net{hopfield::hebb<Weight>(std::as_const(*patterns).view())};
  const auto mode = opts.mode == "sync" ? hopfield::Simulation::sync
                                        : hopfield::Simulation::async;

  std::error_code ec;
  std::filesystem::create_directories(opts.out, ec);
  if (ec) {
    std::cerr << std::format("error: {}: {}\n", opts.out.string(),
                             ec.message());
    return 1;
  }

  std::vector<Weight> h(n);

  auto stability_file = io::open_output(opts.out / "stability.csv");
  if (!stability_file) {
    std::cerr << "error: " << stability_file.error() << '\n';
    return 1;
  }
  *stability_file << "pattern,unstable_neurons,stable\n";
  std::size_t stable_patterns = 0;
  for (std::size_t mu = 0; mu < n_patterns; ++mu) {
    const auto unstable =
        unstable_neurons(net, std::as_const(*patterns).row(mu), h);
    stable_patterns += unstable == 0;
    *stability_file << std::format("{},{},{}\n", mu, unstable,
                                   unstable == 0 ? 1 : 0);
  }

  if (opts.weights) {
    auto weights_file = io::open_output(opts.out / "weights.csv");
    if (!weights_file) {
      std::cerr << "error: " << weights_file.error() << '\n';
      return 1;
    }
    const auto w = net.weights();
    for (std::size_t i = 0; i < w.extent(0); ++i) {
      for (std::size_t j = 0; j < w.extent(1); ++j) {
        *weights_file << (j == 0 ? "" : ",") << std::format("{}", w[i, j]);
      }
      *weights_file << '\n';
    }
  }

  std::optional<std::ofstream> trajectory;
  if (opts.trajectory) {
    auto file = io::open_output(opts.out / "trajectory.csv");
    if (!file) {
      std::cerr << "error: " << file.error() << '\n';
      return 1;
    }
    trajectory = std::move(*file);
    *trajectory << "query,step,status,changed,energy,state\n";
  }

  auto results_file = io::open_output(opts.out / "results.csv");
  if (!results_file) {
    std::cerr << "error: " << results_file.error() << '\n';
    return 1;
  }
  *results_file
      << "query,target,status,steps,category,best_pattern,best_overlap,"
         "target_overlap_initial,hamming_initial,target_overlap,hamming_to_"
         "target,"
         "energy_initial,energy_final\n";

  std::map<std::string, std::size_t> categories;
  std::map<std::string, std::size_t> statuses;
  std::size_t total_steps = 0;
  std::size_t with_target = 0;
  std::size_t recalled = 0;
  double total_target_overlap = 0.0;
  double total_initial_overlap = 0.0;
  const auto n_signed = static_cast<long>(n);

  for (std::size_t q = 0; q < n_queries; ++q) {
    const auto input = std::as_const(*queries).row(q);
    const auto energy_initial = energy(net, input, h);
    const auto seed = opts.seed * 0x9E3779B97F4A7C15ULL + q;

    auto outcome = run_query(
        net, mode, std::vector<State>(input.begin(), input.end()),
        opts.max_steps, seed, q, trajectory ? &*trajectory : nullptr, h);
    const std::span<const State> final_state(outcome.final_state);
    const auto energy_final = energy(net, final_state, h);

    std::size_t best = 0;
    long best_overlap = 0;
    for (std::size_t mu = 0; mu < n_patterns; ++mu) {
      const long m = linalg::dot(std::as_const(*patterns).row(mu), final_state);
      if (std::abs(m) > std::abs(best_overlap)) {
        best = mu;
        best_overlap = m;
      }
    }

    const auto target = targets[q];
    long initial_overlap = 0;
    long target_overlap = 0;
    std::string category;
    if (target >= 0) {
      const auto source =
          std::as_const(*patterns).row(static_cast<std::size_t>(target));
      initial_overlap = linalg::dot(source, input);
      target_overlap = linalg::dot(source, final_state);
      total_initial_overlap +=
          static_cast<double>(initial_overlap) / static_cast<double>(n);
      ++with_target;
      total_target_overlap +=
          static_cast<double>(target_overlap) / static_cast<double>(n);
      if (target_overlap == n_signed) {
        category = "target";
        ++recalled;
      } else if (target_overlap == -n_signed) {
        category = "target_inverse";
      }
    }
    if (category.empty()) {
      if (best_overlap == n_signed) {
        category = "other_pattern";
      } else if (best_overlap == -n_signed) {
        category = "other_inverse";
      } else {
        category = "spurious";
      }
    }

    ++categories[category];
    ++statuses[std::string(status_name(outcome.status))];
    total_steps += outcome.steps;

    auto overlap_text = [&](long m) {
      return target >= 0 ? std::format("{}", static_cast<double>(m) /
                                                 static_cast<double>(n))
                         : std::string{};
    };
    auto hamming_text = [&](long m) {
      return target >= 0 ? std::format("{}", (n_signed - m) / 2)
                         : std::string{};
    };
    *results_file << std::format(
        "{},{},{},{},{},{},{},{},{},{},{},{},{}\n", q, target,
        status_name(outcome.status), outcome.steps, category, best,
        static_cast<double>(best_overlap) / static_cast<double>(n),
        overlap_text(initial_overlap), hamming_text(initial_overlap),
        overlap_text(target_overlap), hamming_text(target_overlap),
        energy_initial, energy_final);
  }

  const auto seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();

  auto json_counts = [](const std::map<std::string, std::size_t> &counts) {
    std::string out = "{";
    bool first = true;
    for (const auto &[name, count] : counts) {
      out += std::format("{}{}: {}", first ? "" : ", ", io::json_string(name),
                         count);
      first = false;
    }
    return out + "}";
  };

  auto summary_file = io::open_output(opts.out / "summary.json");
  if (!summary_file) {
    std::cerr << "error: " << summary_file.error() << '\n';
    return 1;
  }
  *summary_file
      << "{\n"
      << std::format("  \"patterns\": {},\n",
                     io::json_string(opts.patterns.string()))
      << std::format(
             "  \"queries\": {},\n",
             io::json_string(opts.queries ? opts.queries->string() : ""))
      << std::format("  \"rule\": {},\n", io::json_string(opts.rule))
      << std::format("  \"mode\": {},\n", io::json_string(opts.mode))
      << std::format("  \"width\": {},\n  \"height\": {},\n", opts.width,
                     opts.height)
      << std::format("  \"n_neurons\": {},\n  \"n_patterns\": {},\n", n,
                     n_patterns)
      << std::format("  \"stable_patterns\": {},\n", stable_patterns)
      << std::format("  \"n_queries\": {},\n", n_queries)
      << std::format("  \"queries_with_target\": {},\n", with_target)
      << std::format("  \"recalled\": {},\n", recalled)
      << std::format("  \"recall_rate\": {},\n",
                     with_target > 0 ? static_cast<double>(recalled) /
                                           static_cast<double>(with_target)
                                     : 0.0)
      << std::format("  \"mean_initial_target_overlap\": {},\n",
                     with_target > 0 ? total_initial_overlap /
                                           static_cast<double>(with_target)
                                     : 0.0)
      << std::format("  \"mean_target_overlap\": {},\n",
                     with_target > 0 ? total_target_overlap /
                                           static_cast<double>(with_target)
                                     : 0.0)
      << std::format("  \"mean_steps\": {},\n",
                     static_cast<double>(total_steps) /
                         static_cast<double>(n_queries))
      << std::format("  \"categories\": {},\n", json_counts(categories))
      << std::format("  \"statuses\": {},\n", json_counts(statuses))
      << std::format("  \"max_steps\": {},\n  \"seed\": {},\n", opts.max_steps,
                     opts.seed)
      << std::format("  \"seconds\": {}\n", seconds) << "}\n";

  std::cout << std::format("{}: {} {} stable {}/{} recalled {}/{} ({:.2f}s)\n",
                           opts.out.string(), opts.rule, opts.mode,
                           stable_patterns, n_patterns, recalled, with_target,
                           seconds);
  return 0;
}

} // namespace

auto main(int argc, char **argv) -> int {
  auto opts = parse_args(argc, argv);
  if (!opts) {
    if (!opts.error().empty()) {
      std::cerr << "error: " << opts.error() << "\n\n";
    }
    std::cerr << usage;
    return opts.error().empty() ? 0 : 2;
  }
  return run(*opts);
}
