module;

#include <charconv>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

export module io;

import hopfield;

namespace {

auto trim(std::string_view s) -> std::string_view {
  constexpr std::string_view whitespace = " \t\r\n";
  const auto first = s.find_first_not_of(whitespace);
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = s.find_last_not_of(whitespace);
  return s.substr(first, last - first + 1);
}

template <typename Int>
auto parse_int(std::string_view s, Int& out) -> bool {
  const auto* end = s.data() + s.size();
  const auto [ptr, ec] = std::from_chars(s.data(), end, out);
  return ec == std::errc{} && ptr == end;
}

auto read_file(const std::filesystem::path& path) -> std::expected<std::string, std::string> {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return std::unexpected(std::format("{}: cannot open file", path.string()));
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

}

export namespace io {

using hopfield::State;

auto load_states(const std::filesystem::path& path, std::size_t n_values)
    -> std::expected<linalg::Matrix<State>, std::string> {
  auto text = read_file(path);
  if (!text) {
    return std::unexpected(text.error());
  }

  std::vector<State> data;
  std::size_t rows = 0;
  std::size_t line_no = 0;
  std::istringstream lines(*text);
  std::string line;

  while (std::getline(lines, line)) {
    ++line_no;
    const auto content = trim(line);
    if (content.empty()) {
      continue;
    }

    std::size_t count = 0;
    for (const auto part : content | std::views::split(',')) {
      const auto field = trim(std::string_view(part.begin(), part.end()));
      int value = 0;
      if (!parse_int(field, value) || (value != 1 && value != -1)) {
        return std::unexpected(std::format("{}:{}: value '{}' is not -1 or 1", path.string(),
                                           line_no, std::string(field)));
      }
      data.push_back(static_cast<State>(value));
      ++count;
    }

    if (count != n_values) {
      return std::unexpected(std::format("{}:{}: expected {} values, got {}", path.string(),
                                         line_no, n_values, count));
    }
    ++rows;
  }

  if (rows == 0) {
    return std::unexpected(std::format("{}: no data rows", path.string()));
  }
  return linalg::Matrix<State>(linalg::extents{rows, n_values}, std::move(data));
}

auto load_targets(const std::filesystem::path& path, std::size_t n_rows)
    -> std::expected<std::vector<long>, std::string> {
  auto text = read_file(path);
  if (!text) {
    return std::unexpected(text.error());
  }

  std::vector<long> targets;
  std::size_t line_no = 0;
  std::istringstream lines(*text);
  std::string line;

  while (std::getline(lines, line)) {
    ++line_no;
    const auto content = trim(line);
    if (content.empty()) {
      continue;
    }
    long value = 0;
    if (!parse_int(content, value) || value < -1) {
      return std::unexpected(std::format("{}:{}: invalid target '{}'", path.string(), line_no,
                                         std::string(content)));
    }
    targets.push_back(value);
  }

  if (targets.size() != n_rows) {
    return std::unexpected(std::format("{}: expected {} targets, got {}", path.string(), n_rows,
                                       targets.size()));
  }
  return targets;
}

auto open_output(const std::filesystem::path& path) -> std::expected<std::ofstream, std::string> {
  std::ofstream out(path);
  if (!out) {
    return std::unexpected(std::format("{}: cannot open for writing", path.string()));
  }
  return out;
}

auto encode_state(std::span<const State> state) -> std::string {
  std::string encoded(state.size(), '0');
  for (std::size_t i = 0; i < state.size(); ++i) {
    if (state[i] > 0) {
      encoded[i] = '1';
    }
  }
  return encoded;
}

auto json_string(std::string_view s) -> std::string {
  std::string out = "\"";
  for (const char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        out += std::format("\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(c)));
      } else {
        out += c;
      }
    }
  }
  out += '"';
  return out;
}

}
