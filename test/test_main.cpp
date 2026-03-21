/**
 * @file test_main.cpp
 * @brief Minimal test framework and runner for the RV32I simulator.
 *
 * Tests self-register via static initialization of a global vector.
 * Each test file includes the macros below and pushes TestCase entries
 * into g_tests before main() runs.
 *
 * Usage:
 *   ./tests         - run all tests
 *   ./tests decode  - run only tests whose name containts "decode"
 */

#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

// ── Test infrastructure (shared across all test TUs) ───────────────────

struct TestCase {
    std::string           name;
    std::function<bool()> func;
};

std::vector<TestCase> g_tests;

// ── Assertion macros ───────────────────────────────────────────────────

/// General boolean assertion.
#define ASSERT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << " FAILED: " << #cond << "\n"                      \
                      << "   at " << __FILE__ << ":" << __LINE__ << "\n";  \
            return false;                                                  \
        }                                                                  \
    } while(0)

/// Equality assertion - prints both values on failure (decimal).
#define ASSERT_EQ(a, b)                                                      \
    do {                                                                     \
        auto actual_   = (a);                                                \
        auto expected_ = (b);                                                \
        if (actual_ != expected_) {                                          \
            std::cerr << "  FAILED: " << #a << " == " << #b << "\n"          \
                      << "    got: " << static_cast<std::int64_t>(actual_)   \
                      << " != "      << static_cast<std::int64_t>(expected_) \
                      << "\n"                                                \
                      << "    at " << __FILE__ << ":" << __LINE__ << "\n";   \
            return false;                                                    \
        }                                                                    \
    } while(0)

/// Equality assertion - prints both values in hex.
#define ASSERT_HEX_EQ(a, b)                                                 \
    do {                                                                    \
        auto actual_   = (a);                                               \
        auto expected_ = (b);                                               \
        if (actual_ != expected_) {                                         \
            std::cerr << "  FAILED: " << #a << " == " << #b << "\n"         \
                      << std::format("    got: 0x{:x} != 0x{:x}\n",         \
                            static_cast<std::uint64_t>(actual_),            \
                            static_cast<std::uint64_t>(expected_))          \
                      << "    at " << __FILE__ << ":" << __LINE__ << "\n";  \
            return false;                                                   \
        }                                                                   \
    } while (0)

/// Self-registration macro. Each TEST(name) block is a function that
/// returns true on success and false on failure.
#define TEST(name)                                                             \
    bool test_##name();                                                        \
    static bool _reg_##name = (g_tests.push_back({#name, test_##name}), true); \
    bool test_##name()

// ── Runner ─────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int passed  = 0;
    int failed  = 0;
    int skipped = 0;

    std::string filter = (argc > 1) ? argv[1] : "";

    for (const auto& test : g_tests) {
        if (!filter.empty() && test.name.find(filter) == std::string::npos) {
            ++skipped;
            continue;
        }

        std::cout << "Running: " << test.name << "... " << std::flush;
        try {
            if (test.func()) {
                std::cout << "PASSED\n";
                ++passed;
            } else {
                std::cout << "FAILED\n";
                ++failed;
            }
        } catch (const std::exception& e) {
            std::cout << "EXCEPTION: " << e.what() << "\n";
            ++failed;
        }
    }

    std::cout << std::format("\n{} passed, {} failed, {} skipped (of {} total)\n",
                             passed, failed, skipped, g_tests.size());
    return failed > 0 ? 1 : 0;
}
