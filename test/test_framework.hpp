/**
 * @file test_framework.hpp
 * @brief Shared test infrastructure.
 */

#pragma once

#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

struct TestCase
{
    std::string name;
    std::function<bool()> func;
};

std::vector<TestCase>& get_tests();

/// General boolean assertion.
#define ASSERT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << " FAILED: " << #cond << "\n"                      \
                      << "   at " << __FILE__ << ":" << __LINE__ << "\n";  \
            return false;                                                  \
        }                                                                  \
    } while (0)

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
    } while (0)

/// Equality assertion - on failure, prints both values in hex.
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
#define TEST(name)                                                                \
    bool test_##name();                                                           \
    static bool reg_##name = (get_tests().push_back({#name, test_##name}), true); \
    bool test_##name()
