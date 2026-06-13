/**
 * @file test_main.cpp
 * @brief Test runner for the RISC-V microarchitecture engine.
 *
 * @par Usage
 * @code
 *   ./rvsim_tests         - run all tests
 *   ./rvsim_tests [str]   - run only tests whose name containts "str"
 * @endcode
 */

#include "test_framework.hpp"

std::vector<TestCase>& get_tests()
{
    static std::vector<TestCase> tests;
    return tests;
}

int main(int argc, char** argv)
{
    int passed  = 0;
    int failed  = 0;
    int skipped = 0;

    std::string filter = (argc > 1) ? argv[1] : "";

    for (const auto& test : get_tests())
    {
        if (!filter.empty() && test.name.find(filter) == std::string::npos)
        {
            ++skipped;
            continue;
        }

        std::cout << "Running: " << test.name << "... " << std::flush;
        try
        {
            if (test.func())
            {
                std::cout << "PASSED\n";
                ++passed;
            }
            else
            {
                std::cout << "FAILED\n";
                ++failed;
            }
        }
        catch (const std::exception& e)
        {
            std::cout << "EXCEPTION: " << e.what() << "\n";
            ++failed;
        }
    }

    std::cout << std::format("\n{} passed, {} failed, {} skipped (of {} total)\n",
                             passed, failed, skipped, get_tests().size());
    return failed > 0 ? 1 : 0;
}
