/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file tests/test_planner_version.cc
 * The loader refuses a planner built for a different API version, and the
 * refusal names both versions.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include <nnhaul/planner.hh>

#include <cstdio>
#include <cstring>
#include <string>

int main()
{
    bool threw = false;
    std::string what;
    try
    {
        (void)nnhaul::planner::load(NNHAUL_BAD_PLANNER);
    }
    catch(nnhaul::Error const &e)
    {
        threw = true;
        what = e.what();
    }
    if(!threw)
    {
        std::fprintf(stderr, "loader accepted an incompatible planner\n");
        return 1;
    }
    if(what.find("999") == std::string::npos
        || what.find("expects 1") == std::string::npos)
    {
        std::fprintf(stderr, "refusal does not name both versions: %s\n",
            what.c_str());
        return 1;
    }
    std::printf("planner_version: PASS (%s)\n", what.c_str());
    return 0;
}
