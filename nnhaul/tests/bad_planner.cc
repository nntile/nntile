/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file tests/bad_planner.cc
 * A deliberately incompatible planner library: it exports the two entry
 * symbols but declares API version 999. The loader must refuse it.
 *
 * @version 2.0.0
 * */

// SPDX-License-Identifier: MIT

#include <nnhaul/planner.hh>

namespace
{
class Bad final: public nnhaul::planner::Planner
{
  public:
    char const *name() const override { return "test.bad-version"; }
    void plan(nnhaul::PlanRequest const &, nnhaul::planner::Topology const &,
        nnhaul::planner::Sink &) override
    {
    }
};
Bad g_bad;
} // namespace

extern "C" std::uint32_t nnhaul_planner_api_version()
{
    return 999;
}

extern "C" nnhaul::planner::Planner *nnhaul_create_planner()
{
    return &g_bad;
}
