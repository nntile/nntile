/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file tests/graph/model/test_frobenius.hh
 * Relative Frobenius helpers for graph model tests (implementation in pytorch_helper.hh).
 *
 * @version 1.1.0
 * */

#pragma once

#include "../nn/pytorch_helper.hh"

using nntile::test::relative_frobenius_error;
using nntile::test::require_relative_frobenius_error;
