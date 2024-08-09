/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/constants.hh
 * Special constants like transposition.
 *
 * @version 1.1.0
 * */

#pragma once

#include <stdexcept>

namespace nntile
{

//! Transposition operation class
//
// Uses predefined constants TransOp::NoTrans and TransOp::Trans
class TransOp
{
public:
    //! Transposition value
    enum Value: int
    {
        NoTrans,
        Trans
    } value;
    //! Constructor for transposition operation object
    constexpr explicit TransOp(const enum TransOp::Value &value_):
        value(value_)
    {
        if(value != TransOp::NoTrans and value != TransOp::Trans)
        {
            throw std::runtime_error("Invalid value of TransOp object");
        }
    }
    //! All constructors but one are disabled
    template<typename T>
    explicit TransOp(const T &) = delete;
    //! All conversions are disabled
    template<typename T>
    operator T() = delete;
};

} // namespace nntile
