/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file examples/json_config_helpers.hh
 * Shared JSON config field parsers for C++ examples.
 *
 * @version 1.1.0
 * */

#pragma once

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>

namespace nntile::examples
{

//! Read an integer field; accepts JSON number or string (e.g. ``"32000"``).
inline int config_get_int(
    nlohmann::json const &j,
    char const *key,
    int default_val)
{
    if (!j.contains(key))
    {
        return default_val;
    }
    nlohmann::json const &v = j[key];
    if (v.is_number_integer())
    {
        return v.get<int>();
    }
    if (v.is_number_float())
    {
        return static_cast<int>(v.get<double>());
    }
    if (v.is_string())
    {
        return std::stoi(v.get<std::string>());
    }
    throw std::runtime_error(
        std::string("config: '") + key + "' must be int or string, got " +
        v.type_name());
}

//! Read a float field; accepts JSON number or string.
inline float config_get_float(
    nlohmann::json const &j,
    char const *key,
    float default_val)
{
    if (!j.contains(key))
    {
        return default_val;
    }
    nlohmann::json const &v = j[key];
    if (v.is_number_integer() || v.is_number_float())
    {
        return static_cast<float>(v.get<double>());
    }
    if (v.is_string())
    {
        return std::stof(v.get<std::string>());
    }
    throw std::runtime_error(
        std::string("config: '") + key + "' must be number or string, got " +
        v.type_name());
}

//! Read a boolean field.
inline bool config_get_bool(
    nlohmann::json const &j,
    char const *key,
    bool default_val)
{
    if (!j.contains(key))
    {
        return default_val;
    }
    nlohmann::json const &v = j[key];
    if (v.is_boolean())
    {
        return v.get<bool>();
    }
    throw std::runtime_error(
        std::string("config: '") + key + "' must be bool, got " + v.type_name());
}

} // namespace nntile::examples
