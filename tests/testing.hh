/*! @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 * @file tests/testing.hh
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2021-08-19
 * */

#pragma once

#include <stdexcept>

// Check an exception-throwing expression
#define TEST_THROW(...)\
{\
    /* Init no exception was caught */\
    bool caught = false;\
    /* Try to evaluate the expression */\
    try\
    {\
        __VA_ARGS__;\
    }\
    catch(...)\
    {\
        /* The expression did throw an exception */\
        caught = true;\
    }\
    /* Throw an exception if the expression did not throw anything */\
    if(!caught)\
    {\
        throw std::runtime_error(#__VA_ARGS__ " did not throw any exception, "\
                "while it was claimed to raise some exception");\
    }\
}

#define TEST_ASSERT(...)\
{\
    /* Evaluate input */\
    bool eval = __VA_ARGS__;\
    /* Throw exception if assert failed */\
    if(!eval)\
    {\
        throw std::runtime_error("Assert failed");\
    }\
}\

