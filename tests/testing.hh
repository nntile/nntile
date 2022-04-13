/*! @copyright (c) 2016-2021 King Abdullah University of Science and
 *                           Technology (KAUST). All rights reserved.
 * @copyright (c) 2020-2021 RWTH Aachen. All rights reserved.
 *
 * STARS-H-core is a software package, provided by King Abdullah
 *             University of Science and Technology (KAUST)
 *
 * @file tests/testing.hh
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2021-01-01
 * */

#include <limits>
#include <iostream>
#include <exception>

// Test, that must contain an error
#define TESTN(...)\
{\
    bool error=false;\
    try{\
        __VA_ARGS__;\
        error = true;\
    }\
    catch(...)\
    {\
    }\
    if (error)\
    {\
        throw std::runtime_error(#__VA_ARGS__ " is positive, but "\
            "claimed as negative");\
    }\
}

// Test, that must not contain an error
#define TESTP(...)\
{\
    try\
    {\
        __VA_ARGS__;\
    }\
    catch(const std::exception &e)\
    {\
        std::cerr << e.what() << std::endl;\
        throw std::runtime_error(#__VA_ARGS__ " is negative, but "\
            "claimed as positive");\
    }\
    catch(...)\
    {\
        std::cerr << "Caught unexpected error" << std::endl;\
        throw std::runtime_error(#__VA_ARGS__ " is negative, but "\
            "claimed as positive");\
    }\
}

//! Macro to convert integer from __LINE__ to a string
#define LINE STRINGIZE1(__LINE__)
#define STRINGIZE1(X) STRINGIZE2(X)
#define STRINGIZE2(X) #X

// Simple assert
#define TESTA(cond)\
{\
    if(!bool(cond))\
    {\
        std::cerr << "Assertion failed on line " LINE " of file \""\
            __FILE__ "\"" << std::endl;\
        std::terminate();\
    }\
}
