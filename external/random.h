/**
 *
 * @file random.h
 *
 * @copyright 2009-2014 The University of Tennessee and The University of
 *                      Tennessee Research Foundation. All rights reserved.
 * @copyright 2012-2022 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 ***
 *
 * @brief Chameleon coreblas random number generator
 *
 * @version 1.2.0
 * @author Piotr Luszczek
 * @author Mathieu Faverge
 * @date 2022-02-22
 * @precisions normal z -> c d s
 *
 */
#ifndef _chameleon_random_h_
#define _chameleon_random_h_

/*
 Rnd64seed is a global variable but it doesn't spoil thread safety. All matrix
 generating threads only read Rnd64seed. It is safe to set Rnd64seed before
 and after any calls to create_tile(). The only problem can be caused if
 Rnd64seed is changed during the matrix generation time.
 */

//static unsigned long long int Rnd64seed = 100;
#define Rnd64_A 6364136223846793005ULL
#define Rnd64_C 1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20

static inline unsigned long long int
CORE_rnd64_jump(unsigned long long int n, unsigned long long int seed ) {
    unsigned long long int a_k, c_k, ran;
    int i;

    a_k = Rnd64_A;
    c_k = Rnd64_C;

    // NNTile requires 2 uniform random numbers per each normal number
    n <<= 1;

    ran = seed;
    for (i = 0; n; n >>= 1, ++i) {
        if (n & 1) {
            ran = a_k * ran + c_k;
        }
        c_k *= (a_k + 1);
        a_k *= a_k;
    }

    return ran;
}

static inline float
CORE_slaran( unsigned long long int *ran )
{
    //float value = 0.5 - (*ran) * RndF_Mul;
    float value = (*ran) * RndF_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    return value;
}

static inline double
CORE_dlaran( unsigned long long int *ran )
{
    //double value = 0.5 - (*ran) * RndD_Mul;
    double value = (*ran) * RndD_Mul;
    *ran = Rnd64_A * (*ran) + Rnd64_C;

    return value;
}

//static inline CHAMELEON_Complex32_t
//CORE_claran( unsigned long long int *ran )
//{
//    CHAMELEON_Complex32_t value;
//
//    value  = 0.5 - (*ran) * RndF_Mul;
//    *ran   = Rnd64_A * (*ran) + Rnd64_C;
//
//    value += I * (0.5 - (*ran) * RndF_Mul);
//    *ran   = Rnd64_A * (*ran) + Rnd64_C;
//
//    return value;
//}
//
//static inline CHAMELEON_Complex64_t
//CORE_zlaran( unsigned long long int *ran )
//{
//    CHAMELEON_Complex64_t value;
//
//    value  = 0.5 - (*ran) * RndD_Mul;
//    *ran   = Rnd64_A * (*ran) + Rnd64_C;
//
//    value += I * (0.5 - (*ran) * RndD_Mul);
//    *ran   = Rnd64_A * (*ran) + Rnd64_C;
//
//    return value;
//}

#endif /* _chameleon_random_h_ */
