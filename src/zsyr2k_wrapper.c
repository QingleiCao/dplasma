/*
 * Copyright (c) 2010-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> z c d s
 *
 */

#include "dplasma.h"
#include "dplasma/types.h"
#include "dplasmaaux.h"

#include "zsyr2k_LN.h"
#include "zsyr2k_LT.h"
#include "zsyr2k_UN.h"
#include "zsyr2k_UT.h"

/**
 ******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zsyr2k_New - Generates the parsec taskpool to performs one of the
 *  syrmitian rank 2k operations
 *
 *    \f[ C = \alpha [ op( A ) \times conjg( op( B )' )] + conjg( \alpha ) [ op( B ) \times conjg( op( A )' )] + \beta C \f],
 *    or
 *    \f[ C = \alpha [ conjg( op( A )' ) \times op( B ) ] + conjg( \alpha ) [ conjg( op( B )' ) \times op( A ) ] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = conjg( X' )
 *
 *  where alpha and beta are real scalars, C is an n-by-n symmetric
 *  matrix and A and B are an n-by-k matrices the first case and k-by-n
 *  matrices in the second case.
 *
 *  WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of C is stored;
 *          = dplasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is not transposed or conjugate transposed:
 *          = dplasmaNoTrans: \f[ C = \alpha [ op( A )  \times op( B )' ] + conjg( \alpha ) [ op( B )  \times op( A )' ] + \beta C \f]
 *          = dplasmaTrans:   \f[ C = \alpha [ op( A )' \times op( B )  ] + conjg( \alpha ) [ op( B )' \times op( A )  ] + \beta C \f]
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B
 *
 * @param[in] beta
 *          beta specifies the scalar beta.
 *
 * @param[in,out] C
 *          Descriptor of the symmetric matrix C.
 *          On exit, the uplo part of the matrix described by C is overwritten
 *          by the result of the operation.
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_zsyr2k_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zsyr2k
 * @sa dplasma_zsyr2k_Destruct
 * @sa dplasma_csyr2k_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zsyr2k_New( dplasma_enum_t uplo,
                    dplasma_enum_t trans,
                    dplasma_complex64_t alpha,
                    const parsec_tiled_matrix_t* A,
                    const parsec_tiled_matrix_t* B,
                    dplasma_complex64_t beta,
                    parsec_tiled_matrix_t* C)
{
    parsec_taskpool_t* tp;

    /* Check input arguments */
    if ((uplo != dplasmaLower) && (uplo != dplasmaUpper)) {
        dplasma_error("dplasma_zsyr2k_New", "illegal value of uplo");
        return NULL;
    }
    if (trans != dplasmaConjTrans && trans != dplasmaTrans && trans != dplasmaNoTrans ) {
        dplasma_error("dplasma_zsyr2k_New", "illegal value of trans");
        return NULL;
    }

    if ( C->m != C->n ) {
        dplasma_error("dplasma_zsyr2k_New", "illegal descriptor C (C->m != C->n)");
        return NULL;
    }
    if ( A->m != B->m || A->n != B->n ) {
        dplasma_error("dplasma_zsyr2k_New", "illegal descriptor A or B, they must have the same dimensions");
        return NULL;
    }
    if ( (( trans == dplasmaNoTrans ) && ( A->m != C->m ))
         || (( trans != dplasmaNoTrans ) && ( A->n != C->m )) ) {
        dplasma_error("dplasma_zsyr2k_New", "illegal sizes for the matrices");
        return NULL;
    }

    if ( uplo == dplasmaLower ) {
        if ( trans == dplasmaNoTrans ) {
            tp = (parsec_taskpool_t*)
                parsec_zsyr2k_LN_new(uplo, trans,
                                    alpha, A,
                                           B,
                                    beta,  C);
        }
        else {
            tp = (parsec_taskpool_t*)
                parsec_zsyr2k_LT_new(uplo, trans,
                                    alpha, A,
                                           B,
                                    beta,  C);
        }
    }
    else {
        if ( trans == dplasmaNoTrans ) {
            tp = (parsec_taskpool_t*)
                parsec_zsyr2k_UN_new(uplo, trans,
                                    alpha, A,
                                           B,
                                    beta,  C);
        }
        else {
            tp = (parsec_taskpool_t*)
                parsec_zsyr2k_UT_new(uplo, trans,
                                    alpha, A,
                                           B,
                                    beta,  C);
        }
    }

    dplasma_add2arena_tile( &((parsec_zsyr2k_LN_taskpool_t*)tp)->arenas_datatypes[PARSEC_zsyr2k_LN_DEFAULT_ADT_IDX],
                            C->mb*C->nb*sizeof(dplasma_complex64_t),
                            PARSEC_ARENA_ALIGNMENT_SSE,
                            parsec_datatype_double_complex_t, C->mb );

    return tp;
}

/***************************************************************************//**
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zsyr2k_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zsyr2k_New().
 *
 *******************************************************************************
 *
 * @param[in] tp
 *          taskpool to destroy.
 *
 *******************************************************************************
 *
 * @sa dplasma_zsyr2k_New
 * @sa dplasma_zsyr2k
 *
 ******************************************************************************/
void
dplasma_zsyr2k_Destruct( parsec_taskpool_t *tp )
{
    parsec_zsyr2k_LN_taskpool_t *zsyr2k_tp = (parsec_zsyr2k_LN_taskpool_t*)tp;
    dplasma_matrix_del2arena( &zsyr2k_tp->arenas_datatypes[PARSEC_zsyr2k_LN_DEFAULT_ADT_IDX] );
    parsec_taskpool_free(tp);
}

/**
 ******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zsyr2k - Performs one of the symmetric rank 2k operations
 *
 *    \f[ C = \alpha [ op( A ) \times conjg( op( B )' )] + conjg( \alpha ) [ op( B ) \times conjg( op( A )' )] + \beta C \f],
 *    or
 *    \f[ C = \alpha [ conjg( op( A )' ) \times op( B ) ] + conjg( \alpha ) [ conjg( op( B )' ) \times op( A ) ] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = conjg( X' )
 *
 *  where alpha and beta are real scalars, C is an n-by-n symmetric
 *  matrix and A and B are an n-by-k matrices the first case and k-by-n
 *  matrices in the second case.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of C is stored;
 *          = dplasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   \f[ C = \alpha [ op( A ) \times conjg( op( B )' )] + conjg( \alpha ) [ op( B ) \times conjg( op( A )' )] + \beta C \f]
 *          = dplasmaConjTrans: \f[ C = \alpha [ conjg( op( A )' ) \times op( B ) ] + conjg( \alpha ) [ conjg( op( B )' ) \times op( A ) ] + \beta C \f]
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B
 *
 * @param[in] beta
 *          beta specifies the scalar beta.
 *
 * @param[in,out] C
 *          Descriptor of the symmetric matrix C.
 *          On exit, the uplo part of the matrix described by C is overwritten
 *          by the result of the operation.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zsyr2k_New
 * @sa dplasma_zsyr2k_Destruct
 * @sa dplasma_csyr2k
 *
 ******************************************************************************/
int
dplasma_zsyr2k( parsec_context_t *parsec,
                dplasma_enum_t uplo,
                dplasma_enum_t trans,
                dplasma_complex64_t alpha,
                const parsec_tiled_matrix_t *A,
                const parsec_tiled_matrix_t *B,
                dplasma_complex64_t beta,
                parsec_tiled_matrix_t *C)
{
    parsec_taskpool_t *parsec_zsyr2k = NULL;

    /* Check input arguments */
    if ((uplo != dplasmaLower) && (uplo != dplasmaUpper)) {
        dplasma_error("dplasma_zsyr2k", "illegal value of uplo");
        return -1;
    }
    if (trans != dplasmaConjTrans && trans != dplasmaTrans && trans != dplasmaNoTrans ) {
        dplasma_error("dplasma_zsyr2k", "illegal value of trans");
        return -2;
    }

    if ( A->m != B->m || A->n != B->n ) {
        dplasma_error("dplasma_zsyr2k", "illegal descriptor A or B, they must have the same dimensions");
        return -4;
    }
    if ( C->m != C->n ) {
        dplasma_error("dplasma_zsyr2k", "illegal descriptor C (C->m != C->n)");
        return -6;
    }
    if ( (( trans == dplasmaNoTrans ) && ( A->m != C->m )) ||
         (( trans != dplasmaNoTrans ) && ( A->n != C->m )) ) {
        dplasma_error("dplasma_zsyr2k", "illegal sizes for the matrices");
        return -6;
    }

    parsec_zsyr2k = dplasma_zsyr2k_New(uplo, trans,
                                      alpha, A, B,
                                      beta, C);

    if ( parsec_zsyr2k != NULL )
    {
        parsec_context_add_taskpool( parsec, parsec_zsyr2k);
        dplasma_wait_until_completion(parsec);
        dplasma_zsyr2k_Destruct( parsec_zsyr2k );
    }
    return 0;
}
