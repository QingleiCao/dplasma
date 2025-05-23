/*
 * Copyright (c) 2010-2022 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasmaaux.h"

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 * dplasma_zgels -overdetermined or underdetermined linear systems
 *  involving an M-by-N matrix A using the QR or the LQ factorization of A.  It
 *  is assumed that A has full rank.  The following options are provided:
 *
 * 1. If trans = dplasmaNoTrans and M >= N: find the least squares solution of an
 *    overdetermined system, i.e., solve the least squares problem: minimize ||
 *    B - A*X ||.
 *
 * 2. If trans = dplasmaNoTrans and M < N: find the minimum norm solution of an
 *    underdetermined system A * X = B.
 *
 * 3. If trans = dplasmaConjTrans and m >= n:  find the minimum norm solution of
 *    an undetermined system A**H * X = B.
 *
 * 4. If trans = dplasmaConjTrans and m < n:  find the least squares solution of
 *    an overdetermined system, i.e., solve the least squares problem
 *                 minimize || B - A**H * X ||.
 *
 *  Several right hand side vectors B and solution vectors X can be handled in a
 *  single call; they are stored as the columns of the M-by-NRHS right hand side
 *  matrix B and the N-by-NRHS solution matrix X.
 *
 * The QR/LQ operation performed are the one provided by dplasma_zgeqrf() and
 * dplasma_zgelqf().
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] trans
 *          @arg dplasmaNoTrans:   the linear system involves A;
 *          @arg dplasmaConjTrans: the linear system involves A**H.
 *          Currently only dplasmaNoTrans is supported.
 *
  * @param[in] A
 *          Descriptor of the matrix A of size M-by-N factorized with the
 *          dplasma_zgelqf_New() routine.
 *          On entry, the M-by-N matrix A
 *          if M >= N, A is overwritten by details of its QR factorization as
 *                     returned by dplasma_zgeqrf();
 *          if M < N, A is overwritten by details of its LQ factorization as
 *                      returned by dplasma_zgelqf().
 *
 * @param[in] T
 *          Descriptor of the matrix T distributed exactly as the A matrix. T.mb
 *          defines the IB parameter of tile LQ algorithm. This matrix must be
 *          of size A.mt * T.mb - by - A.nt * T.nb, with T.nb == A.nb.
 *          This matrix is initialized by the factorization call and is returned
 *          for further solves.
 *
 * @param[in,out] B
 *          Descriptor that covers both matrix B and X.
 *          On entry, the matrix B of right hand side vectors, stored
 *          columnwise; B is M-by-NRHS if trans = dplasmaNoTrans, or N-by-NRHS if
 *          trans = dplasmaConjTrans.
 *          On exit, if INFO = 0, B is overwritten by the solution
 *          vectors, stored columnwise:
 *            - if trans = dplasmaNoTrans and m >= n, rows 1 to n of B contain
 *            the least squares solution vectors; the residual sum of squares
 *            for the solution in each column is given by the sum of squares of
 *            the modulus of elements N+1 to M in that column;
 *            - if trans = dplasmaNoTrans and m < n, rows 1 to N of B contain the
 *            minimum norm solution vectors;
 *            - if trans = dplasmaConjTrans and m >= n, rows 1 to M of B contain
 *            the minimum norm solution vectors;
 *            - if trans = dplasmaConjTrans and m < n, rows 1 to M of B contain
 *            the least squares solution vectors; the residual sum of squares
 *            for the solution in each column is given by the sum of squares of
 *            the modulus of elements M+1 to N in that column.
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_cgels
 * @sa dplasma_dgels
 * @sa dplasma_sgels
 *
 ******************************************************************************/
int
dplasma_zgels( parsec_context_t *parsec,
               dplasma_enum_t trans,
               parsec_tiled_matrix_t* A,
               parsec_tiled_matrix_t* T,
               parsec_tiled_matrix_t* B )
{
    parsec_tiled_matrix_t *subA;
    parsec_tiled_matrix_t *subB;
    int info = 0;

    /* Check input arguments */
    if ((trans != dplasmaNoTrans) && (trans != dplasmaConjTrans)) {
        dplasma_error("dplasma_zgels", "Invalid trans parameter");
        return -1;
    }
    if ( (T->nt != A->nt) || (T->mt != A->mt) ) {
        dplasma_error("dplasma_zgels", "illegal size of T (T should have as many tiles as A)");
        return -3;
    }
    if ( (B->m < A->n) && (B->m < A->m) ) {
        dplasma_error("dplasma_zgels", "illegal dimension of B, (B->m < max(A->m, A->n))");
        return -4;
    }

    if ( A->m >= A->n ) {
        /*
         * Compute QR factorization of A
         */
        dplasma_zgeqrf( parsec, A, T );

        if ( trans == dplasmaNoTrans ) {
            /*
             * Least-Squares Problem min || A * X - B ||
             */
            dplasma_zunmqr( parsec, dplasmaLeft, dplasmaConjTrans, A, T, B );

            subA = parsec_tiled_matrix_submatrix( A, 0, 0, A->n, A->n );
            subB = parsec_tiled_matrix_submatrix( B, 0, 0, A->n, B->n );
            info = dplasma_ztrsm(  parsec, dplasmaLeft, dplasmaUpper, dplasmaNoTrans, dplasmaNonUnit, 1.0, subA, subB );
        }
        else {
            /*
             * Overdetermined system of equations A**H * X = B
             */
            subA = parsec_tiled_matrix_submatrix( A, 0, 0, A->n, A->n );
            subB = parsec_tiled_matrix_submatrix( B, 0, 0, A->n, B->n );
            info = dplasma_ztrsm(  parsec, dplasmaLeft, dplasmaUpper, dplasmaConjTrans, dplasmaNonUnit, 1.0, subA, subB );

            if (info != 0) {
                free(subA);
                free(subB);
                return info;
            }

            if ( A->m > A->n ) {
                free(subB);
                subB = parsec_tiled_matrix_submatrix( B, A->n, 0, A->m-A->n, B->n );
                dplasma_zlaset( parsec, dplasmaUpperLower, 0., 0., subB );
            }

            dplasma_zunmqr( parsec, dplasmaLeft, dplasmaNoTrans, A, T, B );
        }
    }
    else {
        /*
         * Compute LQ factorization of A
         */
        dplasma_zgelqf( parsec, A, T );

        if ( trans == dplasmaNoTrans ) {
            /*
             * Underdetermined system of equations A * X = B
             */
            subA = parsec_tiled_matrix_submatrix( A, 0, 0, A->m, A->m );
            subB = parsec_tiled_matrix_submatrix( B, 0, 0, A->m, B->n );
            info = dplasma_ztrsm(  parsec, dplasmaLeft, dplasmaLower, dplasmaNoTrans, dplasmaNonUnit, 1.0, subA, subB );

            if (info != 0) {
                free(subA);
                free(subB);
                return info;
            }

            if ( A->n > A->m ) {
                free(subB);
                subB = parsec_tiled_matrix_submatrix( B, A->m, 0, A->n-A->m, B->n );
                dplasma_zlaset( parsec, dplasmaUpperLower, 0., 0., subB );
            }

            dplasma_zunmlq( parsec, dplasmaLeft, dplasmaConjTrans, A, T, B );
        }
        else {
            /*
             * Overdetermined system min || A**H * X - B ||
             */
            dplasma_zunmlq( parsec, dplasmaLeft, dplasmaNoTrans, A, T, B );

            subA = parsec_tiled_matrix_submatrix( A, 0, 0, A->m, A->m );
            subB = parsec_tiled_matrix_submatrix( B, 0, 0, A->m, B->n );
            info = dplasma_ztrsm(  parsec, dplasmaLeft, dplasmaLower, dplasmaConjTrans, dplasmaNonUnit, 1.0, subA, subB );
        }
    }

    free(subA);
    free(subB);

    return info;
}
