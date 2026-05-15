/*
 * Copyright (c) 2026 The University of Tennessee and The University
 *                    of Tennessee Research Foundation. All rights reserved.
 *
 * @precisions normal z -> s d c
 */

#include "common.h"
#include "dplasmaaux.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

static inline int sparse_keep_entry(int gi, int gj, int li, int lj, int threshold_per_thousand)
{
    unsigned int h = (unsigned int)((gi + 1) * 1315423911u) ^ (unsigned int)((gj + 1) * 2654435761u);
    int keep = ((int)(h % 1000u) < threshold_per_thousand);
    if( li == 0 && lj == 0 ) {
        keep = 1;
    }
    return keep;
}

static void apply_sparse_pattern_dense(parsec_matrix_block_cyclic_t *A, int threshold_per_thousand)
{
    for(int m = 0; m < A->super.mt; m++) {
        int rows = (m == A->super.mt - 1) ? (A->super.m - m * A->super.mb) : A->super.mb;
        int lda = BLKLDD(&A->super, m);
        for(int n = 0; n < A->super.nt; n++) {
            int cols = (n == A->super.nt - 1) ? (A->super.n - n * A->super.nb) : A->super.nb;
            if( A->super.super.myrank != A->super.super.rank_of((parsec_data_collection_t*)&A->super, m, n) ) {
                continue;
            }
            parsec_data_t *d = A->super.super.data_of((parsec_data_collection_t*)&A->super, m, n);
            dplasma_complex64_t *ptr = (dplasma_complex64_t*)PARSEC_DATA_COPY_GET_PTR(d->device_copies[0]);
            for(int j = 0; j < cols; j++) {
                for(int i = 0; i < rows; i++) {
                    int gi = m * A->super.mb + i;
                    int gj = n * A->super.nb + j;
                    if( !sparse_keep_entry(gi, gj, i, j, threshold_per_thousand) ) {
                        ptr[j * lda + i] = (dplasma_complex64_t)0.0;
                    }
                }
            }
        }
    }
}

static int check_solution(parsec_context_t *parsec, int loud,
                          dplasma_enum_t transA, dplasma_enum_t transB,
                          dplasma_complex64_t alpha, dplasma_complex64_t beta,
                          parsec_matrix_block_cyclic_t *dcAin,
                          parsec_matrix_block_cyclic_t *dcBin,
                          parsec_matrix_block_cyclic_t *dcCinit,
                          int threshold_per_thousand,
                          parsec_matrix_block_cyclic_t *dcCfinal)
{
    int info_solution = 1;
    int M = dcCfinal->super.m;
    int N = dcCfinal->super.n;
    int Am = dcAin->super.m;
    int An = dcAin->super.n;
    int Bm = dcBin->super.m;
    int Bn = dcBin->super.n;
    int K = (transA == dplasmaNoTrans) ? An : Am;
    int MB = dcCfinal->super.mb;
    int NB = dcCfinal->super.nb;
    int LDA = Am;
    int LDB = Bm;
    int LDC = M;
    int rank = dcCfinal->super.super.myrank;
    double eps = LAPACKE_dlamch_work('e');
    double Anorm, Bnorm, Cinitnorm, Cdplasmanorm, Clapacknorm, Rnorm, result;

    PASTE_CODE_ALLOCATE_MATRIX(dcAref, 1,
        parsec_matrix_block_cyclic, (&dcAref, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_LAPACK,
                               rank, MB, NB, LDA, An, 0, 0,
                               Am, An, 1, 1, 1, 1, 0, 0));
    PASTE_CODE_ALLOCATE_MATRIX(dcBref, 1,
        parsec_matrix_block_cyclic, (&dcBref, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_LAPACK,
                               rank, MB, NB, LDB, Bn, 0, 0,
                               Bm, Bn, 1, 1, 1, 1, 0, 0));
    PASTE_CODE_ALLOCATE_MATRIX(dcCref, 1,
        parsec_matrix_block_cyclic, (&dcCref, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_LAPACK,
                               rank, MB, NB, LDC, N, 0, 0,
                               M, N, 1, 1, 1, 1, 0, 0));
    PASTE_CODE_ALLOCATE_MATRIX(dcCcmp, 1,
        parsec_matrix_block_cyclic, (&dcCcmp, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_LAPACK,
                               rank, MB, NB, LDC, N, 0, 0,
                               M, N, 1, 1, 1, 1, 0, 0));

    dplasma_zlacpy(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)dcAin, (parsec_tiled_matrix_t *)&dcAref);
    dplasma_zlacpy(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)dcBin, (parsec_tiled_matrix_t *)&dcBref);
    dplasma_zlacpy(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)dcCinit, (parsec_tiled_matrix_t *)&dcCref);
    dplasma_zlacpy(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)dcCfinal, (parsec_tiled_matrix_t *)&dcCcmp);
    apply_sparse_pattern_dense(&dcAref, threshold_per_thousand);
    apply_sparse_pattern_dense(&dcBref, threshold_per_thousand);

    Anorm = dplasma_zlange(parsec, dplasmaInfNorm, (parsec_tiled_matrix_t *)&dcAref);
    Bnorm = dplasma_zlange(parsec, dplasmaInfNorm, (parsec_tiled_matrix_t *)&dcBref);
    Cinitnorm = dplasma_zlange(parsec, dplasmaInfNorm, (parsec_tiled_matrix_t *)&dcCref);
    Cdplasmanorm = dplasma_zlange(parsec, dplasmaInfNorm, (parsec_tiled_matrix_t *)&dcCcmp);

    if(rank == 0) {
        cblas_zgemm(CblasColMajor,
                    (CBLAS_TRANSPOSE)transA, (CBLAS_TRANSPOSE)transB,
                    M, N, K,
                    CBLAS_SADDR(alpha), dcAref.mat, LDA,
                                        dcBref.mat, LDB,
                    CBLAS_SADDR(beta),  dcCref.mat, LDC);
    }

    Clapacknorm = dplasma_zlange(parsec, dplasmaInfNorm, (parsec_tiled_matrix_t *)&dcCref);
    dplasma_zgeadd(parsec, dplasmaNoTrans, -1.0, (parsec_tiled_matrix_t *)&dcCcmp,
                                         1.0,  (parsec_tiled_matrix_t *)&dcCref);
    Rnorm = dplasma_zlange(parsec, dplasmaMaxNorm, (parsec_tiled_matrix_t *)&dcCref);

    if(rank == 0) {
        if(loud > 2) {
            printf("  ||A||_inf = %e, ||B||_inf = %e, ||C||_inf = %e\n"
                   "  ||lapack(a*A*B+b*C)||_inf = %e, ||dplasma(a*A*B+b*C)||_inf = %e, ||R||_m = %e\n",
                   Anorm, Bnorm, Cinitnorm, Clapacknorm, Cdplasmanorm, Rnorm);
        }

        result = Rnorm / ((Anorm + Bnorm + Cinitnorm) * max(M, N) * eps);
        info_solution = (isinf(Clapacknorm) || isinf(Cdplasmanorm) ||
                         isnan(result) || isinf(result) || (result > 10.0));
    }

#if defined(PARSEC_HAVE_MPI)
    MPI_Bcast(&info_solution, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    parsec_data_free(dcAref.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcAref);
    parsec_data_free(dcBref.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcBref);
    parsec_data_free(dcCref.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcCref);
    parsec_data_free(dcCcmp.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcCcmp);
    return info_solution;
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec;
    int iparam[IPARAM_SIZEOF];
    int info_solution = 0;
    int Aseed = 3872, Bseed = 4674, Cseed = 2873;
    dplasma_complex64_t alpha = 0.51, beta = -0.42;
    int threshold_per_thousand = 250;
    dplasma_info_t opt;

#if defined(PRECISION_z) || defined(PRECISION_c)
    alpha -= I * 0.32;
    beta  += I * 0.21;
#endif

    iparam_default_gemm(iparam);
    iparam_default_ibnbmb(iparam, 0, 200, 200);

    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    PASTE_CODE_FLOPS(FLOPS_ZGEMM, ((DagDouble_t)M, (DagDouble_t)N, (DagDouble_t)K));

    LDA = max(LDA, max(M, K));
    LDB = max(LDB, max(K, N));
    LDC = max(LDC, M);

    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
        parsec_matrix_block_cyclic, (&dcA, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_TILE,
                               rank, MB, NB, LDA, K, 0, 0,
                               M, K, P, nodes/P, KP, KQ, IP, JQ));
    PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
        parsec_matrix_block_cyclic, (&dcB, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_TILE,
                               rank, MB, NB, LDB, N, 0, 0,
                               K, N, P, nodes/P, KP, KQ, IP, JQ));
    PASTE_CODE_ALLOCATE_MATRIX(dcC, 1,
        parsec_matrix_block_cyclic, (&dcC, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_TILE,
                               rank, MB, NB, LDC, N, 0, 0,
                               M, N, P, nodes/P, KP, KQ, IP, JQ));
    PASTE_CODE_ALLOCATE_MATRIX(dcC0, 1,
        parsec_matrix_block_cyclic, (&dcC0, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_TILE,
                               rank, MB, NB, LDC, N, 0, 0,
                               M, N, P, nodes/P, KP, KQ, IP, JQ));

    dplasma_zplrnt(parsec, 0, (parsec_tiled_matrix_t *)&dcA, Aseed);
    dplasma_zplrnt(parsec, 0, (parsec_tiled_matrix_t *)&dcB, Bseed);
    dplasma_zplrnt(parsec, 0, (parsec_tiled_matrix_t *)&dcC, Cseed);
    dplasma_zlacpy(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcC, (parsec_tiled_matrix_t *)&dcC0);
    apply_sparse_pattern_dense(&dcA, threshold_per_thousand);
    apply_sparse_pattern_dense(&dcB, threshold_per_thousand);

    dplasma_info_create(&opt);
    dplasma_info_set(opt, "DPLASMA:GEMM:SPARSE_IN_TILE", "1");

    SYNC_TIME_START();
    parsec_taskpool_t *parsec_zgemm = dplasma_zgemm_New_ex(dplasmaNoTrans, dplasmaNoTrans,
                                                            alpha, (parsec_tiled_matrix_t *)&dcA,
                                                            (parsec_tiled_matrix_t *)&dcB,
                                                            beta,  (parsec_tiled_matrix_t *)&dcC,
                                                            opt);
    PARSEC_CHECK_ERROR(NULL == parsec_zgemm ? PARSEC_ERROR : PARSEC_SUCCESS, "dplasma_zgemm_New_ex");
    PARSEC_CHECK_ERROR(parsec_context_add_taskpool(parsec, parsec_zgemm), "parsec_context_add_taskpool");
    PASTE_CODE_PROGRESS_KERNEL(parsec, zgemm_sparse);
    dplasma_zgemm_Destruct(parsec_zgemm);
    dplasma_info_free(&opt);

    info_solution = check_solution(parsec, (rank == 0) ? loud : 0,
                                   dplasmaNoTrans, dplasmaNoTrans,
                                   alpha, beta,
                                   &dcA, &dcB, &dcC0, threshold_per_thousand, &dcC);
    if(rank == 0) {
        printf(" ---- TESTING ZGEMM sparse-in-tile (%s)!\n",
               (info_solution == 0) ? "PASSED" : "FAILED");
    }

    parsec_data_free(dcA.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcA);
    parsec_data_free(dcB.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcB);
    parsec_data_free(dcC.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcC);
    parsec_data_free(dcC0.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcC0);
    cleanup_parsec(parsec, iparam);
    return info_solution;
}
