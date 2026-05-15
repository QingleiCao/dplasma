/*
 * Copyright (c) 2010-2025 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasma/types.h"
#include "dplasma/types_lapack.h"
#include "dplasmaaux.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"
#include "parsec/mca/device/device_gpu.h"
#include "utils/dplasma_info.h"
#include "utils/dplasma_lapack_adtt.h"
#include <stdint.h>
#include <string.h>

#include "zgemm_NN.h"
#include "zgemm_NN_sparse.h"
#include "zgemm_NT.h"
#include "zgemm_TN.h"
#include "zgemm_TT.h"

#include "zgemm_NN_summa.h"
#include "zgemm_NT_summa.h"
#include "zgemm_TN_summa.h"
#include "zgemm_TT_summa.h"

#define MAX_SHAPES 3

#include "zgemm_NN_gpu.h"

#include "parsec/utils/mca_param.h"

static parsec_taskpool_t *
dplasma_zgemm_summa_new(dplasma_enum_t transA, dplasma_enum_t transB,
                        dplasma_complex64_t alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                        dplasma_complex64_t beta,  parsec_tiled_matrix_t* C,
                        dplasma_info_t opt)
{
    int P, Q, IP, JQ, m, n;
    parsec_taskpool_t *zgemm_tp;
    parsec_matrix_block_cyclic_t *Cdist;

    P = ((parsec_matrix_block_cyclic_t*)C)->grid.rows;
    Q = ((parsec_matrix_block_cyclic_t*)C)->grid.cols;
    IP = ((parsec_matrix_block_cyclic_t*)C)->grid.ip;
    JQ = ((parsec_matrix_block_cyclic_t*)C)->grid.jq;

    dplasma_data_collection_t * ddc_A = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)A);
    dplasma_data_collection_t * ddc_B = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)B);
    dplasma_data_collection_t * ddc_C = dplasma_wrap_data_collection(C);

    m = dplasma_imax(C->mt, P);
    n = dplasma_imax(C->nt, Q);

    /* Create a copy of the C matrix to be used as a data distribution metric.
     * As it is used as a NULL value we must have a data_copy and a data associated
     * with it, so we can create them here.
     * Create the task distribution */
    Cdist = (parsec_matrix_block_cyclic_t*)malloc(sizeof(parsec_matrix_block_cyclic_t));

    parsec_matrix_block_cyclic_init(
            Cdist, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE,
            C->super.myrank,
            1, 1, /* Dimensions of the tiles              */
            m, n, /* Dimensions of the matrix             */
            0, 0, /* Starting points (not important here) */
            m, n, /* Dimensions of the submatrix          */
            P, Q, 1, 1, IP, JQ);
    Cdist->super.super.data_of = NULL;
    Cdist->super.super.data_of_key = NULL;

    if( dplasmaNoTrans == transA ) {
        if( dplasmaNoTrans == transB ) {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_NN_summa");
            parsec_zgemm_NN_summa_taskpool_t* tp;
            tp = parsec_zgemm_NN_summa_new(transA, transB, alpha, beta,
                                           ddc_A, ddc_B, ddc_C, (parsec_data_collection_t*)Cdist);
#if defined(DPLASMA_HAVE_HIP)
            /* It doesn't cost anything to define these infos if we have HIP but
             * don't have GPUs on the current machine, so we do it non-conditionally */
            tp->_g_hip_handles_infokey = parsec_info_lookup(&parsec_per_stream_infos, "DPLASMA::HIP::HANDLES", NULL);
#else
            tp->_g_hip_handles_infokey = PARSEC_INFO_ID_UNDEFINED;
#endif
            zgemm_tp = (parsec_taskpool_t*)tp;
        } else {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_NT_summa");
            parsec_zgemm_NT_summa_taskpool_t* tp;
            tp = parsec_zgemm_NT_summa_new(transA, transB, alpha, beta,
                                           ddc_A, ddc_B, ddc_C, (parsec_data_collection_t*)Cdist);
#if defined(DPLASMA_HAVE_HIP)
            /* It doesn't cost anything to define these infos if we have HIP but
             * don't have GPUs on the current machine, so we do it non-conditionally */
            tp->_g_hip_handles_infokey = parsec_info_lookup(&parsec_per_stream_infos, "DPLASMA::HIP::HANDLES", NULL);
#else
            tp->_g_hip_handles_infokey = PARSEC_INFO_ID_UNDEFINED;
#endif
            zgemm_tp = (parsec_taskpool_t*)tp;
        }
    } else {
        if( dplasmaNoTrans == transB ) {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_TN_summa");
            parsec_zgemm_TN_summa_taskpool_t* tp;
            tp = parsec_zgemm_TN_summa_new(transA, transB, alpha, beta,
                                           ddc_A, ddc_B, ddc_C, (parsec_data_collection_t*)Cdist);
#if defined(DPLASMA_HAVE_HIP)
            /* It doesn't cost anything to define these infos if we have HIP but
             * don't have GPUs on the current machine, so we do it non-conditionally */
            tp->_g_hip_handles_infokey = parsec_info_lookup(&parsec_per_stream_infos, "DPLASMA::HIP::HANDLES", NULL);
#else
            tp->_g_hip_handles_infokey = PARSEC_INFO_ID_UNDEFINED;
#endif
            zgemm_tp = (parsec_taskpool_t*)tp;
        } else {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_TT_summa");
            parsec_zgemm_TT_summa_taskpool_t* tp;
            tp = parsec_zgemm_TT_summa_new(transA, transB, alpha, beta,
                                           ddc_A, ddc_B, ddc_C,
                                           (parsec_data_collection_t*)Cdist);
#if defined(DPLASMA_HAVE_HIP)
            /* It doesn't cost anything to define these infos if we have HIP but
             * don't have GPUs on the current machine, so we do it non-conditionally */
            tp->_g_hip_handles_infokey = parsec_info_lookup(&parsec_per_stream_infos, "DPLASMA::HIP::HANDLES", NULL);
#else
            tp->_g_hip_handles_infokey = PARSEC_INFO_ID_UNDEFINED;
#endif
            zgemm_tp = (parsec_taskpool_t*)tp;
        }
    }

    int shape = 0;
    dplasma_setup_adtt_all_loc( ddc_A,
                                parsec_datatype_double_complex_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);
    dplasma_setup_adtt_all_loc( ddc_B,
                                parsec_datatype_double_complex_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);
    dplasma_setup_adtt_all_loc( ddc_C,
                                parsec_datatype_double_complex_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);
    assert(shape == MAX_SHAPES);

    (void)opt; //No user-defined options for this algorithm
    return zgemm_tp;
}

typedef struct {
    int nnz;
    int nnz_cap;
    int nrows;
    int ncols;
} dplasma_csr_tile_hdr_t;

static inline int
dplasma_zgemm_sparse_keep_entry(int gi, int gj, int li, int lj, int threshold_per_thousand)
{
    unsigned int h = (unsigned int)((gi + 1) * 1315423911u) ^ (unsigned int)((gj + 1) * 2654435761u);
    int keep = ((int)(h % 1000u) < threshold_per_thousand);
    if( li == 0 && lj == 0 ) {
        keep = 1; /* keep at least one entry candidate per tile */
    }
    return keep;
}

static int
dplasma_zgemm_build_sparse_tile_matrix(const parsec_tiled_matrix_t *Ain,
                                       int threshold_per_thousand,
                                       parsec_matrix_block_cyclic_t **Aout)
{
    parsec_matrix_block_cyclic_t *A = (parsec_matrix_block_cyclic_t*)Ain;
    parsec_matrix_block_cyclic_t *As = (parsec_matrix_block_cyclic_t*)malloc(sizeof(parsec_matrix_block_cyclic_t));
    /* Each sparse tile stores CSR as three arrays in one contiguous payload:
     * rowptr[mb+1], colind[mb*nb], vals[mb*nb].
     * We keep max capacities here and track effective nnz in the header.
     */
    size_t vals_capacity = (size_t)A->super.mb * (size_t)A->super.nb;
    size_t tile_bytes = sizeof(int) * (4 + A->super.mb + 1 + vals_capacity)
                      + vals_capacity * sizeof(dplasma_complex64_t);

    parsec_matrix_block_cyclic_init(As, PARSEC_MATRIX_BYTE, PARSEC_MATRIX_TILE,
                                    A->super.super.myrank,
                                    A->super.mb, A->super.nb,
                                    A->super.lm, A->super.ln,
                                    A->super.i, A->super.j,
                                    A->super.m, A->super.n,
                                    A->grid.rows, A->grid.cols, A->grid.krows, A->grid.kcols, A->grid.ip, A->grid.jq);
    As->super.bsiz = (int)tile_bytes;
    As->mat = parsec_data_allocate((size_t)As->super.nb_local_tiles * (size_t)As->super.bsiz);
    parsec_data_collection_set_key((parsec_data_collection_t*)As, "zgemm_sparse_tile");

    for(int m = 0; m < A->super.mt; m++) {
        int rows = (m == A->super.mt - 1) ? (A->super.m - m * A->super.mb) : A->super.mb;
        int ldas = BLKLDD(&A->super, m);
        for(int n = 0; n < A->super.nt; n++) {
            if( A->super.super.myrank != A->super.super.rank_of((parsec_data_collection_t*)&A->super, m, n) ) {
                continue;
            }
            int cols = (n == A->super.nt - 1) ? (A->super.n - n * A->super.nb) : A->super.nb;
            parsec_data_t *src_data = A->super.super.data_of((parsec_data_collection_t*)&A->super, m, n);
            parsec_data_t *dst_data = As->super.super.data_of((parsec_data_collection_t*)&As->super, m, n);
            dplasma_complex64_t *src = (dplasma_complex64_t*)PARSEC_DATA_COPY_GET_PTR(src_data->device_copies[0]);
            uint8_t *dst = (uint8_t*)PARSEC_DATA_COPY_GET_PTR(dst_data->device_copies[0]);
            dplasma_csr_tile_hdr_t *hdr = (dplasma_csr_tile_hdr_t*)dst;
            int *rowptr = (int*)(hdr + 1);
            int *colind = rowptr + (A->super.mb + 1);
            dplasma_complex64_t *vals = (dplasma_complex64_t*)(colind + vals_capacity);
            int nnz = 0;

            hdr->nnz = 0;
            hdr->nnz_cap = (int)vals_capacity;
            hdr->nrows = A->super.mb;
            hdr->ncols = A->super.nb;
            rowptr[0] = 0;
            for(int i = 0; i < A->super.mb; i++) {
                if( i < rows ) {
                    for(int j = 0; j < cols; j++) {
                        int gi = m * A->super.mb + i;
                        int gj = n * A->super.nb + j;
                        dplasma_complex64_t v = src[j * ldas + i];
                        if( dplasma_zgemm_sparse_keep_entry(gi, gj, i, j, threshold_per_thousand) &&
                            v != (dplasma_complex64_t)0.0 ) {
                            colind[nnz] = j;
                            vals[nnz] = v;
                            nnz++;
                        }
                    }
                }
                rowptr[i + 1] = nnz;
            }
            hdr->nnz = nnz;
        }
    }

    *Aout = As;
    return PARSEC_SUCCESS;
}

static void
dplasma_zgemm_destroy_sparse_tile_matrix(parsec_matrix_block_cyclic_t *A)
{
    if( NULL == A ) {
        return;
    }
    if( NULL != A->mat ) {
        parsec_data_free(A->mat);
        A->mat = NULL;
    }
    parsec_tiled_matrix_destroy(&A->super);
    free(A);
}

static inline void
dplasma_zgemm_csr_tile_arrays(uint8_t *tile,
                              dplasma_csr_tile_hdr_t **hdr,
                              int **rowptr, int **colind, dplasma_complex64_t **vals)
{
    *hdr = (dplasma_csr_tile_hdr_t*)tile;
    *rowptr = (int*)(*hdr + 1);
    *colind = *rowptr + ((*hdr)->nrows + 1);
    *vals = (dplasma_complex64_t*)(*colind + (*hdr)->nnz_cap);
}

static int
dplasma_zgemm_symbolic_nnz_tile(parsec_matrix_block_cyclic_t *sA,
                                parsec_matrix_block_cyclic_t *sB,
                                parsec_matrix_block_cyclic_t *C0,
                                int m, int n, dplasma_complex64_t beta)
{
    int rows = (m == C0->super.mt - 1) ? (C0->super.m - m * C0->super.mb) : C0->super.mb;
    int cols = (n == C0->super.nt - 1) ? (C0->super.n - n * C0->super.nb) : C0->super.nb;
    int ldc0 = BLKLDD(&C0->super, m);
    parsec_data_t *c0_data = C0->super.super.data_of((parsec_data_collection_t*)&C0->super, m, n);
    dplasma_complex64_t *c0 = (dplasma_complex64_t*)PARSEC_DATA_COPY_GET_PTR(c0_data->device_copies[0]);
    uint8_t *marker = (uint8_t*)calloc((size_t)cols, sizeof(uint8_t));
    int nnz = 0;

    if( NULL == marker ) {
        return rows * cols;
    }

    for(int i = 0; i < rows; i++) {
        memset(marker, 0, (size_t)cols);
        if( beta != (dplasma_complex64_t)0.0 ) {
            for(int j = 0; j < cols; j++) {
                if( c0[j * ldc0 + i] != (dplasma_complex64_t)0.0 ) {
                    marker[j] = 1;
                }
            }
        }
        for(int k = 0; k < sA->super.nt; k++) {
            parsec_data_t *a_data = sA->super.super.data_of((parsec_data_collection_t*)&sA->super, m, k);
            parsec_data_t *b_data = sB->super.super.data_of((parsec_data_collection_t*)&sB->super, k, n);
            uint8_t *atile = (uint8_t*)PARSEC_DATA_COPY_GET_PTR(a_data->device_copies[0]);
            uint8_t *btile = (uint8_t*)PARSEC_DATA_COPY_GET_PTR(b_data->device_copies[0]);
            dplasma_csr_tile_hdr_t *ah, *bh;
            int *a_rowptr, *a_colind, *b_rowptr, *b_colind;
            dplasma_complex64_t *a_vals, *b_vals;
            dplasma_zgemm_csr_tile_arrays(atile, &ah, &a_rowptr, &a_colind, &a_vals);
            dplasma_zgemm_csr_tile_arrays(btile, &bh, &b_rowptr, &b_colind, &b_vals);
            for(int ia = a_rowptr[i]; ia < a_rowptr[i + 1]; ia++) {
                int p = a_colind[ia];
                for(int ib = b_rowptr[p]; ib < b_rowptr[p + 1]; ib++) {
                    int j = b_colind[ib];
                    if( j < cols ) {
                        marker[j] = 1;
                    }
                }
            }
            (void)ah; (void)bh; (void)a_vals; (void)b_vals;
        }
        for(int j = 0; j < cols; j++) {
            nnz += (marker[j] != 0);
        }
    }

    free(marker);
    return nnz;
}

static int
dplasma_zgemm_build_sparse_c_matrix(parsec_matrix_block_cyclic_t *sA,
                                    parsec_matrix_block_cyclic_t *sB,
                                    const parsec_tiled_matrix_t *C0in,
                                    dplasma_complex64_t beta,
                                    parsec_matrix_block_cyclic_t **Cout)
{
    parsec_matrix_block_cyclic_t *C0 = (parsec_matrix_block_cyclic_t*)C0in;
    parsec_matrix_block_cyclic_t *Cs = (parsec_matrix_block_cyclic_t*)malloc(sizeof(parsec_matrix_block_cyclic_t));
    int mb = C0->super.mb;
    int nb = C0->super.nb;
    int max_nnz = 1;
    int fallback_dense = (C0->super.super.nodes > 1);

    if( fallback_dense ) {
        max_nnz = mb * nb;
    } else {
        for(int m = 0; m < C0->super.mt; m++) {
            for(int n = 0; n < C0->super.nt; n++) {
                if( C0->super.super.myrank != C0->super.super.rank_of((parsec_data_collection_t*)&C0->super, m, n) ) {
                    continue;
                }
                int nnz = dplasma_zgemm_symbolic_nnz_tile(sA, sB, C0, m, n, beta);
                max_nnz = dplasma_imax(max_nnz, nnz);
            }
        }
    }

    {
        size_t tile_bytes = sizeof(int) * (4 + mb + 1 + (size_t)max_nnz)
                          + (size_t)max_nnz * sizeof(dplasma_complex64_t);
        parsec_matrix_block_cyclic_init(Cs, PARSEC_MATRIX_BYTE, PARSEC_MATRIX_TILE,
                                        C0->super.super.myrank,
                                        mb, nb,
                                        C0->super.lm, C0->super.ln,
                                        C0->super.i, C0->super.j,
                                        C0->super.m, C0->super.n,
                                        C0->grid.rows, C0->grid.cols, C0->grid.krows, C0->grid.kcols, C0->grid.ip, C0->grid.jq);
        Cs->super.bsiz = (int)tile_bytes;
        Cs->mat = parsec_data_allocate((size_t)Cs->super.nb_local_tiles * (size_t)Cs->super.bsiz);
        parsec_data_collection_set_key((parsec_data_collection_t*)Cs, "zgemm_sparse_c_tile");
    }

    for(int m = 0; m < C0->super.mt; m++) {
        int rows = (m == C0->super.mt - 1) ? (C0->super.m - m * C0->super.mb) : C0->super.mb;
        int ldc0 = BLKLDD(&C0->super, m);
        for(int n = 0; n < C0->super.nt; n++) {
            if( C0->super.super.myrank != C0->super.super.rank_of((parsec_data_collection_t*)&C0->super, m, n) ) {
                continue;
            }
            int cols = (n == C0->super.nt - 1) ? (C0->super.n - n * C0->super.nb) : C0->super.nb;
            parsec_data_t *c0_data = C0->super.super.data_of((parsec_data_collection_t*)&C0->super, m, n);
            parsec_data_t *cs_data = Cs->super.super.data_of((parsec_data_collection_t*)&Cs->super, m, n);
            dplasma_complex64_t *c0 = (dplasma_complex64_t*)PARSEC_DATA_COPY_GET_PTR(c0_data->device_copies[0]);
            uint8_t *cst = (uint8_t*)PARSEC_DATA_COPY_GET_PTR(cs_data->device_copies[0]);
            dplasma_csr_tile_hdr_t *ch;
            int *c_rowptr, *c_colind;
            dplasma_complex64_t *c_vals;
            ch = (dplasma_csr_tile_hdr_t*)cst;
            ch->nnz = 0;
            ch->nnz_cap = max_nnz;
            ch->nrows = mb;
            ch->ncols = nb;
            dplasma_zgemm_csr_tile_arrays(cst, &ch, &c_rowptr, &c_colind, &c_vals);
            int nnz = 0;

            c_rowptr[0] = 0;

            if( fallback_dense ) {
                for(int i = 0; i < mb; i++) {
                    if( i < rows ) {
                        for(int j = 0; j < cols; j++) {
                            c_colind[nnz] = j;
                            c_vals[nnz] = beta * c0[j * ldc0 + i];
                            nnz++;
                        }
                    }
                    c_rowptr[i + 1] = nnz;
                }
            } else {
                uint8_t *marker = (uint8_t*)calloc((size_t)cols, sizeof(uint8_t));
                if( NULL == marker ) {
                    dplasma_zgemm_destroy_sparse_tile_matrix(Cs);
                    return PARSEC_ERROR;
                }
                for(int i = 0; i < rows; i++) {
                    memset(marker, 0, (size_t)cols);
                    if( beta != (dplasma_complex64_t)0.0 ) {
                        for(int j = 0; j < cols; j++) {
                            if( c0[j * ldc0 + i] != (dplasma_complex64_t)0.0 ) {
                                marker[j] = 1;
                            }
                        }
                    }
                    for(int k = 0; k < sA->super.nt; k++) {
                        parsec_data_t *a_data = sA->super.super.data_of((parsec_data_collection_t*)&sA->super, m, k);
                        parsec_data_t *b_data = sB->super.super.data_of((parsec_data_collection_t*)&sB->super, k, n);
                        uint8_t *atile = (uint8_t*)PARSEC_DATA_COPY_GET_PTR(a_data->device_copies[0]);
                        uint8_t *btile = (uint8_t*)PARSEC_DATA_COPY_GET_PTR(b_data->device_copies[0]);
                        dplasma_csr_tile_hdr_t *ah, *bh;
                        int *a_rowptr, *a_colind, *b_rowptr, *b_colind;
                        dplasma_complex64_t *a_vals, *b_vals;
                        dplasma_zgemm_csr_tile_arrays(atile, &ah, &a_rowptr, &a_colind, &a_vals);
                        dplasma_zgemm_csr_tile_arrays(btile, &bh, &b_rowptr, &b_colind, &b_vals);
                        for(int ia = a_rowptr[i]; ia < a_rowptr[i + 1]; ia++) {
                            int p = a_colind[ia];
                            for(int ib = b_rowptr[p]; ib < b_rowptr[p + 1]; ib++) {
                                int j = b_colind[ib];
                                if( j < cols ) {
                                    marker[j] = 1;
                                }
                            }
                        }
                        (void)ah; (void)bh; (void)a_vals; (void)b_vals;
                    }
                    for(int j = 0; j < cols; j++) {
                        if( marker[j] ) {
                            c_colind[nnz] = j;
                            c_vals[nnz] = beta * c0[j * ldc0 + i];
                            nnz++;
                        }
                    }
                    c_rowptr[i + 1] = nnz;
                }
                for(int i = rows; i < mb; i++) {
                    c_rowptr[i + 1] = nnz;
                }
                free(marker);
            }
            ch->nnz = nnz;
        }
    }

    *Cout = Cs;
    return PARSEC_SUCCESS;
}

static void
dplasma_zgemm_sparse_c_to_dense(parsec_matrix_block_cyclic_t *Cs,
                                parsec_tiled_matrix_t *Cdense)
{
    parsec_matrix_block_cyclic_t *C = (parsec_matrix_block_cyclic_t*)Cdense;
    for(int m = 0; m < C->super.mt; m++) {
        int rows = (m == C->super.mt - 1) ? (C->super.m - m * C->super.mb) : C->super.mb;
        int ldc = BLKLDD(&C->super, m);
        for(int n = 0; n < C->super.nt; n++) {
            if( C->super.super.myrank != C->super.super.rank_of((parsec_data_collection_t*)&C->super, m, n) ) {
                continue;
            }
            int cols = (n == C->super.nt - 1) ? (C->super.n - n * C->super.nb) : C->super.nb;
            parsec_data_t *d_data = C->super.super.data_of((parsec_data_collection_t*)&C->super, m, n);
            parsec_data_t *s_data = Cs->super.super.data_of((parsec_data_collection_t*)&Cs->super, m, n);
            dplasma_complex64_t *dptr = (dplasma_complex64_t*)PARSEC_DATA_COPY_GET_PTR(d_data->device_copies[0]);
            uint8_t *sptr = (uint8_t*)PARSEC_DATA_COPY_GET_PTR(s_data->device_copies[0]);
            dplasma_csr_tile_hdr_t *ch;
            int *c_rowptr, *c_colind;
            dplasma_complex64_t *c_vals;
            dplasma_zgemm_csr_tile_arrays(sptr, &ch, &c_rowptr, &c_colind, &c_vals);
            for(int j = 0; j < cols; j++) {
                for(int i = 0; i < rows; i++) {
                    dptr[j * ldc + i] = (dplasma_complex64_t)0.0;
                }
            }
            for(int i = 0; i < rows; i++) {
                for(int ic = c_rowptr[i]; ic < c_rowptr[i + 1]; ic++) {
                    int j = c_colind[ic];
                    if( j < cols ) {
                        dptr[j * ldc + i] = c_vals[ic];
                    }
                }
            }
            (void)ch;
        }
    }
}

static parsec_taskpool_t *
dplasma_zgemm_sparse_new(dplasma_enum_t transA, dplasma_enum_t transB,
                         dplasma_complex64_t alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                         dplasma_complex64_t beta,  parsec_tiled_matrix_t* C,
                         dplasma_info_t opt)
{
    parsec_taskpool_t* zgemm_tp = NULL;
    parsec_matrix_block_cyclic_t *sA = NULL, *sB = NULL, *sC = NULL;
    int info_found = 0;
    char info_value[DPLASMA_MAX_INFO_VAL];
    int threshold_per_thousand = 1000;

    if( dplasmaNoTrans != transA || dplasmaNoTrans != transB ) {
        dplasma_error("dplasma_zgemm_sparse_new", "sparse-in-tile PTG implementation currently supports NoTrans/NoTrans only");
        return NULL;
    }

    dplasma_info_get(opt, "DPLASMA:GEMM:SPARSE_PATTERN_PER_THOUSAND", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        threshold_per_thousand = atoi(info_value);
        if( threshold_per_thousand < 0 ) threshold_per_thousand = 0;
        if( threshold_per_thousand > 1000 ) threshold_per_thousand = 1000;
    }

    if( PARSEC_SUCCESS != dplasma_zgemm_build_sparse_tile_matrix(A, threshold_per_thousand, &sA) ||
        PARSEC_SUCCESS != dplasma_zgemm_build_sparse_tile_matrix(B, threshold_per_thousand, &sB) ||
        PARSEC_SUCCESS != dplasma_zgemm_build_sparse_c_matrix(sA, sB, C, beta, &sC) ) {
        dplasma_zgemm_destroy_sparse_tile_matrix(sA);
        dplasma_zgemm_destroy_sparse_tile_matrix(sB);
        dplasma_zgemm_destroy_sparse_tile_matrix(sC);
        dplasma_error("dplasma_zgemm_sparse_new", "failed to build CSR tile descriptors");
        return NULL;
    }

    PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_NN_sparse");
    parsec_zgemm_NN_sparse_taskpool_t* tp;
    tp = parsec_zgemm_NN_sparse_new(transA, transB, alpha, beta,
                                    (parsec_data_collection_t*)sA,
                                    (parsec_data_collection_t*)sB,
                                    (parsec_data_collection_t*)sC,
                                    C);
    if( NULL == tp ) {
        dplasma_zgemm_destroy_sparse_tile_matrix(sA);
        dplasma_zgemm_destroy_sparse_tile_matrix(sB);
        dplasma_zgemm_destroy_sparse_tile_matrix(sC);
        return NULL;
    }
    zgemm_tp = (parsec_taskpool_t*)tp;

    (void)opt;
    return zgemm_tp;
}

static parsec_taskpool_t *
dplasma_zgemm_default_new(dplasma_enum_t transA, dplasma_enum_t transB,
                          dplasma_complex64_t alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                          dplasma_complex64_t beta,  parsec_tiled_matrix_t* C,
                          dplasma_info_t opt)
{
    parsec_taskpool_t* zgemm_tp;

    dplasma_data_collection_t * ddc_A = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)A);
    dplasma_data_collection_t * ddc_B = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)B);
    dplasma_data_collection_t * ddc_C = dplasma_wrap_data_collection(C);

    if( dplasmaNoTrans == transA ) {
        if( dplasmaNoTrans == transB ) {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_NN");
            parsec_zgemm_NN_taskpool_t* tp;
            tp = parsec_zgemm_NN_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C);
            zgemm_tp = (parsec_taskpool_t*)tp;
        } else {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_NT");
            parsec_zgemm_NT_taskpool_t* tp;
            tp = parsec_zgemm_NT_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C);
            zgemm_tp = (parsec_taskpool_t*)tp;
        }
    } else {
        if( dplasmaNoTrans == transB ) {
            parsec_zgemm_TN_taskpool_t* tp;
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_TN");
            tp = parsec_zgemm_TN_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C);
            zgemm_tp = (parsec_taskpool_t*)tp;
        }
        else {
            PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_TT");
            parsec_zgemm_TT_taskpool_t* tp;
            tp = parsec_zgemm_TT_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C);
            zgemm_tp = (parsec_taskpool_t*)tp;
        }
    }

    int shape = 0;
    dplasma_setup_adtt_all_loc( ddc_A,
                                parsec_datatype_double_complex_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);
    dplasma_setup_adtt_all_loc( ddc_B,
                                parsec_datatype_double_complex_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);
    dplasma_setup_adtt_all_loc( ddc_C,
                                parsec_datatype_double_complex_t,
                                PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                &shape);
    assert(shape == MAX_SHAPES);

    (void)opt; //No user-defined options for this algorithm
    return zgemm_tp;
}

#if defined(DPLASMA_HAVE_CUDA) || defined(DPLASMA_HAVE_HIP)
static parsec_taskpool_t*
dplasma_zgemm_gpu_new( dplasma_enum_t transA, dplasma_enum_t transB,
                       dplasma_complex64_t alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                       dplasma_complex64_t beta,  parsec_tiled_matrix_t* C,
                       dplasma_info_t opt)
{

    parsec_taskpool_t* zgemm_tp = NULL;
    int64_t gpu_mem_nb_blocks = -1;
    size_t gpu_mem_block_size = 0;
    size_t tile_size;
    int64_t nb_block_per_tile, nb_tile_per_gpu;
    int mt, nt, kt;
    int info_found;
    char info_value[DPLASMA_MAX_INFO_VAL];
    double vd;

    int *dev_index, nbgpu, dev;
    int u, v;
    int M, Mbound, Mlim;
    int N, Nbound, Nlim;
    int K;

    int b, c, d, p, q, look_ahead;

    if( dplasmaNoTrans != transA || dplasmaNoTrans != transB ) {
        dplasma_error("dplasma_zgemm_gpu_new", "NoTrans for A or B not implemented yet in JDF for GPUs");
        return NULL;
    }

    nbgpu = 0;
    for(dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type || PARSEC_DEV_HIP == device->type ) {
            parsec_device_gpu_module_t *gpu_device = (parsec_device_gpu_module_t*)device;
            nbgpu++;
            if( 0 == gpu_mem_block_size )
                gpu_mem_block_size = gpu_device->mem_block_size;
            if( -1 == gpu_mem_nb_blocks || gpu_device->mem_nb_blocks < gpu_mem_nb_blocks )
                gpu_mem_nb_blocks = gpu_device->mem_nb_blocks;
        }
    }
    if(nbgpu == 0) {
        dplasma_error("dplasma_Zgemm_gpu_New", "Trying to instantiate JDF for GPUs on machine without GPUs");
        return NULL;
    }
    dev_index = (int*)malloc(nbgpu * sizeof(int));
    nbgpu= 0;
    for(dev = 0; dev < (int)parsec_nb_devices; dev++) {
        parsec_device_module_t *device = parsec_mca_device_get(dev);
        if( PARSEC_DEV_CUDA == device->type || PARSEC_DEV_HIP == device->type ) {
            dev_index[nbgpu++] = device->device_index;
        }
    }

    p = ((parsec_matrix_block_cyclic_t*)C)->grid.rows;
    q = ((parsec_matrix_block_cyclic_t*)C)->grid.cols;

    vd = 1.0; // Default percentage of available memory dedicated to this GEMM
    dplasma_info_get(opt, "DPLASMA:GEMM:GPU:mem_ratio", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        vd = strtod(info_value, NULL);
        if(vd <= 0.0 || vd > 1.0) {
            dplasma_error("dplasma_zgemm_gpu_new",
                          "Invalid value for DPLASMA:GEMM:GPU:mem_ratio. Mem ratio must be real in ]0, 1]");
            goto cleanup;
        }
    }
    tile_size = A->mb*A->nb*sizeof(dplasma_complex64_t);
    nb_block_per_tile = (tile_size + gpu_mem_block_size -1 ) / gpu_mem_block_size;
    gpu_mem_nb_blocks = vd * gpu_mem_nb_blocks;
    nb_tile_per_gpu = gpu_mem_nb_blocks / nb_block_per_tile;
    if(0 == nb_tile_per_gpu) {
        dplasma_error("dplasma_zgemm_gpu_new",
                      "Not enough memory on the GPU to store a single tile!");
        goto cleanup;
    }

    mt = A->mt;
    nt = B->nt;
    kt = A->nt;

    // We find (b, c) such that b*c tiles of C fill at most 75% of the GPU memory
    // and b*p divides MT
    // and c*q divides NT
    vd = 0.75; // By default it's up to 75% of the memory to host C
    dplasma_info_get(opt, "DPLASMA:GEMM:GPU:c_ratio", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        vd = strtod(info_value, NULL);
        if(vd <= 0.0 || vd >= 1.0) {
            dplasma_error("dplasma_zgemm_gpu_new",
                          "Invalid value for DPLASMA:GEMM:GPU:c_ratio. Ratio of memory dedicated to hosting tiles of C must be real in ]0, 1[");
            goto cleanup;
        }
    }
    int fact = 1;
    while( fact < mt && fact < nt && ((mt/fact) * (nt/fact)) / (p * q * nbgpu) > nb_tile_per_gpu * vd ) fact++;
    b = mt/(p*fact);
    c = nt/(q*fact);

    // Usually, look ahead is detrimental to performance when fact=1
    // and critical to performance when fact > 1
    look_ahead = 1 + (fact > 1);
    dplasma_info_get(opt, "DPLASMA:GEMM:GPU:look_ahead", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        look_ahead = atoi(info_value);
        if(look_ahead <= 0) {
            dplasma_error("dplasma_zgemm_gpu_new",
                          "Invalid value for DPLASMA:GEMM:GPU:look_ahead. Look ahead must be 1 or more");
            goto cleanup;
        }
    }

    // OK, now we fill up each GPU with data from A and B
    int c_per_gpu = c / nbgpu;
    int maxd = (nb_tile_per_gpu - b*c_per_gpu)/(b+c_per_gpu) - 1;
    d = maxd < kt ? maxd : kt;
    d = d < 1? 1: d;

    // Now we let the user overwrite the b, c and d parameters
    dplasma_info_get(opt, "DPLASMA:GEMM:GPU:b", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        b = atoi(info_value);
        if(b <= 0 || b*p > A->mt) {
            dplasma_error("dplasma_zgemm_gpu_new",
                          "Invalid value for DPLASMA:GEMM:GPU:b. b must be > 0 and b*P less or equal to A.mt");
            goto cleanup;
        }
    }
    dplasma_info_get(opt, "DPLASMA:GEMM:GPU:c", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        c = atoi(info_value);
        if(c <= 0 || c*q > A->nt) {
            dplasma_error("dplasma_zgemm_gpu_new",
                          "Invalid value for DPLASMA:GEMM:GPU:c. c must be > 0 and c*Q less or equal to A.nt");
            goto cleanup;
        }
    }
    dplasma_info_get(opt, "DPLASMA:GEMM:GPU:d", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found ) {
        d = atoi(info_value);
        if(d <= 0 || d > B->mt) {
            dplasma_error("dplasma_zgemm_gpu_new",
                          "Invalid value for DPLASMA:GEMM:GPU:d. d must be > 0 and less or equal to B.mt");
            goto cleanup;
        }
    }

    assert(d <= B->mt);
    assert( b*p <= A->mt );
    assert( c*q <= C->nt );

    {
        dplasma_data_collection_t * ddc_A = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)A);
        dplasma_data_collection_t * ddc_B = dplasma_wrap_data_collection((parsec_tiled_matrix_t*)B);
        dplasma_data_collection_t * ddc_C = dplasma_wrap_data_collection(C);

        PARSEC_DEBUG_VERBOSE(3, parsec_debug_output, "zgemm_NN_gpu");
        parsec_zgemm_NN_gpu_taskpool_t *tp;
        tp = parsec_zgemm_NN_gpu_new(transA, transB, alpha, beta,
                                     ddc_A, ddc_B, ddc_C, b, c, d, p, q, look_ahead,
                                     nbgpu, dev_index);

        u = C->super.myrank / q;
        v = C->super.myrank % q;

        M = A->mt;
        Mbound = M / (p * b);
        Mlim = p * b * Mbound + u;
        tp->_g_xMax = Mbound + (Mlim < M) - 1;

        N = C->nt;
        Nbound = N / (c * q);
        Nlim = c * q * Nbound + v;
        tp->_g_yMax = Nbound + (Nlim < N) - 1;

        K = B->mt;
        tp->_g_zMax = (K + d - 1) / d - 1;

#if defined(DPLASMA_HAVE_HIP)
        /* It doesn't cost anything to define these infos if we have HIP but
         * don't have GPUs on the current machine, so we do it non-conditionally */
        tp->_g_hip_handles_infokey = parsec_info_lookup(&parsec_per_stream_infos, "DPLASMA::HIP::HANDLES", NULL);
#else
        tp->_g_hip_handles_infokey = PARSEC_INFO_ID_UNDEFINED;
#endif

        int shape = 0;
        dplasma_setup_adtt_all_loc( ddc_A,
                                    parsec_datatype_double_complex_t,
                                    PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                    &shape);
        dplasma_setup_adtt_all_loc( ddc_B,
                                    parsec_datatype_double_complex_t,
                                    PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                    &shape);
        dplasma_setup_adtt_all_loc( ddc_C,
                                    parsec_datatype_double_complex_t,
                                    PARSEC_MATRIX_FULL/*uplo*/, 1/*diag:for PARSEC_MATRIX_UPPER or PARSEC_MATRIX_LOWER types*/,
                                    &shape);
        assert(shape == MAX_SHAPES);

        zgemm_tp = (parsec_taskpool_t *) tp;
        return zgemm_tp;
    }

  cleanup:
    if(NULL != dev_index)
        free(dev_index);
    return NULL;
}
#endif /* DPLASMA_HAVE_CUDA || DPLASMA_HAVE_HIP */

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgemm_New - Generates the taskpool that performs one of the following
 *  matrix-matrix operations. WARNING: The computations are not done by this call.
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = conjg( X' )
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   A is not transposed;
 *          = dplasmaTrans:     A is transposed;
 *          = dplasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed, not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   B is not transposed;
 *          = dplasmaTrans:     B is transposed;
 *          = dplasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C are overwritten by the matrix (
 *          alpha*op( A )*op( B ) + beta*C )
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_zgemm_Destruct();
 *
 *******************************************************************************
 *
 * @sa dplasma_zgemm
 * @sa dplasma_zgemm_Destruct
 * @sa dplasma_cgemm_New
 * @sa dplasma_dgemm_New
 * @sa dplasma_sgemm_New
 *
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zgemm_New_ex( dplasma_enum_t transA, dplasma_enum_t transB,
                      dplasma_complex64_t alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                      dplasma_complex64_t beta,  parsec_tiled_matrix_t* C, dplasma_info_t opt)
{
    parsec_taskpool_t* zgemm_tp = NULL;
    int info_found = 0;
    char info_value[DPLASMA_MAX_INFO_VAL];

    /* Check input arguments */
    if ((transA != dplasmaNoTrans) && (transA != dplasmaTrans) && (transA != dplasmaConjTrans)) {
        dplasma_error("dplasma_zgemm_New", "illegal value of transA");
        return NULL /*-1*/;
    }
    if ((transB != dplasmaNoTrans) && (transB != dplasmaTrans) && (transB != dplasmaConjTrans)) {
        dplasma_error("dplasma_zgemm_New", "illegal value of transB");
        return NULL /*-2*/;
    }

    dplasma_info_get(opt, "DPLASMA:GEMM:SPARSE_IN_TILE", DPLASMA_MAX_INFO_VAL, info_value, &info_found);
    if( info_found && atoi(info_value) != 0 ) {
        zgemm_tp = dplasma_zgemm_sparse_new(transA, transB, alpha, A, B, beta, C, opt);
        if( NULL != zgemm_tp ) {
            return zgemm_tp;
        }
    }

    if ( C->dtype & parsec_matrix_block_cyclic_type ) {
#if defined(DPLASMA_HAVE_CUDA) || defined(DPLASMA_HAVE_HIP)
        int nb_gpu_devices = 0, devid;
        int p = ((parsec_matrix_block_cyclic_t*)C)->grid.rows;
        int q = ((parsec_matrix_block_cyclic_t*)C)->grid.cols;
        int64_t gpu_mem_block_size = 0;
        int64_t gpu_mem_nb_blocks = -1;
        for(devid = 0; devid < (int)parsec_nb_devices; devid++) {
            parsec_device_module_t *device = parsec_mca_device_get(devid);
            if( PARSEC_DEV_CUDA == device->type || PARSEC_DEV_HIP == device->type ) {
                parsec_device_gpu_module_t *gpu_device = (parsec_device_gpu_module_t*)device;
                nb_gpu_devices++;
                if( 0 == gpu_mem_block_size )
                    gpu_mem_block_size = gpu_device->mem_block_size;
                if( -1 == gpu_mem_nb_blocks || gpu_device->mem_nb_blocks < gpu_mem_nb_blocks )
                    gpu_mem_nb_blocks = gpu_device->mem_nb_blocks;
            }
        }
        if(0 < nb_gpu_devices) {
            int64_t tile_size = A->mb*A->nb*sizeof(dplasma_complex64_t);
            int64_t nb_block_per_tile = (tile_size + gpu_mem_block_size -1 ) / gpu_mem_block_size;
            int64_t nb_tile_per_gpu = gpu_mem_nb_blocks / nb_block_per_tile;
            int64_t nb_active_tiles_per_gpu = C->mt * C->nt / (p*q) + dplasma_aux_getGEMMLookahead(C) * A->mt / p + dplasma_aux_getGEMMLookahead(C) * B->nt / q;
            if( (A->dtype & parsec_matrix_block_cyclic_type) &&
                (B->dtype & parsec_matrix_block_cyclic_type) &&
                transA == dplasmaNoTrans &&
                transB == dplasmaNoTrans &&
                (nb_active_tiles_per_gpu > 0.95* nb_tile_per_gpu) ) {
                zgemm_tp = dplasma_zgemm_gpu_new(transA, transB, alpha, A, B, beta, C, opt);
                return zgemm_tp;
            }
        }
#endif /* DPLASMA_HAVE_CUDA || DPLASMA_HAVE_HIP */
        zgemm_tp = dplasma_zgemm_summa_new(transA, transB, alpha, A, B, beta, C, opt);
        return zgemm_tp;
    }
    zgemm_tp = dplasma_zgemm_default_new(transA, transB, alpha, A, B, beta, C, opt);
    return zgemm_tp;
}

parsec_taskpool_t*
dplasma_zgemm_New( dplasma_enum_t transA, dplasma_enum_t transB,
                   dplasma_complex64_t alpha, const parsec_tiled_matrix_t* A, const parsec_tiled_matrix_t* B,
                   dplasma_complex64_t beta,  parsec_tiled_matrix_t* C)
{
    parsec_taskpool_t *tp;
    dplasma_info_t opt;
    dplasma_info_create(&opt);
    tp = dplasma_zgemm_New_ex(transA, transB, alpha, A, B, beta, C, opt);
    dplasma_info_free(&opt);
    return tp;
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgemm_Destruct - Free the data structure associated to an taskpool
 *  created with dplasma_zgemm_New().
 *
 *******************************************************************************
 *
 * @param[in,out] taskpool
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgemm_New
 * @sa dplasma_zgemm
 *
 ******************************************************************************/
void
dplasma_zgemm_Destruct( parsec_taskpool_t *tp )
{
    parsec_zgemm_NN_taskpool_t *zgemm_tp = (parsec_zgemm_NN_taskpool_t *)tp;
    dplasma_data_collection_t *ddc_A = NULL, *ddc_B = NULL, *ddc_C = NULL;
    parsec_matrix_block_cyclic_t *sparseA = NULL, *sparseB = NULL, *sparseC = NULL;
    parsec_tiled_matrix_t *denseC = NULL;

    switch( zgemm_tp->_g_gemm_type ) {
    case DPLASMA_ZGEMM_NN:
    case DPLASMA_ZGEMM_NT:
    case DPLASMA_ZGEMM_TN:
    case DPLASMA_ZGEMM_TT:
        ddc_A = zgemm_tp->_g_ddescA;
        ddc_B = zgemm_tp->_g_ddescB;
        ddc_C = zgemm_tp->_g_ddescC;
        break;
    case DPLASMA_ZGEMM_NN_SPARSE: {
        parsec_zgemm_NN_sparse_taskpool_t *zgemm_sparse_tp = (parsec_zgemm_NN_sparse_taskpool_t*)tp;
        sparseA = (parsec_matrix_block_cyclic_t*)zgemm_sparse_tp->_g_ddescA;
        sparseB = (parsec_matrix_block_cyclic_t*)zgemm_sparse_tp->_g_ddescB;
        sparseC = (parsec_matrix_block_cyclic_t*)zgemm_sparse_tp->_g_ddescC;
        denseC = zgemm_sparse_tp->_g_denseC;
        /* Finalize: scatter CSR C into the user's dense C while the taskpool
         * and tile copies are still valid (before parsec_taskpool_free). */
        if( NULL != sparseC && NULL != denseC ) {
            dplasma_zgemm_sparse_c_to_dense(sparseC, denseC);
        }
        break; }
    case DPLASMA_ZGEMM_NN_SUMMA:
    case DPLASMA_ZGEMM_NT_SUMMA:
    case DPLASMA_ZGEMM_TN_SUMMA:
    case DPLASMA_ZGEMM_TT_SUMMA: {
        parsec_zgemm_NN_summa_taskpool_t *zgemm_summa_tp = (parsec_zgemm_NN_summa_taskpool_t *)tp;
        ddc_A = zgemm_summa_tp->_g_ddescA;
        ddc_B = zgemm_summa_tp->_g_ddescB;
        ddc_C = zgemm_summa_tp->_g_ddescC;
        parsec_tiled_matrix_t* Cdist = (parsec_tiled_matrix_t*)zgemm_summa_tp->_g_Cdist;
        if ( NULL != Cdist ) {
            parsec_tiled_matrix_destroy( Cdist );
            free( Cdist );
        }
        break; }
#if defined(DPLASMA_HAVE_CUDA) || defined(DPLASMA_HAVE_HIP)
    case DPLASMA_ZGEMM_NN_GPU: {
        parsec_zgemm_NN_gpu_taskpool_t *zgemm_gpu_tp = (parsec_zgemm_NN_gpu_taskpool_t *)tp;
        ddc_A = zgemm_gpu_tp->_g_ddescA;
        ddc_B = zgemm_gpu_tp->_g_ddescB;
        ddc_C = zgemm_gpu_tp->_g_ddescC;
        free(zgemm_gpu_tp->_g_gpu_device_index);
        break; }
#endif /* DPLASMA_HAVE_CUDA || defined(DPLASMA_HAVE_HIP) */
    default:
        parsec_warning("Invalid GEMM taskpool type during destruct!");
    }

    if( NULL != ddc_A ) {
        dplasma_clean_adtt_all_loc(ddc_A, MAX_SHAPES);
    }
    if( NULL != ddc_B ) {
        dplasma_clean_adtt_all_loc(ddc_B, MAX_SHAPES);
    }
    if( NULL != ddc_C ) {
        dplasma_clean_adtt_all_loc(ddc_C, MAX_SHAPES);
    }

    parsec_taskpool_free(tp);

    if( NULL != sparseA ) {
        dplasma_zgemm_destroy_sparse_tile_matrix(sparseA);
    }
    if( NULL != sparseB ) {
        dplasma_zgemm_destroy_sparse_tile_matrix(sparseB);
    }
    if( NULL != sparseC ) {
        dplasma_zgemm_destroy_sparse_tile_matrix(sparseC);
    }

    /* free the dplasma_data_collection_t, after the tp stops referring to them */
    if( NULL != ddc_A ) {
        dplasma_unwrap_data_collection(ddc_A);
    }
    if( NULL != ddc_B ) {
        dplasma_unwrap_data_collection(ddc_B);
    }
    if( NULL != ddc_C ) {
        dplasma_unwrap_data_collection(ddc_C);
    }
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_complex64
 *
 *  dplasma_zgemm - Performs one of the following matrix-matrix operations
 *
 *    \f[ C = \alpha [op( A )\times op( B )] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X' or op( X ) = conjg( X' )
 *
 *  alpha and beta are scalars, and A, B and C  are matrices, with op( A )
 *  an m by k matrix, op( B ) a k by n matrix and C an m by n matrix.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] transA
 *          Specifies whether the matrix A is transposed, not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   A is not transposed;
 *          = dplasmaTrans:     A is transposed;
 *          = dplasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] transB
 *          Specifies whether the matrix B is transposed, not transposed or conjugate transposed:
 *          = dplasmaNoTrans:   B is not transposed;
 *          = dplasmaTrans:     B is transposed;
 *          = dplasmaConjTrans: B is conjugate transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha
 *
 * @param[in] A
 *          Descriptor of the distributed matrix A.
 *
 * @param[in] B
 *          Descriptor of the distributed matrix B.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          Descriptor of the distributed matrix C.
 *          On exit, the data described by C are overwritten by the matrix (
 *          alpha*op( A )*op( B ) + beta*C )
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 *******************************************************************************
 *
 * @sa dplasma_zgemm_New
 * @sa dplasma_zgemm_Destruct
 * @sa dplasma_cgemm
 * @sa dplasma_dgemm
 * @sa dplasma_sgemm
 *
 ******************************************************************************/
int
dplasma_zgemm( parsec_context_t *parsec,
               dplasma_enum_t transA, dplasma_enum_t transB,
               dplasma_complex64_t alpha, const parsec_tiled_matrix_t *A,
                                        const parsec_tiled_matrix_t *B,
               dplasma_complex64_t beta,        parsec_tiled_matrix_t *C)
{
    parsec_taskpool_t *parsec_zgemm = NULL;
    int M, N, K;
    int Am, An, Ai, Aj, Amb, Anb;
    int Bm, Bn, Bi, Bj, Bmb, Bnb;

    /* Check input arguments */
    if ((transA != dplasmaNoTrans) && (transA != dplasmaTrans) && (transA != dplasmaConjTrans)) {
        dplasma_error("dplasma_zgemm", "illegal value of transA");
        return -1;
    }
    if ((transB != dplasmaNoTrans) && (transB != dplasmaTrans) && (transB != dplasmaConjTrans)) {
        dplasma_error("dplasma_zgemm", "illegal value of transB");
        return -2;
    }

    if ( transA == dplasmaNoTrans ) {
        Am  = A->m;
        An  = A->n;
        Amb = A->mb;
        Anb = A->nb;
        Ai  = A->i;
        Aj  = A->j;
    } else {
        Am  = A->n;
        An  = A->m;
        Amb = A->nb;
        Anb = A->mb;
        Ai  = A->j;
        Aj  = A->i;
    }

    if ( transB == dplasmaNoTrans ) {
        Bm  = B->m;
        Bn  = B->n;
        Bmb = B->mb;
        Bnb = B->nb;
        Bi  = B->i;
        Bj  = B->j;
    } else {
        Bm  = B->n;
        Bn  = B->m;
        Bmb = B->nb;
        Bnb = B->mb;
        Bi  = B->j;
        Bj  = B->i;
    }

    if ( (Amb != C->mb) || (Anb != Bmb) || (Bnb != C->nb) ) {
        dplasma_error("dplasma_zgemm", "tile sizes have to match");
        return -101;
    }
    if ( (Am != C->m) || (An != Bm) || (Bn != C->n) ) {
        dplasma_error("dplasma_zgemm", "sizes of matrices have to match");
        return -101;
    }
    if ( (Ai != C->i) || (Aj != Bi) || (Bj != C->j) ) {
        dplasma_error("dplasma_zgemm", "start indexes have to match");
        return -101;
    }

    M = C->m;
    N = C->n;
    K = An;

    /* Quick return */
    if (M == 0 || N == 0 ||
        ((alpha == (dplasma_complex64_t)0.0 || K == 0) && beta == (dplasma_complex64_t)1.0))
        return 0;

    parsec_zgemm = dplasma_zgemm_New(transA, transB,
                                    alpha, A, B,
                                    beta, C);

    if ( parsec_zgemm != NULL )
    {
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_zgemm);
        dplasma_wait_until_completion(parsec);
        dplasma_zgemm_Destruct( parsec_zgemm );
        return 0;
    }
    return -101;
}
