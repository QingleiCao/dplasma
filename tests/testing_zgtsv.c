/*
 * Copyright (c) 2009-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "parsec/data_dist/matrix/two_dim_rectangle_cyclic.h"

static int dplasma_add_diag_operator( parsec_execution_stream_t *es,
                         const parsec_tiled_matrix_t *descA,
                         void *_A,
                         dplasma_enum_t uplo, int m, int n,
                         void *op_data ) {
    for(int j = 0; j <  descA->nb; j++) {
        ((double *)_A)[descA->mb+j] += ((double *)op_data)[0];
    }
    return 0;
}


int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    int Aseed = 3872;
    int Bseed = 4674;
    double* add_diag = (double *)malloc(sizeof(double)); 
    add_diag[0] = 1000000.0;
    double gflops = -1.0, flops = 1;
    iparam_default_facto(iparam); 
    iparam_default_ibnbmb(iparam, 0, 180, 180); 
    //iparam[IPARAM_NGPUS] = 0;

    /* Initialize PaRSEC */
    parsec = setup_parsec(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam);
    NB=3;

    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
            parsec_matrix_block_cyclic, (&dcA, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_TILE,
                rank, MB, NB, MB, N, 0, 0,
                MB, N, 1, nodes, 1, 1, 0, 0));

    PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
            parsec_matrix_block_cyclic, (&dcB, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_TILE,
                rank, MB, 1, MB, N, 0, 0,
                MB, N, 1, nodes, 1, 1, 0, 0));

    for(int t = 0; t < nruns; t++) {
        /* matrix (re)generation */
        if(loud > 3) printf("+++ Generate matrices ... ");
        parsec_apply(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcA, dplasma_add_diag_operator, add_diag); 
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_t *)&dcA, Aseed);
        dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_t *)&dcB, Bseed);
        if(loud > 3) printf("Done\n");

        parsec_devices_release_memory();
        SYNC_TIME_START(); 
        dplasma_zgtsv( parsec, (parsec_tiled_matrix_t*)&dcA, (parsec_tiled_matrix_t*)&dcB);
        SYNC_TIME_PRINT(rank, ("dplasma_zgtsv nodes %d P %d Q %d gpus %d N %d MB %d\n",
                    nodes, P, Q, gpus, N, MB));
        parsec_devices_reset_load(parsec);
    }

    parsec_data_free(dcA.mat); dcA.mat = NULL;
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcA);
    parsec_data_free(dcB.mat); dcB.mat = NULL;
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcB);

    cleanup_parsec(parsec, iparam);
    return 0;
}
