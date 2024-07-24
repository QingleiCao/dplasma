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
    for(int i = 0; i <  descA->mb; i++) {
        ((double *)_A)[descA->mb+i] += ((double *)op_data)[0];
    }
    return 0;
}


int main(int argc, char ** argv)
{
    parsec_context_t* parsec;
    int iparam[IPARAM_SIZEOF];
    //int Aseed = 3872;
    //int Bseed = 4674;
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
    int batch = N;

    SYNC_TIME_START();
    PASTE_CODE_ALLOCATE_MATRIX(dcA, 1,
            parsec_matrix_block_cyclic, (&dcA, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_TILE,
                rank, MB, NB, MB, NB*batch, 0, 0,
                MB, NB*batch, 1, nodes, 1, 1, 0, 0));

    PASTE_CODE_ALLOCATE_MATRIX(dcB, 1,
            parsec_matrix_block_cyclic, (&dcB, PARSEC_MATRIX_COMPLEX_DOUBLE, PARSEC_MATRIX_TILE,
                rank, MB, 1, MB, batch, 0, 0,
                MB, batch, 1, nodes, 1, 1, 0, 0));
    SYNC_TIME_PRINT(rank, ("Allocate memory\n"));

    //if(0 == rank && loud) fprintf(stderr, "nodes %d P %d Q %d gpus %d cores %d matrix_size %d batch %d\n",
    //        nodes, P, Q, gpus, cores, MB, batch); 

    /* matrix (re)generation */
    SYNC_TIME_START();
    for(int k = 0; k < dcA.super.nb_local_tiles; k++) {
        // Init dcA
        for(int j = 0; j < dcA.super.nb; j++) {
            for(int i = 0; i < dcA.super.mb; i++) {
                if(1 == j) {
                    ((double *)dcA.mat)[k*dcA.super.mb*dcA.super.nb+j*dcA.super.mb+i] = 1000.0;
                } else {
                    ((double *)dcA.mat)[k*dcA.super.mb*dcA.super.nb+j*dcA.super.mb+i] = 1.0;
                }
            }
        }
    }

    // Init dcB
    for(int k = 0; k < dcB.super.nb_local_tiles; k++) {
        for(int j = 0; j < dcB.super.nb; j++) {
            for(int i = 0; i < dcB.super.mb; i++) {
                ((double *)dcB.mat)[k*dcB.super.mb*dcB.super.nb+j*dcB.super.mb+i] = 1.0;
            }
        }
    }
    //parsec_apply(parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcA, dplasma_add_diag_operator, add_diag); 
    //dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_t *)&dcA, Aseed);
    //dplasma_zplrnt( parsec, 0, (parsec_tiled_matrix_t *)&dcB, Bseed);
    SYNC_TIME_PRINT(rank, ("Init matrix\n"));

    for(int t = 0; t < nruns; t++) {
        SYNC_TIME_START(); 
        dplasma_zgtsv( parsec, (parsec_tiled_matrix_t*)&dcA, (parsec_tiled_matrix_t*)&dcB);
        SYNC_TIME_PRINT(rank, ("dplasma_zgtsv nodes %d P %d Q %d gpus %d cores %d matrix_size %d batch %d\n",
                    nodes, P, Q, gpus, cores, MB, batch)); 
    }

#if 0
    // Print result 
    for(int k = 0; k < dcB.super.nb_local_tiles; k++) {
        printf("Batch %d\n", k);
        for(int j = 0; j < dcB.super.nb; j++) {
            for(int i = 0; i < dcB.super.mb; i++) {
                fprintf(stderr, "%lf ", ((double *)dcB.mat)[k*dcB.super.mb*dcB.super.nb+j*dcB.super.mb+i]);
            }
        }
        printf("\n\n");
    }
#endif

    parsec_data_free(dcA.mat); dcA.mat = NULL;
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcA);
    parsec_data_free(dcB.mat); dcB.mat = NULL;
    parsec_tiled_matrix_destroy( (parsec_tiled_matrix_t*)&dcB);

    cleanup_parsec(parsec, iparam);
    return 0;
}
