/*
 * Copyright (c) 2010-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2013      Inria. All rights reserved.
 * $COPYRIGHT
 *
 * @precisions normal z -> s d c
 *
 */

#include "dplasma.h"
#include "dplasma/types.h"
#include "dplasma/types_lapack.h"
#if defined(DPLASMA_HAVE_CUDA)
#include <cublas_v2.h>
#include "potrf_cublas_utils.h"
#include "parsec/utils/zone_malloc.h"
#endif  /* defined(DPLASMA_HAVE_CUDA) */
#include "dplasmaaux.h"

#include "zgtsv.h"
#include "cores/dplasma_plasmatypes.h"

#define MAX_SHAPES 2


/**
 *******************************************************************************
 ******************************************************************************/
parsec_taskpool_t*
dplasma_zgtsv_New( parsec_tiled_matrix_t *A,
                   parsec_tiled_matrix_t *B)
{
    parsec_zgtsv_taskpool_t *parsec_zgtsv = NULL;
    parsec_taskpool_t *tp = NULL;
    dplasma_data_collection_t * ddc_A = dplasma_wrap_data_collection(A);
    dplasma_data_collection_t * ddc_B = dplasma_wrap_data_collection(B);

    tp = (parsec_taskpool_t*)parsec_zgtsv_new( ddc_A, ddc_B);

    parsec_zgtsv =  (parsec_zgtsv_taskpool_t*)tp;

#if defined(DPLASMA_HAVE_CUDA)
    /* It doesn't cost anything to define these infos if we have CUDA but
     * don't have GPUs on the current machine, so we do it non-conditionally */
    parsec_zgtsv->_g_CuHandlesID = parsec_info_lookup(&parsec_per_stream_infos, "DPLASMA::CUDA::HANDLES", NULL);
#else
    parsec_zgtsv->_g_CuHandlesID = PARSEC_INFO_ID_UNDEFINED;
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

    assert(shape == MAX_SHAPES);
    return tp;
}

/**
 *******************************************************************************
 ******************************************************************************/
void
dplasma_zgtsv_Destruct( parsec_taskpool_t *tp )
{
    parsec_zgtsv_taskpool_t *parsec_zgtsv = (parsec_zgtsv_taskpool_t *)tp;
    dplasma_clean_adtt_all_loc(parsec_zgtsv->_g_ddescA, MAX_SHAPES);
    dplasma_data_collection_t * ddc_A = parsec_zgtsv->_g_ddescA;
    dplasma_clean_adtt_all_loc(parsec_zgtsv->_g_ddescB, MAX_SHAPES);
    dplasma_data_collection_t * ddc_B = parsec_zgtsv->_g_ddescB;

    parsec_taskpool_free(tp);
    /* free the dplasma_data_collection_t */
    dplasma_unwrap_data_collection(ddc_A);
    dplasma_unwrap_data_collection(ddc_B);
}

/**
 *******************************************************************************
 ******************************************************************************/
int
dplasma_zgtsv( parsec_context_t *parsec,
                parsec_tiled_matrix_t *A,
                parsec_tiled_matrix_t *B )
{
    parsec_taskpool_t *parsec_zgtsv = NULL;

    parsec_zgtsv = dplasma_zgtsv_New( A, B ); 

    if ( parsec_zgtsv != NULL )
    {
        parsec_context_add_taskpool( parsec, (parsec_taskpool_t*)parsec_zgtsv);
        dplasma_wait_until_completion(parsec);
        dplasma_zgtsv_Destruct( parsec_zgtsv );
    }
    return 0; 
}
