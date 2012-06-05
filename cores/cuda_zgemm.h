/*
 * Copyright (c) 2010-2012 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> z c d s
 *
 */

#ifndef _cuda_zgemm_h_
#define _cuda_zgemm_h_

#include "dague_config.h"
#include "gpu_data.h"
#include "dague.h"
#include "execution_unit.h"
#include "scheduling.h"
#include "fifo.h"
#include "data_dist/matrix/matrix.h"

#define GEMM_KEY(M, N) (uint32_t)(NULL == dague_gpu_map.desc ? \
                                  0 : (M) * (((tiled_matrix_desc_t*)(dague_gpu_map.desc))->lmt) + (N))

int gpu_kernel_init_zgemm( dague_context_t* dague_context,
                           tiled_matrix_desc_t *tileA );

int gpu_zgemm( dague_execution_unit_t* eu_context,
               dague_execution_context_t* this_task,
               int uplo );

#endif /* _cuda_zgemm_h_ */