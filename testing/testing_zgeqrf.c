/*
 * Copyright (c) 2009-2011 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * @precisions normal z -> s d c
 *
 */

#include "common.h"
#include "data_dist/matrix/two_dim_rectangle_cyclic.h"

#if defined(HAVE_CUDA) && defined(PRECISION_s)
#include "dplasma/cores/cuda_stsmqr.h"
#endif

#define FMULS_GEQRF(M, N) (((M) > (N)) ? ((N) * ((N) * (  0.5-(1./3.) * (N) + (M)) + (M))) \
                                       : ((M) * ((M) * ( -0.5-(1./3.) * (M) + (N)) + 2.*(N))))
#define FADDS_GEQRF(M, N) (((M) > (N)) ? ((N) * ((N) * (  0.5-(1./3.) * (N) + (M)))) \
                                       : ((M) * ((M) * ( -0.5-(1./3.) * (M) + (N)) + (N))))

static int check_orthogonality(dague_context_t *dague, tiled_matrix_desc_t *Q);
static int check_factorization(dague_context_t *dague, tiled_matrix_desc_t *Aorig, tiled_matrix_desc_t *A, tiled_matrix_desc_t *Q);

int main(int argc, char ** argv)
{
    dague_context_t* dague;
    int iparam[IPARAM_SIZEOF];

    /* Set defaults for non argv iparams */
    iparam_default_facto(iparam);
    iparam_default_ibnbmb(iparam, 48, 144, 144);
    iparam[IPARAM_LDA] = -'m';
    iparam[IPARAM_LDB] = -'m';
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    iparam[IPARAM_NGPUS] = 0;
#endif

    /* Initialize DAGuE */
    dague = setup_dague(argc, argv, iparam);
    PASTE_CODE_IPARAM_LOCALS(iparam)
    PASTE_CODE_FLOPS_COUNT(FADDS_GEQRF, FMULS_GEQRF, ((DagDouble_t)M,(DagDouble_t)N))
      
    LDA = max(M, LDA);
    /* initializing matrix structure */
    PASTE_CODE_ALLOCATE_MATRIX(ddescA, 1, 
        two_dim_block_cyclic, (&ddescA, matrix_ComplexDouble, 
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescT, 1, 
        two_dim_block_cyclic, (&ddescT, matrix_ComplexDouble, 
                               nodes, cores, rank, IB, NB, MT*IB, N, 0, 0, 
                               MT*IB, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescA0, check, 
        two_dim_block_cyclic, (&ddescA0, matrix_ComplexDouble, 
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
                               M, N, SMB, SNB, P));
    PASTE_CODE_ALLOCATE_MATRIX(ddescQ, check, 
        two_dim_block_cyclic, (&ddescQ, matrix_ComplexDouble, 
                               nodes, cores, rank, MB, NB, LDA, N, 0, 0, 
                               M, N, SMB, SNB, P));

#if defined(DAGUE_PROF_TRACE)
    ddescA.super.super.key = strdup("A");
    ddescT.super.super.key = strdup("T");
#endif

    /* load the GPU kernel */
#if defined(HAVE_CUDA) && defined(PRECISION_s)
    if(iparam[IPARAM_NGPUS] > 0)
    {
        if(loud) printf("+++ Load GPU kernel ... ");
        if(0 != stsmqr_cuda_init(dague, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescT)) 
        {
            fprintf(stderr, "XXX Unable to load GPU kernel.\n");
            exit(3);
        }
        if(loud) printf("Done\n");
    }
#endif

    /* matrix generation */
    if(loud > 2) printf("+++ Generate matrices ... ");
    dplasma_zplrnt( dague, (tiled_matrix_desc_t *)&ddescA, 3872);
    if( check )
        dplasma_zlacpy( dague, PlasmaUpperLower,
                        (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescA0 );
    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&ddescT);
    if(loud > 2) printf("Done\n");
    
    /* Create DAGuE */
    PASTE_CODE_ENQUEUE_KERNEL(dague, zgeqrf, 
                              ((tiled_matrix_desc_t*)&ddescA,
                               (tiled_matrix_desc_t*)&ddescT));
    
    /* lets rock! */
    PASTE_CODE_PROGRESS_KERNEL(dague, zgeqrf);

    if( check ) {
        int info_ortho, info_facto;

        if(loud > 2) fprintf(stderr, "+++ Generate the Q ...");
        dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&ddescQ);
        dplasma_zungqr( dague, (tiled_matrix_desc_t *)&ddescA, (tiled_matrix_desc_t *)&ddescT, 
                        (tiled_matrix_desc_t *)&ddescQ);
        if(loud > 2) fprintf(stderr, "Done\n");

        /* Check the orthogonality, factorization and the solution */
        info_ortho = check_orthogonality(dague, (tiled_matrix_desc_t *)&ddescQ);
        info_facto = check_factorization(dague, (tiled_matrix_desc_t *)&ddescA0, 
                                         (tiled_matrix_desc_t *)&ddescA, 
                                         (tiled_matrix_desc_t *)&ddescQ);


        dague_data_free(ddescA0.mat);
        dague_data_free(ddescQ.mat);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescA0);
        dague_ddesc_destroy((dague_ddesc_t*)&ddescQ);
    }

#if defined(HAVE_CUDA) && defined(PRECISION_s)
    if(iparam[IPARAM_NGPUS] > 0) 
    {
        stsmqr_cuda_fini(dague);
    }
#endif

    dague_data_free(ddescA.mat);
    dague_data_free(ddescT.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescA);
    dague_ddesc_destroy((dague_ddesc_t*)&ddescT);
    
    cleanup_dague(dague, iparam);

    return EXIT_SUCCESS;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */

static int check_orthogonality(dague_context_t *dague, tiled_matrix_desc_t *Q)
{
    two_dim_block_cyclic_t *twodQ = (two_dim_block_cyclic_t *)Q;
    double normQ = 999999.0;
    double result;
    double eps = LAPACKE_dlamch_work('e');
    int info_ortho;
    int M = Q->m;
    int N = Q->n;
    int minMN = min(M, N);

    PASTE_CODE_ALLOCATE_MATRIX(Id, 1, 
        two_dim_block_cyclic, (&Id, matrix_ComplexDouble, 
                               Q->super.nodes, Q->super.cores, twodQ->grid.rank, 
                               Q->mb, Q->nb, minMN, minMN, 0, 0, 
                               minMN, minMN, twodQ->grid.strows, twodQ->grid.stcols, twodQ->grid.rows));

    dplasma_zlaset( dague, PlasmaUpperLower, 0., 1., (tiled_matrix_desc_t *)&Id);

    /* Perform Id - Q'Q (could be done with Herk) */
    if ( M >= N ) {
      dplasma_zgemm( dague, PlasmaConjTrans, PlasmaNoTrans, 
                     1.0, Q, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    } else {
      dplasma_zgemm( dague, PlasmaNoTrans, PlasmaConjTrans, 
                     1.0, Q, Q, -1.0, (tiled_matrix_desc_t*)&Id );
    }

    normQ = dplasma_zlange(dague, PlasmaMaxNorm, (tiled_matrix_desc_t*)&Id);

    result = normQ / (minMN * eps);
    printf("============\n");
    printf("Checking the orthogonality of Q \n");
    printf("||Id-Q'*Q||_oo / (N*eps) = %e \n", result);

    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        printf("-- Orthogonality is suspicious ! \n");
        info_ortho=1;
    }
    else {
        printf("-- Orthogonality is CORRECT ! \n");
        info_ortho=0;
    }

    dague_data_free(Id.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&Id);
    return info_ortho;
}

/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */

static int check_factorization(dague_context_t *dague, tiled_matrix_desc_t *Aorig, tiled_matrix_desc_t *A, tiled_matrix_desc_t *Q)
{
    two_dim_block_cyclic_t *twodA = (two_dim_block_cyclic_t *)A;
    double Anorm, Rnorm;
    double result;
    double eps = LAPACKE_dlamch_work('e');
    int info_factorization;
    int M = A->m;
    int N = A->n;
    int minMN = min(M, N);

    PASTE_CODE_ALLOCATE_MATRIX(Residual, 1, 
        two_dim_block_cyclic, (&Residual, matrix_ComplexDouble, 
                               A->super.nodes, A->super.cores, twodA->grid.rank, 
                               A->mb, A->nb, M, N, 0, 0, 
                               M, N, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    PASTE_CODE_ALLOCATE_MATRIX(RL, 1, 
        two_dim_block_cyclic, (&RL, matrix_ComplexDouble, 
                               A->super.nodes, A->super.cores, twodA->grid.rank, 
                               A->mb, A->nb, minMN, minMN, 0, 0, 
                               minMN, minMN, twodA->grid.strows, twodA->grid.stcols, twodA->grid.rows));

    /* Extract the L */
    dplasma_zlacpy( dague, PlasmaUpperLower, Aorig, (tiled_matrix_desc_t *)&Residual );

    dplasma_zlaset( dague, PlasmaUpperLower, 0., 0., (tiled_matrix_desc_t *)&RL);
    if (M >= N) {
        /* Extract the R */
        dplasma_zlacpy( dague, PlasmaUpper, A, (tiled_matrix_desc_t *)&RL );
        
        /* Perform Residual = Aorig - Q*R */
        dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, 
                       -1.0, Q, (tiled_matrix_desc_t *)&RL, 
                       1.0, (tiled_matrix_desc_t *)&Residual);
    } else {
        /* Extract the L */
        dplasma_zlacpy( dague, PlasmaLower, A, (tiled_matrix_desc_t *)&RL );
        
        /* Perform Residual = Aorig - L*Q */
        dplasma_zgemm( dague, PlasmaNoTrans, PlasmaNoTrans, 
                       -1.0, (tiled_matrix_desc_t *)&RL, Q, 
                       1.0, (tiled_matrix_desc_t *)&Residual);
    }
    
    /* Free RL */
    dague_data_free(RL.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&RL);
    
    Rnorm = dplasma_zlange(dague, PlasmaMaxNorm, (tiled_matrix_desc_t*)&Residual);
    Anorm = dplasma_zlange(dague, PlasmaMaxNorm, Aorig);

    result = Rnorm / ( Anorm * minMN * eps);

    if (M >= N) {
        printf("============\n");
        printf("Checking the QR Factorization \n");
        printf("-- ||A-QR||_oo/(||A||_oo.N.eps) = %e \n", result );
    }
    else {
        printf("============\n");
        printf("Checking the LQ Factorization \n");
        printf("-- ||A-LQ||_oo/(||A||_oo.N.eps) = %e \n", result );
    }

    if ( isnan(result) || isinf(result) || (result > 60.0) ) {
        printf("-- Factorization is suspicious ! \n");
        info_factorization = 1;
    }
    else {
        printf("-- Factorization is CORRECT ! \n");
        info_factorization = 0;
    }

    dague_data_free(Residual.mat);
    dague_ddesc_destroy((dague_ddesc_t*)&Residual);
    return info_factorization;
}
