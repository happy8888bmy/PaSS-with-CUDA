/**
 * PaSS_BLAS.cuh
 * The basic linear algebra subprograms for PaSS
 *
 * @author emfo
 */

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>


/**
    * The vector struct
    */
struct vec {
    uint32_t* n;    // the length
    double* e;      // the array of entries
};

/**
    * The matrix struct
    * <remarks>
    * Note that *(m->col[i]->n) = m->n_row for all i, where m is a mat*
    * </remarks>
    */
struct mat {
    uint32_t n_row; // the number of rows
    uint32_t n_col; // the number of cols
    vec** col;      // the array of columns
};

__device__ vec* construct(const uint32_t n);                            // construct a n-by-1 vector
__device__ mat* construct(const uint32_t p, const uint32_t q);          // construct a p-by-q matrix

__device__ void destruct(const vec* v);                                 // destruct the vector
__device__ void destruct(const mat* m);                                 // destruct the matrix

__device__ void print(const vec* v);                                    // display the vector
__device__ void print(const mat* m);                                    // display the matrix

__device__ vec* copy(const vec* m);                                     // copy the vector
__device__ mat* copy(const mat* v);                                     // copy the matrix

__device__ vec* add(const vec* a, const vec* b);                        // calculate a+b
__device__ mat* add(const mat* a, const mat* b);                        // calculate a+b

__device__ void addeq(vec* a, const vec* b);                            // a += b
__device__ void addeq(mat* a, const mat* b);                            // a += b

__device__ vec* mul(const vec* v, const double d);                      // calculate d*v
__device__ mat* mul(const mat* m, const double d);                      // calculate d*m
__device__ vec* mul(const mat* m, const vec* v);                        // calculate m*v
__device__ vec* mul(const vec* v, const mat* m);                        // calculate m'*v (same as (v'*m)')
__device__ mat* mul(const vec* v, const vec* w);                        // calculate v*w'

__device__ void muleq(vec* v, const double d);                          // v *= d
__device__ void muleq(mat* m, const double d);                          // m *= d
__device__ void muleq(const mat* m, vec* v);                            // v = m*v
__device__ void muleq(vec* v, const mat* m);                            // v = m'*v

__device__ double inner(const vec* v);                                  // calculate sum(v.*v)
__device__ double inner(const vec* a, const vec* b);                    // calculate sum(a.*b)
__device__ double inner(const mat* m);                                  // calculate sum(sum(m.*m))
__device__ double inner(const mat* a, const mat* b);                    // calculate sum(sum(a.*b))

__device__ void insert(vec* v, const double e);                         // add a new entry at the end
__device__ void insert_col(mat* m, const vec* v);                       // add a new col at the end
__device__ void insert_row(mat* m, const vec* v);                       // add a new row at the end

__device__ void shed(vec* v);                                           // remove the last entry
__device__ void shed_col(mat* m);                                       // remove the last col
__device__ void shed_row(mat* m);                                       // remove the last row

__device__ void swap(vec* v, const uint32_t i, const uint32_t j);       // swap two entries
__device__ void swap_col(mat* m, const uint32_t i, const uint32_t j);   // swap two cols
__device__ void swap_row(mat* m, const uint32_t i, const uint32_t j);   // swap two rows
