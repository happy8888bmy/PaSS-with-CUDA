/**
 * PaSS_BLAS.cu
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
    uint32_t n; // the length
    double* e;  // the array of entries
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


/**
 * Construct a n-by-1 vector.
 */
__device__ vec* construct(const uint32_t n){
    vec* v = (vec*)malloc(sizeof(vec));
    v->n = n;
    v->e = (double*)malloc(n * sizeof(double));
    return v;
}


/**
 * Construct a p-by-q matrix.
 */
__device__ mat* construct(const uint32_t p, const uint32_t q){
    mat* a = (mat*) malloc(sizeof(mat));
    a->n_row = p;
    a->n_col = q;
    a->col = (vec**)malloc(q * sizeof(vec*));
    for(uint32_t i = 0; i < q; i++){
        a->col[i] = (vec*)malloc(sizeof(vec));
        a->col[i]->n = p;
        a->col[i]->e = (double*)malloc(p * sizeof(double));
    }
    return a;
}


/**
 * Destruct the vector.
 */
__device__ void destruct(vec* v){
    free(v->e);
    free(v);
}


/**
 * Destruct the matrix.
 */
__device__ void destruct(mat* a){
    for(uint32_t i = 0; i < a->n_col; i++){
        free(a->col[i]->e);
        free(a->col[i]);
    }
    free(a->col);
    free(a);
}


/**
 * Display the vector.
 */
__device__ void print(const vec* v){
    for(uint32_t i = 0; i < v->n; i++){
        printf("%8.3f\n", v->e[i]);
    }
    printf("\n");
}


/**
 * Display the matrix.
 */
__device__ void print(const mat* a){
    for(uint32_t j = 0; j < a->n_row; j++){
        for(uint32_t i = 0; i < a->n_col; i++){
            printf("%8.3f", a->col[i]->e[j]);
        }
        printf("\n");
    }
    printf("\n");
}


/**
 * Copy the vector.
 */
__device__ vec* copy(const vec* v){
    vec* u = construct(v->n);
    memcpy(u->e, v->e, v->n * sizeof(double));
    return u;
}


/**
 * Copy the matrix.
 */
__device__ mat* copy(const mat* a){
    mat* c = construct(a->n_row, a->n_col);
    for(uint32_t i = 0; i < a->n_col; i++){
        memcpy(c->col[i], a->col[i], a->n_row * sizeof(double));
    }
    return c;
}


/**
 * u = v+w.
 */
__device__ vec* add(const vec* v, const vec* w){
    if(v->n != w->n){
        printf("add v+w not aligned!\n");
        return NULL;
    }
    vec* u = construct(v->n);
    for(uint32_t i = 0; i < v->n; i++){
        u->e[i] = v->e[i] + w->e[i];
    }
    return u;
}


/**
 * c = a+b.
 */
__device__ mat* add(const mat* a, const mat* b){
    if(a->n_col != b->n_col || a->n_row != b->n_row){
        printf("add a+b not aligned!\n");
        return NULL;
    }
    mat* c = construct(a->n_row, a->n_col);
    for(uint32_t i = 0; i < a->n_col; i++){
        for(uint32_t j = 0; j < a->n_row; j++){
            c->col[i]->e[j] = a->col[i]->e[j] + b->col[i]->e[j];
        }
    }
    return c;
}


/**
 * v+=w.
 */
__device__ void addeq(vec* v, const vec* w){
    if(v->n != w->n){
        printf("addeq v+=w not aligned!\n");
        return;
    }
    for(uint32_t i = 0; i < v->n; i++){
        v->e[i] += w->e[i];
    }
}


/**
 * a+=b.
 */
__device__ void addeq(mat* a, const mat* b){
    if(a->n_col != b->n_col || a->n_row != b->n_row){
        printf("addeq a+=b not aligned!\n");
        return;
    }
    for(uint32_t i = 0; i < a->n_col; i++){
        for(uint32_t j = 0; j < a->n_row; j++){
            a->col[i]->e[j] += b->col[i]->e[j];
        }
    }
}


/**
 * u = d*v.
 */
__device__ vec* mul(const vec* v, const double d){
    vec* u = construct(v->n);
    for(uint32_t i = 0; i < v->n; i++){
        u->e[i] = v->e[i] * d;
    }
    return u;
}


/**
 * c = d*a.
 */
__device__ mat* mul(const mat* a, const double d){
    mat* c = construct(a->n_row, a->n_col);
    for(uint32_t i = 0; i < a->n_col; i++){
        for(uint32_t j = 0; j < a->n_row; j++){
            c->col[i]->e[j] = a->col[i]->e[j] * d;
        }
    }
    return c;
}


/**
 * u = a*v
 */
__device__ vec* mul(const mat* a, const vec* v){
    if(a->n_col != v->n){
        printf("mul a*v not aligned!\n");
        return NULL;
    }
    vec* u = construct(a->n_row);
    for(uint32_t j = 0; j < a->n_row; j++){
        u->e[j] = 0;
    }
    for(uint32_t i = 0; i < a->n_col; i++){
        for(uint32_t j = 0; j < a->n_row; j++){
            u->e[j] += a->col[i]->e[j] * v->e[i];
        }
    }
    return u;
}


/**
 * u = a'*v (= (v'*a)')
 */
__device__ vec* mul(const vec* v, const mat* a){
    if(a->n_row != v->n){
        printf("mul a*v not aligned!\n");
        return NULL;
    }
    vec* u = construct(a->n_col);
    for(uint32_t i = 0; i < a->n_col; i++){
        u->e[i] = 0;
        for(uint32_t j = 0; j < a->n_row; j++){
            u->e[i] += a->col[i]->e[j] * v->e[j];
        }
    }
    return u;
}


/**
 * c = v*w'
 */
__device__ mat* mul(const vec* v, const vec* w){
    mat* c = construct(v->n, w->n);
    for(uint32_t i = 0; i < c->n_col; i++){
        for(uint32_t j = 0; j < c->n_row; j++){
            c->col[i]->e[j]  = v->e[j] * w->e[i];
        }
    }
    return c;
}


/**
 * v *= d.
 */
__device__ void muleq(vec* v, const double d){
    for(uint32_t i = 0; i < v->n; i++){
        v->e[i] *= d;
    }
}


/**
 * a *= d.
 */
__device__ void muleq(mat* a, const double d){
    mat* c = construct(a->n_row, a->n_col);
    for(uint32_t i = 0; i < a->n_col; i++){
        for(uint32_t j = 0; j < a->n_row; j++){
            a->col[i]->e[j] *= d;
        }
    }
}


/**
 * d = sum(v.*v).
 */
__device__ double inner(const vec* v){
    double d = 0;
    for(uint32_t i = 0; i < v->n; i++){
        d += v->e[i] * v->e[i];
    }
    return d;
}


/**
 * d = sum(v.*w).
 */
__device__ double inner(const vec* v, const vec* w){
    if(v->n != w->n){
        printf("inner v and w not aligned!\n");
        return 0.0/0;
    }
    double d = 0;
    for(uint32_t i = 0; i < v->n; i++){
        d += v->e[i] * w->e[i];
    }
    return d;
}


/**
 * d = sum(sum(a.*a)).
 */
__device__ double inner(const mat* a){
    double d = 0;
    for(uint32_t i = 0; i < a->n_col; i++){
        for(uint32_t j = 0; j < a->n_row; j++){
            d += a->col[i]->e[j] * a->col[i]->e[j];
        }
    }
    return d;
}


/**
 * d = sum(sum(a.*b)).
 */
__device__ double inner(const mat* a, const mat* b){
    if(a->n_col != b->n_col || a->n_row != b->n_row){
        printf("inner a and b not aligned!\n");
        return 0.0/0;
    }
    double d = 0;
    for(uint32_t i = 0; i < a->n_col; i++){
        for(uint32_t j = 0; j < a->n_row; j++){
            d += a->col[i]->e[j] * b->col[i]->e[j];
        }
    }
    return d;
}


/**
 * Add a new entry at the end.
 */
__device__ void insert(vec* v, const double d){
    v->n++;
    double* temp = (double*)malloc(v->n * sizeof(double));
    memcpy(temp, v->e, (v->n-1) * sizeof(double));
    free(v->e);
    v->e = temp;
    v->e[v->n-1] = d;
}


/**
 * Add a new row at the end.
 */
__device__ void insert_row(mat* a, const vec* v){
    if(a->n_col != v->n){
        printf("insert_row v to a not aligned!\n");
        return;
    }
    a->n_row++;
    double* temp;
    for(uint32_t i = 0; i < a->n_col; i++){
        a->col[i]->n++;
        temp = (double*)malloc(a->n_row * sizeof(double));
        memcpy(temp, a->col[i]->e, (a->n_row-1) * sizeof(double));
        free(a->col[i]->e);
        a->col[i]->e = temp;
        a->col[i]->e[a->n_row-1] = v->e[i];
    }
}


/**
 * Add a new column at the end.
 */
__device__ void insert_col(mat* a, const vec* v){
    if(a->n_row != v->n){
        printf("insert_row v to a not aligned!\n");
        return;
    }
    a->n_col++;
    vec** temp = (vec**)malloc(a->n_col * sizeof(vec*));
    memcpy(temp, a->col, (a->n_col-1) * sizeof(vec*));
    free(a->col);
    a->col = temp;
    a->col[a->n_col-1] = copy(v);
}


/**
 * Remove the last entry.
 */
__device__ void shed(vec* v){
    v->n--;
    double* temp = (double*)malloc(v->n * sizeof(double));
    memcpy(temp, v->e, v->n * sizeof(double));
    free(v->e);
    v->e = temp;
}


/**
 * Remove the last row.
 */
__device__ void shed_row(mat* a){
    a->n_row--;
    double* temp;
    for(uint32_t i = 0; i < a->n_col; i++){
        a->col[i]->n--;
        temp = (double*)malloc(a->n_row * sizeof(double));
        memcpy(temp, a->col[i]->e, a->n_row * sizeof(double));
        free(a->col[i]->e);
        a->col[i]->e = temp;
    }
}


/**
 * Remove the last column.
 */
__device__ void shed_col(mat* a){
    free(a->col[a->n_col]);
    a->n_col--;
    vec** temp = (vec**)malloc(a->n_col * sizeof(vec*));
    memcpy(temp, a->col, a->n_col * sizeof(vec*));
    free(a->col);
    a->col = temp;
}


/**
 * Swap two entries.
 */
__device__ void swap(vec* v, const uint32_t i, const uint32_t j){
    double temp;
    temp = v->e[i];
    v->e[i] = v->e[j];
    v->e[j] = temp;
}


/**
 * Swap two rows.
 */
__device__ void swap_row(mat* a, const uint32_t i, const uint32_t j){
    double temp;
    for(uint32_t k = 0; k < a->n_col; k++){
        temp = a->col[k]->e[i];
        a->col[k]->e[i] = a->col[k]->e[j];
        a->col[k]->e[j] = temp;
    }
}


/**
 * Swap two columns.
 */
__device__ void swap_col(mat* a, const uint32_t i, const uint32_t j){
    vec* temp;
    temp = a->col[i];
    a->col[i] = a->col[j];
    a->col[j] = temp;
}
