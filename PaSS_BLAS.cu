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
 * The vector structure
 */
struct pass_vec {
    uint32_t n; // the length
    float* e;  // the array of entries
};


/**
 * The matrix structure
 */
struct pass_mat {
    uint32_t n_row; // the number of rows
    uint32_t n_col; // the number of cols
    pass_vec** col; // the array of columns
};


/**
 * Construct a n-by-1 vector.
 */
__device__ pass_vec* pass_new(const uint32_t n) {
    pass_vec* v = (pass_vec*)malloc(sizeof(pass_vec));
    v->n = n;
    v->e = (float*)malloc(n * sizeof(float));
    return v;
}


/**
 * Construct a p-by-q matrix.
 */
__device__ pass_mat* pass_new(const uint32_t p, const uint32_t q) {
    pass_mat* a = (pass_mat*) malloc(sizeof(pass_mat));
    a->n_row = p;
    a->n_col = q;
    a->col = (pass_vec**)malloc(q * sizeof(pass_vec*));
    for(uint32_t i = 0; i < q; i++) {
        a->col[i] = (pass_vec*)malloc(sizeof(pass_vec));
        a->col[i]->n = p;
        a->col[i]->e = (float*)malloc(p * sizeof(float));
    }
    return a;
}


/**
 * Destruct the vector.
 */
__device__ void pass_free(pass_vec* v) {
    if(v == NULL) {
        printf("Null pointer counld not destruct.\n");
        return;
    }
    free(v->e);
    free(v);
}


/**
 * Destruct the matrix.
 */
__device__ void pass_free(pass_mat* a) {
    if(a == NULL) {
        printf("Null pointer counld not destruct.\n");
        return;
    }
    for(uint32_t i = 0; i < a->n_col; i++) {
        pass_free(a->col[i]);
    }
    free(a->col);
    free(a);
}


/**
 * Display the vector.
 */
__device__ void pass_print(const pass_vec* v) {
    if(v == NULL) {
        printf("Null pointer counld not print.\n");
        return;
    }
    for(uint32_t i = 0; i < v->n; i++) {
        printf("%8.3f\n", v->e[i]);
    }
    printf("\n");
}


/**
 * Display the matrix.
 */
__device__ void pass_print(const pass_mat* a) {
    if(a == NULL) {
        printf("Null pointer counld not print.\n");
        return;
    }
    for(uint32_t j = 0; j < a->n_row; j++) {
        for(uint32_t i = 0; i < a->n_col; i++) {
            printf("%8.3f", a->col[i]->e[j]);
        }
        printf("\n");
    }
    printf("\n");
}


/**
 * Copy the vector.
 */
__device__ pass_vec* pass_copy(const pass_vec* v) {
    if(v == NULL) {
        printf("Null pointer counld not copy.\n");
        return NULL;
    }
    pass_vec* u = pass_new(v->n);
    memcpy(u->e, v->e, v->n * sizeof(float));
    return u;
}


/**
 * Copy the matrix.
 */
__device__ pass_mat* pass_copy(const pass_mat* a) {
    if(a == NULL) {
        printf("Null pointer counld not copy.\n");
        return NULL;
    }
    pass_mat* c = pass_new(a->n_row, a->n_col);
    for(uint32_t i = 0; i < a->n_col; i++) {
        memcpy(c->col[i], a->col[i], a->n_row * sizeof(float));
    }
    return c;
}


/**
 * u = v+w.
 */
__device__ pass_vec* pass_add(const pass_vec* v, const pass_vec* w) {
    if(v == NULL || w == NULL) {
        printf("Null pointer counld not add.\n");
        return NULL;
    }
    if(v->n != w->n) {
        printf("add for vectors not aligned!\n");
        return NULL;
    }
    pass_vec* u = pass_new(v->n);
    for(uint32_t i = 0; i < v->n; i++) {
        u->e[i] = v->e[i] + w->e[i];
    }
    return u;
}


/**
 * c = a+b.
 */
__device__ pass_mat* pass_add(const pass_mat* a, const pass_mat* b) {
    if(a == NULL || b == NULL) {
        printf("Null pointer counld not add.\n");
        return NULL;
    }
    if(a->n_col != b->n_col || a->n_row != b->n_row) {
        printf("add for matrices not aligned!\n");
        return NULL;
    }
    pass_mat* c = pass_new(a->n_row, a->n_col);
    for(uint32_t i = 0; i < a->n_col; i++) {
        for(uint32_t j = 0; j < a->n_row; j++) {
            c->col[i]->e[j] = a->col[i]->e[j] + b->col[i]->e[j];
        }
    }
    return c;
}


/**
 * v+=w.
 */
__device__ void pass_addeq(pass_vec* v, const pass_vec* w) {
    if(v == NULL || w == NULL) {
        printf("Null pointer counld not addeq.\n");
        return;
    }
    if(v->n != w->n) {
        printf("addeq for vectors not aligned!\n");
        return;
    }
    for(uint32_t i = 0; i < v->n; i++) {
        v->e[i] += w->e[i];
    }
}


/**
 * a+=b.
 */
__device__ void pass_addeq(pass_mat* a, const pass_mat* b) {
    if(a == NULL || b == NULL) {
        printf("Null pointer counld not addeq.\n");
        return;
    }
    if(a->n_col != b->n_col || a->n_row != b->n_row) {
        printf("addeq for matrices not aligned!\n");
        return;
    }
    for(uint32_t i = 0; i < a->n_col; i++) {
        for(uint32_t j = 0; j < a->n_row; j++) {
            a->col[i]->e[j] += b->col[i]->e[j];
        }
    }
}


/**
 * u = d*v.
 */
__device__ pass_vec* pass_mul(const pass_vec* v, const float d) {
    if(v == NULL) {
        printf("Null pointer counld not mul.\n");
        return NULL;
    }
    pass_vec* u = pass_new(v->n);
    for(uint32_t i = 0; i < v->n; i++) {
        u->e[i] = v->e[i] * d;
    }
    return u;
}


/**
 * c = d*a.
 */
__device__ pass_mat* pass_mul(const pass_mat* a, const float d) {
    if(a == NULL) {
        printf("Null pointer counld not mul.\n");
        return NULL;
    }
    pass_mat* c = pass_new(a->n_row, a->n_col);
    for(uint32_t i = 0; i < a->n_col; i++) {
        for(uint32_t j = 0; j < a->n_row; j++) {
            c->col[i]->e[j] = a->col[i]->e[j] * d;
        }
    }
    return c;
}


/**
 * u = a*v
 */
__device__ pass_vec* pass_mul(const pass_mat* a, const pass_vec* v) {
    if(a == NULL || v == NULL) {
        printf("Null pointer counld not mul.\n");
        return NULL;
    }
    if(a->n_col != v->n) {
        printf("mul for matrix multiply vector not aligned!\n");
        return NULL;
    }
    pass_vec* u = pass_new(a->n_row);
    for(uint32_t j = 0; j < a->n_row; j++) {
        u->e[j] = 0;
    }
    for(uint32_t i = 0; i < a->n_col; i++) {
        for(uint32_t j = 0; j < a->n_row; j++) {
            u->e[j] += a->col[i]->e[j] * v->e[i];
        }
    }
    return u;
}


/**
 * u = a'*v (= (v'*a)')
 */
__device__ pass_vec* pass_mul(const pass_vec* v, const pass_mat* a) {
    if(a == NULL || v == NULL) {
        printf("Null pointer counld not mul.\n");
        return NULL;
    }
    if(a->n_row != v->n) {
        printf("mul for vector multiply matrix not aligned!\n");
        return NULL;
    }
    pass_vec* u = pass_new(a->n_col);
    for(uint32_t i = 0; i < a->n_col; i++) {
        u->e[i] = 0;
        for(uint32_t j = 0; j < a->n_row; j++) {
            u->e[i] += a->col[i]->e[j] * v->e[j];
        }
    }
    return u;
}


/**
 * c = v*w'
 */
__device__ pass_mat* pass_mul(const pass_vec* v, const pass_vec* w) {
    if(v == NULL || w == NULL) {
        printf("Null pointer counld not mul.\n");
        return NULL;
    }
    pass_mat* c = pass_new(v->n, w->n);
    for(uint32_t i = 0; i < c->n_col; i++) {
        for(uint32_t j = 0; j < c->n_row; j++) {
            c->col[i]->e[j]  = v->e[j] * w->e[i];
        }
    }
    return c;
}


/**
 * v *= d.
 */
__device__ void pass_muleq(pass_vec* v, const float d) {
    if(v == NULL) {
        printf("Null pointer counld not muleq.\n");
        return;
    }
    for(uint32_t i = 0; i < v->n; i++) {
        v->e[i] *= d;
    }
}


/**
 * a *= d.
 */
__device__ void pass_muleq(pass_mat* a, const float d) {
    if(a == NULL) {
        printf("Null pointer counld not muleq.\n");
        return;
    }
    pass_mat* c = pass_new(a->n_row, a->n_col);
    for(uint32_t i = 0; i < a->n_col; i++) {
        for(uint32_t j = 0; j < a->n_row; j++) {
            a->col[i]->e[j] *= d;
        }
    }
}


/**
 * d = sum(v.*v).
 */
__device__ float pass_inner(const pass_vec* v) {
    if(v == NULL) {
        printf("Null pointer counld not inner.\n");
        return 0.0/0;
    }
    float d = 0;
    for(uint32_t i = 0; i < v->n; i++) {
        d += v->e[i] * v->e[i];
    }
    return d;
}


/**
 * d = sum(v.*w).
 */
__device__ float pass_inner(const pass_vec* v, const pass_vec* w) {
    if(v == NULL || w == NULL) {
        printf("Null pointer counld not inner.\n");
        return 0.0/0;
    }
    if(v->n != w->n) {
        printf("inner for vectors not aligned!\n");
        return 0.0/0;
    }
    float d = 0;
    for(uint32_t i = 0; i < v->n; i++) {
        d += v->e[i] * w->e[i];
    }
    return d;
}


/**
 * d = sum(sum(a.*a)).
 */
__device__ float pass_inner(const pass_mat* a) {
    if(a == NULL) {
        printf("Null pointer counld not inner.\n");
        return 0.0/0;
    }
    float d = 0;
    for(uint32_t i = 0; i < a->n_col; i++) {
        for(uint32_t j = 0; j < a->n_row; j++) {
            d += a->col[i]->e[j] * a->col[i]->e[j];
        }
    }
    return d;
}


/**
 * d = sum(sum(a.*b)).
 */
__device__ float pass_inner(const pass_mat* a, const pass_mat* b) {
    if(a == NULL || b == NULL) {
        printf("Null pointer counld not inner.\n");
        return 0.0/0;
    }
    if(a->n_col != b->n_col || a->n_row != b->n_row) {
        printf("inner for matrices not aligned!\n");
        return 0.0/0;
    }
    float d = 0;
    for(uint32_t i = 0; i < a->n_col; i++) {
        for(uint32_t j = 0; j < a->n_row; j++) {
            d += a->col[i]->e[j] * b->col[i]->e[j];
        }
    }
    return d;
}


/**
 * Add a new entry at the end.
 */
__device__ void pass_insert(pass_vec* v, const float d) {
    if(v == NULL) {
        printf("Null pointer counld not insert.\n");
        return;
    }
    v->n++;
    float* temp = (float*)malloc(v->n * sizeof(float));
    memcpy(temp, v->e, (v->n-1) * sizeof(float));
    free(v->e);
    v->e = temp;
    v->e[v->n-1] = d;
}


/**
 * Add a new row at the end.
 */
__device__ void pass_insert_row(pass_mat* a, const pass_vec* v) {
    if(a == NULL || v == NULL) {
        printf("Null pointer counld not insert_row.\n");
        return;
    }
    if(a->n_col != v->n) {
        printf("insert_row not aligned!\n");
        return;
    }
    a->n_row++;
    float* temp;
    for(uint32_t i = 0; i < a->n_col; i++) {
        a->col[i]->n++;
        temp = (float*)malloc(a->n_row * sizeof(float));
        memcpy(temp, a->col[i]->e, (a->n_row-1) * sizeof(float));
        free(a->col[i]->e);
        a->col[i]->e = temp;
        a->col[i]->e[a->n_row-1] = v->e[i];
    }
}


/**
 * Add a new column at the end.
 */
__device__ void pass_insert_col(pass_mat* a, const pass_vec* v) {
    if(a == NULL || v == NULL) {
        printf("Null pointer counld not insert_col.\n");
        return;
    }
    if(a->n_row != v->n) {
        printf("insert_col not aligned!\n");
        return;
    }
    a->n_col++;
    pass_vec** temp = (pass_vec**)malloc(a->n_col * sizeof(pass_vec*));
    memcpy(temp, a->col, (a->n_col-1) * sizeof(pass_vec*));
    free(a->col);
    a->col = temp;
    a->col[a->n_col-1] = pass_copy(v);
}


/**
 * Remove the last entry.
 */
__device__ void pass_shed(pass_vec* v) {
    if(v == NULL) {
        printf("Null pointer counld not shed.\n");
        return;
    }
    v->n--;
    float* temp = (float*)malloc(v->n * sizeof(float));
    memcpy(temp, v->e, v->n * sizeof(float));
    free(v->e);
    v->e = temp;
}


/**
 * Remove the last row.
 */
__device__ void pass_shed_row(pass_mat* a) {
    if(a == NULL) {
        printf("Null pointer counld not shed_row.\n");
        return;
    }
    a->n_row--;
    float* temp;
    for(uint32_t i = 0; i < a->n_col; i++) {
        a->col[i]->n--;
        temp = (float*)malloc(a->n_row * sizeof(float));
        memcpy(temp, a->col[i]->e, a->n_row * sizeof(float));
        free(a->col[i]->e);
        a->col[i]->e = temp;
    }
}


/**
 * Remove the last column.
 */
__device__ void pass_shed_col(pass_mat* a) {
    if(a == NULL) {
        printf("Null pointer counld not shed_col.\n");
        return;
    }
    a->n_col--;
    free(a->col[a->n_col]);
    pass_vec** temp = (pass_vec**)malloc(a->n_col * sizeof(pass_vec*));
    memcpy(temp, a->col, a->n_col * sizeof(pass_vec*));
    free(a->col);
    a->col = temp;
}


/**
 * Swap two entries.
 */
__device__ void pass_swap(pass_vec* v, const uint32_t i, const uint32_t j) {
    if(v == NULL) {
        printf("Null pointer counld not swap.\n");
        return;
    }
    float temp;
    temp = v->e[i];
    v->e[i] = v->e[j];
    v->e[j] = temp;
}


/**
 * Swap two rows.
 */
__device__ void pass_swap_row(pass_mat* a, const uint32_t i, const uint32_t j) {
    if(a == NULL) {
        printf("Null pointer counld not swap_row.\n");
        return;
    }
    float temp;
    for(uint32_t k = 0; k < a->n_col; k++) {
        temp = a->col[k]->e[i];
        a->col[k]->e[i] = a->col[k]->e[j];
        a->col[k]->e[j] = temp;
    }
}


/**
 * Swap two columns.
 */
__device__ void pass_swap_col(pass_mat* a, const uint32_t i, const uint32_t j) {
    if(a == NULL) {
        printf("Null pointer counld not swap_col.\n");
        return;
    }
    pass_vec* temp;
    temp = a->col[i];
    a->col[i] = a->col[j];
    a->col[j] = temp;
}
