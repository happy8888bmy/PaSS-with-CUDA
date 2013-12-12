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
 *
 * @param n the length of the vector.
 * @return the pointer of new vector.
 */
__device__ pass_vec* pass_new(const uint32_t n) {
    pass_vec* v = (pass_vec*)malloc(sizeof(pass_vec));
    v->n = n;
    v->e = (float*)malloc(n * sizeof(float));
    return v;
}


/**
 * Construct a p-by-q matrix.
 *
 * @param p the numbers of rows.
 * @param q the numbers of columns.
 * @return the pointer of new matrix.
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
 *
 * @param v the vector.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_free(pass_vec* v) {
    free(v->e);
    free(v);
    return true;
}


/**
 * Destruct the matrix.
 *
 * @param a the matrix.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_free(pass_mat* a) {
    for(uint32_t i = 0; i < a->n_col; i++) {
        pass_free(a->col[i]);
    }
    free(a->col);
    free(a);
    return true;
}


/**
 * Display the vector.
 *
 * @param v the vector.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_print(const pass_vec* v) {
    for(uint32_t i = 0; i < v->n; i++) {
        printf("%8.3f\n", v->e[i]);
    }
    printf("\n");
    return true;
}


/**
 * Display the matrix.
 *
 * @param a the vector.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_print(const pass_mat* a) {
    for(uint32_t j = 0; j < a->n_row; j++) {
        for(uint32_t i = 0; i < a->n_col; i++) {
            printf("%8.3f", a->col[i]->e[j]);
        }
        printf("\n");
    }
    printf("\n");
    return true;
}


/**
 * u = v
 *
 * @param u the new vector.
 * @param v the original vector.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_copy(pass_vec* u, const pass_vec* v) {
    memcpy(u->e, v->e, v->n * sizeof(float));
    return true;
}


/**
 * c = a
 *
 * @param u the new vector.
 * @param v the original vector.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_copy(pass_mat* c, const pass_mat* a) {
    for(uint32_t i = 0; i < a->n_col; i++) {
        memcpy(c->col[i], a->col[i], a->n_row * sizeof(float));
    }
    return true;
}


/**
 * u = v+w.
 *
 * @param u the sum vector.
 * @param v the augend vector.
 * @param w the addend vector.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_add(pass_vec* u, const pass_vec* v, const pass_vec* w) {
    if(u->n != v->n || u->n != w->n) {
        printf("(pass_add: vector) not aligned!\n");
        return false;
    }
    for(uint32_t i = 0; i < v->n; i++) {
        u->e[i] = v->e[i] + w->e[i];
    }
    return true;
}


/**
 * c = a+b.
 *
 * @param b the sum matrix.
 * @param c the augend matrix.
 * @param a the addend matrix.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_add(pass_mat* c, const pass_mat* a, const pass_mat* b) {
    if(c->n_col != a->n_col || c->n_col != b->n_col || c->n_row != a->n_row || c->n_row != b->n_row) {
        printf("(pass_add: matrix) not aligned!\n");
        return false;
    }
    for(uint32_t i = 0; i < a->n_col; i++) {
        for(uint32_t j = 0; j < a->n_row; j++) {
            c->col[i]->e[j] = a->col[i]->e[j] + b->col[i]->e[j];
        }
    }
    return true;
}


/**
 * u = d*v.
 *
 * @param u the product vector.
 * @param v the multiplier vector.
 * @param d the multiplicand number.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_mul(pass_vec* u, const pass_vec* v, const float d) {
    for(uint32_t i = 0; i < v->n; i++) {
        u->e[i] = v->e[i] * d;
    }
    return true;
}


/**
 * c = d*a.
 *
 * @param c the product matrix.
 * @param a the multiplier vector.
 * @param d the multiplicand number.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_mul(pass_mat* c, const pass_mat* a, const float d) {
    for(uint32_t i = 0; i < a->n_col; i++) {
        for(uint32_t j = 0; j < a->n_row; j++) {
            c->col[i]->e[j] = a->col[i]->e[j] * d;
        }
    }
    return true;
}


/**
 * u = a*v
 *
 * @param u the product vector.
 * @param a the multiplicand matrix.
 * @param v the multiplier vector.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_mul(pass_vec* u, const pass_mat* a, const pass_vec* v) {
    if(u->n != v->n || u->n != a->n_row) {
        printf("(pass_mul: matrix left) not aligned!\n");
        return false;
    }
    for(uint32_t j = 0; j < a->n_row; j++) {
        u->e[j] = 0;
    }
    for(uint32_t i = 0; i < a->n_col; i++) {
        for(uint32_t j = 0; j < a->n_row; j++) {
            u->e[j] += a->col[i]->e[j] * v->e[i];
        }
    }
    return true;
}


/**
 * u' = v'*a (u = a'*v)
 *
 * @param u the product vector.
 * @param v the multiplicand vector.
 * @param a the multiplier matrix.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_mul(pass_vec* u, const pass_vec* v, const pass_mat* a) {
    if(u->n != v->n || u->n != a->n_row) {
        printf("(pass_mul: matrix right) not aligned!\n");
        return false;
    }
    for(uint32_t i = 0; i < a->n_col; i++) {
        u->e[i] = 0;
        for(uint32_t j = 0; j < a->n_row; j++) {
            u->e[i] += a->col[i]->e[j] * v->e[j];
        }
    }
    return true;
}


/**
 * c = v*w'
 *
 * @param c the product matrix.
 * @param v the multiplicand vector.
 * @param w the multiplier vector.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_mul(pass_mat c, const pass_vec* v, const pass_vec* w) {
    for(uint32_t i = 0; i < c->n_col; i++) {
        for(uint32_t j = 0; j < c->n_row; j++) {
            c->col[i]->e[j]  = v->e[j] * w->e[i];
        }
    }
    return true;
}


/**
 * d = sum(v.*v).
 *
 * @param d the product number.
 * @param v the vector.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_inner(uint32_t* d, const pass_vec* v) {
    *d = 0;
    for(uint32_t i = 0; i < v->n; i++) {
        *d += v->e[i] * v->e[i];
    }
    return true;
}


/**
 * d = sum(v.*w).
 *
 * @param v the multiplicand vector.
 * @param w the multiplier vector.
 * @param d the product number.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_inner(uint32_t* d, const pass_vec* v, const pass_vec* w) {
    if(v->n != w->n) {
        printf("(pass_inner: vector) not aligned!\n");
        return false;
    }
    *d = 0;
    for(uint32_t i = 0; i < v->n; i++) {
        *d += v->e[i] * w->e[i];
    }
    return true;
}


/**
 * d = sum(sum(a.*a)).
 *
 * @param a the matrix.
 * @param d the product number.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_inner(uint32_t* d, const pass_mat* a) {
    *d = 0;
    for(uint32_t i = 0; i < a->n_col; i++) {
        for(uint32_t j = 0; j < a->n_row; j++) {
            *d += a->col[i]->e[j] * a->col[i]->e[j];
        }
    }
    return true;
}


/**
 * d = sum(sum(a.*b)).
 *
 * @param a the multiplicand matrix.
 * @param b the multiplier matrix.
 * @param d the product number.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_inner(uint32_t* d, const pass_mat* a, const pass_mat* b) {
    if(a->n_col != b->n_col || a->n_row != b->n_row) {
        printf("(pass_inner: matrix) not aligned!\n");
        return false;
    }
    *d = 0;
    for(uint32_t i = 0; i < a->n_col; i++) {
        for(uint32_t j = 0; j < a->n_row; j++) {
            *d += a->col[i]->e[j] * b->col[i]->e[j];
        }
    }
    return true;
}


/**
 * Add a new entry at the end.
 *
 * @param v the vector.
 * @param d the new entry.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_insert(pass_vec* v, const float d) {
    v->n++;
    float* temp = (float*)malloc(v->n * sizeof(float));
    memcpy(temp, v->e, (v->n-1) * sizeof(float));
    free(v->e);
    v->e = temp;
    v->e[v->n-1] = d;
    return true;
}


/**
 * Add a new row at the end.
 *
 * @param a the matrix.
 * @param v the new vector.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_insert_row(pass_mat* a, const pass_vec* v) {
    if(a->n_col != v->n) {
        printf("(insert_row) not aligned!\n");
        return false;
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
    return true;
}


/**
 * Add a new column at the end.
 *
 * @param a the matrix.
 * @param v the new vector.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_insert_col(pass_mat* a, const pass_vec* v) {
    if(a->n_row != v->n) {
        printf("(insert_col) not aligned!\n");
        return false;
    }
    a->n_col++;
    pass_vec** temp = (pass_vec**)malloc(a->n_col * sizeof(pass_vec*));
    memcpy(temp, a->col, (a->n_col-1) * sizeof(pass_vec*));
    free(a->col);
    a->col = temp;
    pass_copy(a->col[a->n_col-1], v);
    return true;
}


/**
 * Remove the last entry.
 *
 * @param v the vector.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_shed(pass_vec* v) {
    v->n--;
    float* temp = (float*)malloc(v->n * sizeof(float));
    memcpy(temp, v->e, v->n * sizeof(float));
    free(v->e);
    v->e = temp;
    return true;
}


/**
 * Remove the last row.
 *
 * @param a the matrix.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_shed_row(pass_mat* a) {
    a->n_row--;
    float* temp;
    for(uint32_t i = 0; i < a->n_col; i++) {
        a->col[i]->n--;
        temp = (float*)malloc(a->n_row * sizeof(float));
        memcpy(temp, a->col[i]->e, a->n_row * sizeof(float));
        free(a->col[i]->e);
        a->col[i]->e = temp;
    }
    return true;
}


/**
 * Remove the last column.
 *
 * @param a the matrix.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_shed_col(pass_mat* a) {
    a->n_col--;
    free(a->col[a->n_col]);
    pass_vec** temp = (pass_vec**)malloc(a->n_col * sizeof(pass_vec*));
    memcpy(temp, a->col, a->n_col * sizeof(pass_vec*));
    free(a->col);
    a->col = temp;
    return true;
}


/**
 * Swap two entries.
 *
 * @param v the vector.
 * @param i the index of first entry.
 * @param j the index of second entry.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_swap(pass_vec* v, const uint32_t i, const uint32_t j) {
    float temp;
    temp = v->e[i];
    v->e[i] = v->e[j];
    v->e[j] = temp;
    return true;
}


/**
 * Swap two rows.
 *
 * @param a the matrix.
 * @param i the index of first row.
 * @param j the index of second row.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_swap_row(pass_mat* a, const uint32_t i, const uint32_t j) {
    float temp;
    for(uint32_t k = 0; k < a->n_col; k++) {
        temp = a->col[k]->e[i];
        a->col[k]->e[i] = a->col[k]->e[j];
        a->col[k]->e[j] = temp;
    }
    return true;
}


/**
 * Swap two columns.
 *
 * @param a the matrix.
 * @param i the index of first column.
 * @param j the index of second column.
 * @return whether this function has been excuted successfully.
 */
__device__ bool pass_swap_col(pass_mat* a, const uint32_t i, const uint32_t j) {
    pass_vec* temp;
    temp = a->col[i];
    a->col[i] = a->col[j];
    a->col[j] = temp;
    return true;
}
