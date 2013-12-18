/**
 * BLAS.cu
 * The basic linear algebra subprograms for PaSS
 *
 * @author emfo
 */

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "curand_kernel.h"
#include "device_functions.h"
#include "math_constants.h"
#include "math_functions.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef uint32_t u32;

/**
 * The PaSS namespace
 */
namespace pass {
	/**
	 * The vector structure
	 */
	struct vec {
		u32 n;    // the length
		float* e; // the array of entries

		__host__ __device__ vec(const u32 n) {
			this->n = n;
			this->e = (float*)malloc(n * sizeof(float));
		}

		__host__ __device__ vec(const u32 n, const float d) {
			this->n = n;
			this->e = (float*)malloc(n * sizeof(float));
			for(u32 i = 0; i < n; i++) {
				this->e[i] = d;
			}
		}

		__host__ __device__ ~vec() {
			free(this->e);
		}
	};


	/**
	 * The matrix structure
	 */
	struct mat {
		u32 n_row; // the number of rows
		u32 n_col; // the number of cols
		vec** col; // the array of columns

		__host__ __device__ mat(const u32 p, const u32 q) {
			this->n_row = p;
			this->n_col = q;
			this->col = (vec**)malloc(q * sizeof(vec*));
			for(u32 i = 0; i < q; i++) {
				this->col[i] = new vec(p);
			}
		}

		__host__ __device__ mat(const u32 p, const u32 q, const float d) {
			this->n_row = p;
			this->n_col = q;
			this->col = (vec**)malloc(q * sizeof(vec*));
			for(u32 i = 0; i < q; i++) {
				this->col[i] = new vec(p, d);
			}
		}
		
		__host__ __device__ ~mat() {
			for(u32 i = 0; i < this->n_col; i++) {
				delete this->col[i];
			}
			free(this->col);
		}
	};


	/**
	 * The index structure
	 */
	struct idx {
		u32 n;  // the length
		u32* e; // the array of entries

		__host__ __device__ idx(const u32 n) {
			this->n = n;
			this->e = (u32*)malloc(n * sizeof(u32));
		}

		__host__ __device__ idx(const u32 n, const u32 d) {
			this->n = n;
			this->e = (u32*)malloc(n * sizeof(u32));
			for(u32 i = 0; i < n; i++) {
				this->e[i] = d;
			}
		}

		__host__ __device__ ~idx() {
			free(this->e);
		}
	};


	/**
	 * Display the vector.
	 *
	 * @param v the vector.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool print(const vec* v) {
		for(u32 i = 0; i < v->n; i++) {
			printf("%8.3f  ", v->e[i]);
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
	__host__ __device__ bool print(const mat* a) {
		for(u32 j = 0; j < a->n_row; j++) {
			for(u32 i = 0; i < a->n_col; i++) {
				printf("%8.3f  ", a->col[i]->e[j]);
			}
			printf("\n");
		}
		printf("\n");
		return true;
	}


	/**
	 * Display the index.
	 *
	 * @param x the index.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool print(const idx* x) {
		for(u32 i = 0; i < x->n; i++) {
			printf("%4u ", x->e[i]);
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
	__host__ __device__ bool copy(vec* u, const vec* v) {
		if(u->n != v->n) {
			printf("(copy: vector) not aligned!\n");
			return false;
		}
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
	__host__ __device__ bool copy(mat* c, const mat* a) {
		if(c->n_col != a->n_col || c->n_row != a->n_row) {
			printf("(copy: matrix) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < a->n_col; i++) {
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
	__host__ __device__ bool add(vec* u, const vec* v, const vec* w) {
		if(u->n != v->n || u->n != w->n) {
			printf("(add: vector) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < v->n; i++) {
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
	__host__ __device__ bool add(mat* c, const mat* a, const mat* b) {
		if(c->n_col != a->n_col || c->n_col != b->n_col || c->n_row != a->n_row || c->n_row != b->n_row) {
			printf("(add: matrix) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < a->n_col; i++) {
			for(u32 j = 0; j < a->n_row; j++) {
				c->col[i]->e[j] = a->col[i]->e[j] + b->col[i]->e[j];
			}
		}
		return true;
	}


	/**
	 * u = v+w.
	 *
	 * @param u the difference vector.
	 * @param v the minuend vector.
	 * @param w the subtrahend vector.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool sub(vec* u, const vec* v, const vec* w) {
		if(u->n != v->n || u->n != w->n) {
			printf("(sub: vector) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < v->n; i++) {
			u->e[i] = v->e[i] - w->e[i];
		}
		return true;
	}


	/**
	 * c = a+b.
	 *
	 * @param b the difference matrix.
	 * @param c the minuend matrix.
	 * @param a the subtrahend matrix.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool sub(mat* c, const mat* a, const mat* b) {
		if(c->n_col != a->n_col || c->n_col != b->n_col || c->n_row != a->n_row || c->n_row != b->n_row) {
			printf("(sub: matrix) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < a->n_col; i++) {
			for(u32 j = 0; j < a->n_row; j++) {
				c->col[i]->e[j] = a->col[i]->e[j] - b->col[i]->e[j];
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
	__host__ __device__ bool mul(vec* u, const vec* v, const float d) {
		for(u32 i = 0; i < v->n; i++) {
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
	__host__ __device__ bool mul(mat* c, const mat* a, const float d) {
		for(u32 i = 0; i < a->n_col; i++) {
			for(u32 j = 0; j < a->n_row; j++) {
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
	__host__ __device__ bool mul(vec* u, const mat* a, const vec* v) {
		if(u->n != a->n_row || v->n != a->n_col) {
			printf("(mul: matrix left) not aligned!\n");
			return false;
		}
		for(u32 j = 0; j < a->n_row; j++) {
			u->e[j] = 0;
		}
		for(u32 i = 0; i < a->n_col; i++) {
			for(u32 j = 0; j < a->n_row; j++) {
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
	__host__ __device__ bool mul(vec* u, const vec* v, const mat* a) {
		if(u->n != a->n_col || v->n != a->n_row) {
			printf("(mul: matrix right) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < a->n_col; i++) {
			u->e[i] = 0;
			for(u32 j = 0; j < a->n_row; j++) {
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
	__host__ __device__ bool mul(mat* c, const vec* v, const vec* w) {
		for(u32 i = 0; i < c->n_col; i++) {
			for(u32 j = 0; j < c->n_row; j++) {
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
	__host__ __device__ bool inner(float* d, const vec* v) {
		*d = 0;
		for(u32 i = 0; i < v->n; i++) {
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
	__host__ __device__ bool inner(float* d, const vec* v, const vec* w) {
		if(v->n != w->n) {
			printf("(inner: vector) not aligned!\n");
			return false;
		}
		*d = 0;
		for(u32 i = 0; i < v->n; i++) {
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
	__host__ __device__ bool inner(float* d, const mat* a) {
		*d = 0;
		for(u32 i = 0; i < a->n_col; i++) {
			for(u32 j = 0; j < a->n_row; j++) {
				*d += a->col[i]->e[j] * a->col[i]->e[j];
			}
		}
		return true;
	}


	/**
	 * u' = sum(a.*b).
	 *
	 * @param a the multiplicand matrix.
	 * @param b the multiplier matrix.
	 * @param u the product vector.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool inner(vec* u, const mat* a, const mat* b) {
		if(u->n != a->n_col || a->n_col != b->n_col || a->n_row != b->n_row) {
			printf("(inner: matrix) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < a->n_col; i++) {
			u->e[i] = 0;
			for(u32 j = 0; j < a->n_row; j++) {
				u->e[i] += a->col[i]->e[j] * b->col[i]->e[j];
			}
		}
		return true;
	}


	/**
	 * d = norm(v, 2).
	 *
	 * @param d the Euclidean norm.
	 * @param v the vector.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool norm(float* d, const vec* v) {
		inner(d, v);
		*d = sqrt(*d);
		return true;
	}


	/**
	 * Add a new entry at the end.
	 *
	 * @param v the vector.
	 * @param d the new entry.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool insert(vec* v, const float d) {
		v->n++;
		float* temp = (float*)malloc(v->n * sizeof(float));
		memcpy(temp, v->e, (v->n-1) * sizeof(float));
		free(v->e);
		v->e = temp;
		v->e[v->n-1] = d;
		return true;
	}


	/**
	 * Add a new entry at the end.
	 *
	 * @param x the index.
	 * @param i the new entry.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool insert(idx* x, const u32 i) {
		x->n++;
		u32* temp = (u32*)malloc(x->n * sizeof(u32));
		memcpy(temp, x->e, (x->n-1) * sizeof(u32));
		free(x->e);
		x->e = temp;
		x->e[x->n-1] = i;
		return true;
	}


	/**
	 * Add a new row at the end.
	 *
	 * @param a the matrix.
	 * @param v the new vector.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool insert_row(mat* a, const vec* v) {
		if(a->n_col != v->n) {
			printf("(insert_row) not aligned!\n");
			return false;
		}
		a->n_row++;
		float* temp;
		for(u32 i = 0; i < a->n_col; i++) {
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
	__host__ __device__ bool insert_col(mat* a, const vec* v) {
		if(a->n_row != v->n) {
			printf("(insert_col) not aligned!\n");
			return false;
		}
		a->n_col++;
		vec** temp = (vec**)malloc(a->n_col * sizeof(vec*));
		memcpy(temp, a->col, (a->n_col-1) * sizeof(vec*));
		free(a->col);
		a->col = temp;
		a->col[a->n_col-1] = new vec(v->n);
		copy(a->col[a->n_col-1], v);
		return true;
	}


	/**
	 * Add a new row and a new column at the end.
	 *
	 * @param a the matrix.
	 * @param d the number.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool insert(mat* a, const float d) {
		a->n_row++;
		a->n_col++;
		float* ftemp;
		for(u32 i = 0; i < a->n_col-1; i++) {
			a->col[i]->n++;
			ftemp = (float*)malloc(a->n_row * sizeof(float));
			memcpy(ftemp, a->col[i]->e, (a->n_row-1) * sizeof(float));
			free(a->col[i]->e);
			a->col[i]->e = ftemp;
			a->col[i]->e[a->n_row-1] = d;
		}
		vec** temp = (vec**)malloc(a->n_col * sizeof(vec*));
		memcpy(temp, a->col, (a->n_col-1) * sizeof(vec*));
		free(a->col);
		a->col = temp;
		a->col[a->n_col-1] = new vec(a->n_row, d);
		return true;
	}


	/**
	 * Remove the last entry.
	 *
	 * @param v the vector.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool shed(vec* v) {
		v->n--;
		float* temp = (float*)malloc(v->n * sizeof(float));
		memcpy(temp, v->e, v->n * sizeof(float));
		free(v->e);
		v->e = temp;
		return true;
	}


	/**
	 * Remove the last entry.
	 *
	 * @param x the index.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool shed(idx* x) {
		x->n--;
		u32* temp = (u32*)malloc(x->n * sizeof(u32));
		memcpy(temp, x->e, x->n * sizeof(u32));
		free(x->e);
		x->e = temp;
		return true;
	}


	/**
	 * Remove the last row.
	 *
	 * @param a the matrix.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool shed_row(mat* a) {
		a->n_row--;
		float* temp;
		for(u32 i = 0; i < a->n_col; i++) {
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
	__host__ __device__ bool shed_col(mat* a) {
		a->n_col--;
		free(a->col[a->n_col]);
		vec** temp = (vec**)malloc(a->n_col * sizeof(vec*));
		memcpy(temp, a->col, a->n_col * sizeof(vec*));
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
	__host__ __device__ bool swap(vec* v, const u32 i, const u32 j) {
		float temp;
		temp = v->e[i];
		v->e[i] = v->e[j];
		v->e[j] = temp;
		return true;
	}


	/**
	 * Swap two entries.
	 *
	 * @param X the index.
	 * @param i the index of first entry.
	 * @param j the index of second entry.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool swap(idx* x, const u32 i, const u32 j) {
		u32 temp;
		temp = x->e[i];
		x->e[i] = x->e[j];
		x->e[j] = temp;
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
	__host__ __device__ bool swap_row(mat* a, const u32 i, const u32 j) {
		float temp;
		for(u32 k = 0; k < a->n_col; k++) {
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
	__host__ __device__ bool swap_col(mat* a, const u32 i, const u32 j) {
		vec* temp;
		temp = a->col[i];
		a->col[i] = a->col[j];
		a->col[j] = temp;
		return true;
	}


	/**
	 * Find the index of target element
	 *
	 * @param k the index.
	 * @param x the vector.
	 * @param d the element.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool find_index(u32* k, const idx* v, const u32 i) {
		for(u32 j = 0; j < v->n; j++) {
			if(v->e[j] == i) {
				*k = j;
				return true;
			}
		}
		*k = UINT32_MAX;
		return false;
	}


	/**
	 * Sort a index in ascending order
	 *
	 * @param x the index.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool sort_ascend(idx* x) {
		u32 temp;
		for(u32 i = x->n-1; i > 0; i--) {
			for(u32 j = 0; j < i; j++) {
				if(x->e[j] > x->e[j+1]) {
					temp = x->e[j];
					x->e[j] = x->e[j+1];
					x->e[j+1] = temp;
				}
			}
		}
		return true;
	}


	/**
	 * Sort a index in descending order
	 *
	 * @param x the index.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool sort_descend(idx* x) {
		u32 temp;
		for(u32 i = x->n-1; i > 0; i--) {
			for(u32 j = 0; j < i; j++) {
				if(x->e[j] < x->e[j+1]) {
					temp = x->e[j];
					x->e[j] = x->e[j+1];
					x->e[j+1] = temp;
				}
			}
		}
		return true;
	}


	/**
	 * Sort index of a vector in ascending order
	 *
	 * @param x the sorted index.
	 * @param v the vector.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool sort_index_ascend(idx* x, const vec* v) {
		if(x->n != v->n) {
			printf("(sort_index) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < v->n; i++) {
			x->e[i] = i;
		}
		if(v->n < 2) {
			return true;
		}
		vec* u = new vec(v->n);
		copy(u, v);
		float utemp;
		u32 xtemp;
		for(u32 i = u->n-1; i > 0; i--) {
			for(u32 j = 0; j < i; j++) {
				if(u->e[j] > u->e[j+1]) {
					utemp = u->e[j];
					u->e[j] =u->e[j+1];
					u->e[j+1] = utemp;
					xtemp = x->e[j];
					x->e[j] = x->e[j+1];
					x->e[j+1] = xtemp;
				}
			}
		}
		delete u;
		return true;
	}


	/**
	 * Sort index of a vector in descending order
	 *
	 * @param x the sorted index.
	 * @param v the vector.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool sort_index_descend(idx* x, const vec* v) {
		if(x->n != v->n) {
			printf("(sort_index_descend) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < v->n; i++) {
			x->e[i] = i;
		}
		if(v->n < 2) {
			return true;
		}
		vec* u = new vec(v->n);
		copy(u, v);
		float utemp;
		u32 xtemp;
		for(u32 i = u->n-1; i > 0; i--) {
			for(u32 j = 0; j < i; j++) {
				if(u->e[j] < u->e[j+1]) {
					utemp = u->e[j];
					u->e[j] =u->e[j+1];
					u->e[j+1] = utemp;
					xtemp = x->e[j];
					x->e[j] = x->e[j+1];
					x->e[j+1] = xtemp;
				}
			}
		}
		delete u;
		return true;
	}


	/**
	 * Transform a vector to a diagonal matrix
	 *
	 * @param c the matrix.
	 * @param v the vector.
	 * @return whether this function has been excuted successfully.
	 */
	__host__ __device__ bool diag(mat* c, const vec* v) {
		if(c->n_col != v->n || c->n_row != v->n) {
			printf("(sort_index_descend) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < v->n; i++) {
			c->col[i]->e[i] = v->e[i];
		}
		return true;
	}
}
