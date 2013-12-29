/**
 * PaSS_BLAS.cu
 * The basic linear algebra sub-programs for PaSS
 *
 * @author emfo
 */

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "curand_kernel.h"
#include "device_functions.h"
#include "float.h"
#include "math_constants.h"
#include "math_functions.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

/**
 * The PaSS_BLAS namespace
 */
namespace pass_blas {
	/**
	 * 32-bit unsigned integer.
	 */
	typedef uint32_t u32;

	/**
	 * The vector structure
	 */
	struct vec {
		u32 n;    /**< the length */
		float* e; /**< the array of entries */
		
		/**
		 * Construct a vector.
		 *
		 * @param n the length of the vector.
		 */
		__host__ __device__ vec(const u32 n) {
			this->n = n;
			this->e = (float*)malloc(n * sizeof(float));
		}
		
		/**
		 * Construct a vector and fill it with given value.
		 *
		 * @param n the length of the vector.
		 * @param f the value of entries.
		 */
		__host__ __device__ vec(const u32 n, const float f) {
			this->n = n;
			this->e = (float*)malloc(n * sizeof(float));
			for(u32 i = 0; i < n; i++) {
				this->e[i] = f;
			}
		}
		
		/**
		 * Destruct the vector.
		 */
		__host__ __device__ ~vec() {
			free(this->e);
		}
	};


	/**
	 * The matrix structure
	 */
	struct mat {
		u32 n_row; /**< the number of rows */
		u32 n_col; /**< the number of columns */
		vec** col; /**< the array of columns */
		
		/**
		 * Construct a matrix.
		 *
		 * @param p the number of rows of the vector.
		 * @param q the number of columns of the vector.
		 */
		__host__ __device__ mat(const u32 p, const u32 q) {
			this->n_row = p;
			this->n_col = q;
			this->col = (vec**)malloc(q * sizeof(vec*));
			for(u32 i = 0; i < q; i++) {
				this->col[i] = new vec(p);
			}
		}
		
		/**
		 * Construct a matrix and fill it with given value.
		 *
		 * @param p the number of rows of the vector.
		 * @param q the number of columns of the vector.
		 * @param f the value of entries.
		 */
		__host__ __device__ mat(const u32 p, const u32 q, const float f) {
			this->n_row = p;
			this->n_col = q;
			this->col = (vec**)malloc(q * sizeof(vec*));
			for(u32 i = 0; i < q; i++) {
				this->col[i] = new vec(p, f);
			}
		}
		
		/**
		 * Destruct the matrix.
		 */
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
		u32 n;  /**< the length */
		u32* e; /**< the array of entries */
		
		/**
		 * Construct a index.
		 *
		 * @param n the length of the index.
		 */
		__host__ __device__ idx(const u32 n) {
			this->n = n;
			this->e = (u32*)malloc(n * sizeof(u32));
		}
		
		/**
		 * Construct a index and fill it with given value.
		 *
		 * @param n the length of the index.
		 * @param f the value of entries.
		 */
		__host__ __device__ idx(const u32 n, const u32 f) {
			this->n = n;
			this->e = (u32*)malloc(n * sizeof(u32));
			for(u32 i = 0; i < n; i++) {
				this->e[i] = f;
			}
		}
		
		/**
		 * Destruct the index.
		 */
		__host__ __device__ ~idx() {
			free(this->e);
		}
	};


	/**
	 * Display the vector.
	 *
	 * @param v the vector.
	 * @return whether this function has been executed successfully or not.
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
	 * @return whether this function has been executed successfully or not.
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
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool print(const idx* x) {
		for(u32 i = 0; i < x->n; i++) {
			printf("%4u ", x->e[i]);
		}
		printf("\n");
		return true;
	}


	/**
	 * Copy a vector.
	 *
	 * @param u the new vector.
	 * @param v the original vector.
	 * @return whether this function has been executed successfully or not.
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
	 * Copy a matrix.
	 *
	 * @param c the new matrix.
	 * @param a the original matrix.
	 * @return whether this function has been executed successfully or not.
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
	 * Copy a index.
	 *
	 * @param z the new index.
	 * @param x the original index.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool copy(idx* z, const idx* x) {
		if(z->n != x->n) {
			printf("(copy: index) not aligned!\n");
			return false;
		}
		memcpy(z->e, x->e, x->n * sizeof(u32));
		return true;
	}


	/**
	 * Put a index in another index. (Note that this function won't check the length if z)
	 *
	 * @param z the target index.
	 * @param x the input index.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool put(idx* z, const idx* x) {
		z->n = x->n;
		memcpy(z->e, x->e, x->n * sizeof(u32));
		return true;
	}


	/**
	 * u = v+w.
	 *
	 * @param u the sum vector.
	 * @param v the augend vector.
	 * @param w the addend vector.
	 * @return whether this function has been executed successfully or not.
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
	 * @return whether this function has been executed successfully or not.
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
	 * @return whether this function has been executed successfully or not.
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
	 * @return whether this function has been executed successfully or not.
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
	 * u = f*v.
	 *
	 * @param u the product vector.
	 * @param v the multiplier vector.
	 * @param f the multiplicand number.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool mul(vec* u, const vec* v, const float f) {
		for(u32 i = 0; i < v->n; i++) {
			u->e[i] = v->e[i] * f;
		}
		return true;
	}


	/**
	 * c = f*a.
	 *
	 * @param c the product matrix.
	 * @param a the multiplier vector.
	 * @param f the multiplicand number.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool mul(mat* c, const mat* a, const float f) {
		for(u32 i = 0; i < a->n_col; i++) {
			for(u32 j = 0; j < a->n_row; j++) {
				c->col[i]->e[j] = a->col[i]->e[j] * f;
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
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool mul(vec* u, const mat* a, const vec* v) {
		if(u->n != a->n_row || v->n != a->n_col) {
			printf("(mul: matrix left) not aligned!\n");
			return false;
		}
		for(u32 j = 0; j < a->n_row; j++) {
			u->e[j] = 0;
			for(u32 i = 0; i < a->n_col; i++) {
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
	 * @return whether this function has been executed successfully or not.
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
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool mul(mat* c, const vec* v, const vec* w) {
		if(v->n != c->n_row || w->n != c->n_col) {
			printf("(mul: matrix) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < c->n_col; i++) {
			for(u32 j = 0; j < c->n_row; j++) {
				c->col[i]->e[j] = v->e[j] * w->e[i];
			}
		}
		return true;
	}


	/**
	 * f = sum(v.*v).
	 *
	 * @param f the product number.
	 * @param v the vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool inner(float* f, const vec* v) {
		*f = 0;
		for(u32 i = 0; i < v->n; i++) {
			*f += v->e[i] * v->e[i];
		}
		return true;
	}


	/**
	 * f = sum(v.*w).
	 *
	 * @param v the multiplicand vector.
	 * @param w the multiplier vector.
	 * @param f the product number.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool inner(float* f, const vec* v, const vec* w) {
		if(v->n != w->n) {
			printf("(inner: vector) not aligned!\n");
			return false;
		}
		*f = 0;
		for(u32 i = 0; i < v->n; i++) {
			*f += v->e[i] * w->e[i];
		}
		return true;
	}


	/**
	 * u = sum(a.*a).
	 *
	 * @param u the product vector.
	 * @param a the matrix.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool inner(vec* u, const mat* a) {
		if(u->n != a->n_col) {
			printf("(inner: matrix) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < a->n_col; i++) {
			u->e[i] = 0;
			for(u32 j = 0; j < a->n_row; j++) {
				u->e[i] += a->col[i]->e[j] * a->col[i]->e[j];
			}
		}
		return true;
	}


	/**
	 * u' = sum(a.*b).
	 *
	 * @param u the product vector.
	 * @param a the multiplicand matrix.
	 * @param b the multiplier matrix.
	 * @return whether this function has been executed successfully or not.
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
	 * f = norm(v, 2).
	 *
	 * @param f the Euclidean norm.
	 * @param v the vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool norm(float* f, const vec* v) {
		inner(f, v);
		*f = sqrt(*f);
		return true;
	}


	/**
	 * Add a new entry at the end.
	 *
	 * @param v the vector.
	 * @param f the new entry.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool insert(vec* v, const float f) {
		v->n++;
		float* temp = (float*)malloc(v->n * sizeof(float));
		memcpy(temp, v->e, (v->n-1) * sizeof(float));
		free(v->e);
		v->e = temp;
		v->e[v->n-1] = f;
		return true;
	}


	/**
	 * Add a new entry at the end.
	 *
	 * @param x the index.
	 * @param i the new entry.
	 * @return whether this function has been executed successfully or not.
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
	 * Add a new row and a new column at the end.
	 *
	 * @param a the matrix.
	 * @param f the number.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool insert(mat* a, const float f) {
		a->n_row++;
		a->n_col++;
		float* ftemp;
		for(u32 i = 0; i < a->n_col-1; i++) {
			a->col[i]->n++;
			ftemp = (float*)malloc(a->n_row * sizeof(float));
			memcpy(ftemp, a->col[i]->e, (a->n_row-1) * sizeof(float));
			free(a->col[i]->e);
			a->col[i]->e = ftemp;
			a->col[i]->e[a->n_row-1] = f;
		}
		vec** temp = (vec**)malloc(a->n_col * sizeof(vec*));
		memcpy(temp, a->col, (a->n_col-1) * sizeof(vec*));
		free(a->col);
		a->col = temp;
		a->col[a->n_col-1] = new vec(a->n_row, f);
		return true;
	}


	/**
	 * Add a new row at the end.
	 *
	 * @param a the matrix.
	 * @param v the new vector.
	 * @return whether this function has been executed successfully or not.
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
	 * @return whether this function has been executed successfully or not.
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
	 * Remove the last entry.
	 *
	 * @param v the vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool shed(vec* v) {
		if(v->n <= 1) {
			printf("(shed: vector) too small!\n");
			return false;
		}
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
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool shed(idx* x) {
		if(x->n <= 1) {
			printf("(shed: index) too small!\n");
			return false;
		}
		x->n--;
		u32* temp = (u32*)malloc(x->n * sizeof(u32));
		memcpy(temp, x->e, x->n * sizeof(u32));
		free(x->e);
		x->e = temp;
		return true;
	}


	/**
	 * Remove the last n entries.
	 *
	 * @param x the index.
	 * @param n number of entries.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool shed(idx* x, const u32 n) {
		if(n == 0) {
			return true;
		}
		if(x->n < n) {
			printf("(shed: index) n too large!\n");
			return false;
		}
		x->n -= n;
		if(x->n == 0) {
			free(x->e);
			x->e = NULL;
			return true;
		}
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
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool shed_row(mat* a) {
		if(a->n_row == 0) {
			printf("(shed_row) empty!\n");
			return false;	
		}
		a->n_row--;
		if(a->n_row == 0) {
			for(u32 i = 0; i < a->n_col; i++) {
				a->col[i]->n = 0;
				free(a->col[i]->e);
				a->col[i]->e = 0;
			}
			return true;
		}
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
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool shed_col(mat* a) {
		if(a->n_col == 0) {
			printf("(shed_col) empty!\n");
			return false;
		}
		a->n_col--;
		delete a->col[a->n_col];
		if(a->n_col == 0) {
			free(a->col);
			a->col = NULL;
			return true;
		}
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
	 * @return whether this function has been executed successfully or not.
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
	 * @param x the index.
	 * @param i the index of first entry.
	 * @param j the index of second entry.
	 * @return whether this function has been executed successfully or not.
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
	 * @return whether this function has been executed successfully or not.
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
	 * @return whether this function has been executed successfully or not.
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
	 * @param i the element.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool find_index(u32* k, const idx* x, const u32 i) {
		for(u32 j = 0; j < x->n; j++) {
			if(x->e[j] == i) {
				*k = j;
				return true;
			}
		}
		*k = (u32)(-1);
		printf("(find_index) index not found!\n");
		return false;
	}


	/**
	 * Find the index of minimal element
	 *
	 * @param k the index.
	 * @param v the vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool find_min_index(u32* k, const vec* v) {
		float f = FLT_MAX;
		for(u32 j = 0; j < v->n; j++) {
			if(v->e[j] < f) {
				f = v->e[j];
				*k = j;
			}
		}
		return true;
	}


	/**
	 * Sort a index in ascending order
	 *
	 * @param x the index.
	 * @return whether this function has been executed successfully or not.
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
	 * @return whether this function has been executed successfully or not.
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
	 * @param z the sorted index.
	 * @param v the vector (will be sorted).
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool sort_index_ascend(idx* z, vec* v) {
		if(z->n != v->n) {
			printf("(sort_index_ascend) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < v->n; i++) {
			z->e[i] = i;
		}
		if(v->n < 2) {
			return true;
		}
		float vtemp;
		u32 ztemp;
		for(u32 i = v->n-1; i > 0; i--) {
			for(u32 j = 0; j < i; j++) {
				if(v->e[j] > v->e[j+1]) {
					vtemp = v->e[j];
					v->e[j] = v->e[j+1];
					v->e[j+1] = vtemp;
					ztemp = z->e[j];
					z->e[j] = z->e[j+1];
					z->e[j+1] = ztemp;
				}
			}
		}
		return true;
	}


	/**
	 * Sort index of a vector in descending order
	 *
	 * @param z the sorted index.
	 * @param v the vector (will be sorted).
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool sort_index_descend(idx* z, vec* v) {
		if(z->n != v->n) {
			printf("(sort_index_descend) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < v->n; i++) {
			z->e[i] = i;
		}
		if(v->n < 2) {
			return true;
		}
		float vtemp;
		u32 ztemp;
		for(u32 i = v->n-1; i > 0; i--) {
			for(u32 j = 0; j < i; j++) {
				if(v->e[j] < v->e[j+1]) {
					vtemp = v->e[j];
					v->e[j] = v->e[j+1];
					v->e[j+1] = vtemp;
					ztemp = z->e[j];
					z->e[j] = z->e[j+1];
					z->e[j+1] = ztemp;
				}
			}
		}
		return true;
	}


	/**
	 * Set complement of a sorted index
	 *
	 * @param z the complement set index.
	 * @param x the origin set index.
	 * @param n the length of universe.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool complement(idx* z, const idx* x, const u32 n) {
		if(z->n + x->n != n) {
			printf("(complement) not aligned!\n");
			return false;
		}
		u32 i, j, f;
		for(i = 0, j = 0, f = 0; j < z->n; f++) {
			if(i < x->n && f == x->e[i]) {
				i++;
			}
			else {
				z->e[j] = f;
				j++;
			}
		}
		return true;
	}


	/**
	 * Set difference of two sorted index
	 *
	 * @param z the difference set index.
	 * @param x the minuend set index.
	 * @param y the subtrahend set index.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool set_difference(idx* z, const idx* x, const idx* y) {
		if(z->n != x->n) {
			printf("(set_difference) not aligned!\n");
			return false;
		}
		u32 i, j, k;
		for(i = 0, j = 0, k = 0; i < x->n && j < y->n;) {
			if(x->e[i] < y->e[j]) {
				z->e[k] = x->e[i];
				i++;
				k++;
			} else if(x->e[i] == y->e[j]) {
				i++;
				j++;
			} else {
				j++;
			}
		}
		for(; i < x->n; i++, k++) {
			z->e[k] = x->e[i];
		}
		while(k != z->n) {
			shed(z, z->n - k);
		}
		return true;
	}
}
