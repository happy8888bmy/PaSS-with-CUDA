/**
 * PaSS_BLAS.cu <br>
 * The basic linear algebra sub-programs for PaSS
 *
 * \author emfo
 * \date 2014.01.13 03:44
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

#define n_ents(a) \
        (a)[0]
#define m_ents(a) \
        (a)[1]
#define size_vec(a) \
        (m_ents(a)+2)

#define entry(a, i) \
        (a)[(i)+2]

#define n_cols(a) \
        (a)[0]
#define m_cols(a) \
        (a)[1]
#define n_rows(a) \
        (a)[2]
#define m_rows(a) \
        (a)[3]
#define size_mat(a) \
        ((m_rows(a)+2)*m_cols(a)+2)

#define col(a, j) \
        ((a)+(j)*((u32)m_rows(a)+2)+2)
#define entry2(a, i, j) \
        (a)[(j)*((u32)m_rows(a)+2)+i+4]


/**
 * The PaSS_BLAS namespace
 */
namespace pass_blas {
	/**
	 * 32-bit unsigned integer type.
	 */
	typedef uint32_t u32;
	/**
	 * vector type. <br>
	 * | number of entries | max number of entries  | entries |
	 */
	typedef float vec;
	/**
	 * matrix type. <br>
	 * | number of columns | max number of columns | column 0 (vec) | column 1 (vec) | ... |
	 */
	typedef float mat;
	/**
	 * u32 vector type. <br>
	 * | number of entries | max number of entries  | entries |
	 */
	typedef u32 uvec;

	/**
	 * Initial the vector.
	 *
	 * @param v the vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void init_vec(vec* v, const u32 n, const u32 m) {
		n_ents(v) = n;
		m_ents(v) = m;
	}

	/**
	 * Initial the matrax.
	 *
	 * @param a the matrix.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void init_mat(mat* a, const u32 p, const u32 q, const u32 mp, const u32 mq) {
		u32 j;
		n_cols(a) = q;
		m_cols(a) = mq;
		n_rows(a) = p;
		m_rows(a) = mp;
		for(j = 1; j < n_cols(a); j++) {
			a[j*((u32)m_rows(a)+2)+2] = p;
			a[j*((u32)m_rows(a)+2)+3] = mp;
		}
	}

	/**
	 * Initial the u32 vector.
	 *
	 * @param x the u32 vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void init_uvec(uvec* x, const u32 n, const u32 m) {
		n_ents(x) = n;
		m_ents(x) = m;
	}

	/**
	 * Display the vector.
	 *
	 * @param v the vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void print_vec(const vec* v) {
		u32 i;
		for(i = 0; i < n_ents(v); i++) {
			printf("%8.3f  ", entry(v, i));
		}
		printf("\n");
	}


	/**
	 * Display the matrix.
	 *
	 * @param a the vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void print_mat(const mat* a) {
		u32 i, j;
		for(i = 0; i < n_rows(a); i++) {
			for(j = 0; j < n_cols(a); j++) {
				printf("%8.3f  ", entry2(a, i, j));
			}
			printf("\n");
		}
		printf("\n");
	}


	/**
	 * Display the u32 vector.
	 *
	 * @param x the u32 vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void print_uvec(const uvec* x) {
		u32 i;
		for(i = 0; i < n_ents(x); i++) {
			printf("%4u ", entry(x, i));
		}
		printf("\n");
	}


	/**
	 * u = v+w.
	 *
	 * @param u the sum vector.
	 * @param v the augend vector.
	 * @param w the addend vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool add_vec(vec* u, const vec* v, const vec* w) {
		u32 i;
		if(n_ents(v) != n_ents(w)) {
			printf("(add_vec) not aligned!\n");
			return false;
		}
		n_ents(u) = n_ents(v);
		for(i = 0; i < n_ents(v); i++) {
			entry(u, i) = entry(v, i) + entry(w, i);
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
	__host__ __device__ bool add_mat(mat* c, const mat* a, const mat* b) {
		u32 i, j;
		if(n_rows(a) != n_rows(b) || n_cols(a) != n_cols(b)) {
			printf("(add_mat) not aligned!\n");
			return false;
		}
		n_rows(c) = n_rows(a);
		n_cols(c) = n_cols(a);
		for(j = 0; j < n_cols(a); j++) {
			for(i = 0; i < n_rows(a); i++) {
				entry2(c, i, j) = entry2(a, i, j) + entry2(b, i, j);
			}
		}
		return true;
	}


	/**
	 * u = v-w.
	 *
	 * @param u the difference vector.
	 * @param v the minuend vector.
	 * @param w the subtrahend vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool sub_vec(vec* u, const vec* v, const vec* w) {
		u32 i;
		if(n_ents(v) != n_ents(w)) {
			printf("(sub_vec) not aligned!\n");
			return false;
		}
		n_ents(u) = n_ents(v);
		for(i = 0; i < n_ents(v); i++) {
			entry(u, i) = entry(v, i) - entry(w, i);
		}
		return true;
	}


	/**
	 * c = a-b.
	 *
	 * @param b the difference matrix.
	 * @param c the minuend matrix.
	 * @param a the subtrahend matrix.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool sub_mat(mat* c, const mat* a, const mat* b) {
		u32 i, j;
		if(n_cols(a) != n_cols(b) || n_rows(a) != n_rows(b)) {
			printf("(sub_mat) not aligned!\n");
			return false;
		}
		n_rows(c) = n_rows(a);
		n_cols(c) = n_cols(a);
		for(j = 0; j < n_cols(a); j++) {
			for(i = 0; i < n_rows(a); i++) {
				entry2(c, i, j) = entry2(a, i, j) - entry2(b, i, j);
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
	__host__ __device__ void mul_vec(vec* u, const vec* v, float f) {
		u32 i;
		n_ents(u) = n_ents(v);
		for(i = 0; i < n_ents(v); i++) {
			entry(u, i) = entry(v, i) * f;
		}
	}


	/**
	 * c = f*a.
	 *
	 * @param c the product matrix.
	 * @param a the multiplier vector.
	 * @param f the multiplicand number.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void mul_mat(mat* c, const mat* a, float f) {
		u32 i, j;
		n_rows(c) = n_rows(a);
		n_cols(c) = n_cols(a);
		for(j = 0; j < n_cols(a); j++) {
			for(i = 0; i < n_rows(a); i++) {
				entry2(c, i, j) = entry2(a, i, j) * f;
			}
		}
	}


	/**
	 * u = a*v
	 *
	 * @param u the product vector.
	 * @param a the multiplicand matrix.
	 * @param v the multiplier vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool mul_matvec(vec* u, const mat* a, const vec* v) {
		u32 i, j;
		if(n_ents(v) != n_cols(a)) {
			printf("(mul_matvec) not aligned!\n");
			return false;
		}
		n_ents(u) = n_rows(a);
		for(i = 0; i < n_rows(a); i++) {
			entry(u, i) = 0;
			for(j = 0; j < n_cols(a); j++) {
				entry(u, i) += entry2(a, i, j) * entry(v, j);
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
	__host__ __device__ bool mul_vecmat(vec* u, const vec* v, const mat* a) {
		u32 i, j;
		if(n_ents(v) != n_rows(a)) {
			printf("(mul_vecmat) not aligned!\n");
			return false;
		}
		n_ents(u) = n_cols(a);
		for(j = 0; j < n_cols(a); j++) {
			entry(u, j) = 0;
			for(i = 0; i < n_rows(a); i++) {
				entry(u, j) += entry2(a, i, j) * entry(v, i);
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
	__host__ __device__ void mul_vecvec(mat* c, const vec* v, const vec* w) {
		u32 i, j;
		n_rows(c) = n_ents(v);
		n_cols(c) = n_ents(w);
		for(j = 0; j < n_cols(c); j++) {
			for(i = 0; i < n_rows(c); i++) {
				entry2(c, i, j) = entry(v, i) * entry(w, j);
			}
		}
	}


	/**
	 * f = sum(v.*v).
	 *
	 * @param f the product number.
	 * @param v the vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void inner_vec(float* f, const vec* v) {
		u32 i;
		*f = 0;
		for(i = 0; i < n_ents(v); i++) {
			*f += entry(v, i) * entry(v, i);
		}
	}


	/**
	 * f = sum(v.*w).
	 *
	 * @param v the multiplicand vector.
	 * @param w the multiplier vector.
	 * @param f the product number.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool inner_vec(float* f, const vec* v, const vec* w) {
		if(n_ents(v) != n_ents(w)) {
			printf("(inner_vec) not aligned!\n");
			return false;
		}
		*f = 0;
		for(u32 i = 0; i < n_ents(v); i++) {
			*f += entry(v, i) * entry(w, i);
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
	__host__ __device__ void inner_mat(vec* u, const mat* a) {
		u32 i, j;
		n_ents(u) = n_cols(a);
		for(j = 0; j < n_cols(a); i++) {
			entry(u, j) = 0;
			for(i = 0; i < n_rows(a); i++) {
				entry(u, j) += entry2(a, i, j) * entry2(a, i, j);
			}
		}
	}


	/**
	 * u' = sum(a.*b).
	 *
	 * @param u the product vector.
	 * @param a the multiplicand matrix.
	 * @param b the multiplier matrix.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool inner_mat(vec* u, const mat* a, const mat* b) {
		u32 i, j;
		if(n_rows(a) != n_rows(b) || n_cols(a) != n_cols(b)) {
			printf("(inner_mat) not aligned!\n");
			return false;
		}
		n_ents(u) = n_cols(a);
		for(j = 0; j < n_cols(a); j++) {
			entry(u, j) = 0;
			for(i = 0; i < n_rows(a); i++) {
				entry(u, j) += entry2(a, i, j) * entry2(b, i, j);
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
	__host__ __device__ void norm_vec(float* f, vec* v) {
		inner_vec(f, v);
		*f = sqrt(*f);
	}


	/**
	 * Add a new entry at the end.
	 *
	 * @param v the vector.
	 * @param f the new entry.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void insert_vec(vec* v, float f) {
		entry(v, (u32)n_ents(v)) = f;
		n_ents(v)++;
	}


	/**
	 * Add a new entry at the end.
	 *
	 * @param x the index.
	 * @param i the new entry.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void insert_uvec(uvec* x, u32 i) {
		entry(x, n_ents(x)) = i;
		n_ents(x)++;
	}


	/**
	 * Add a new row and a new column at the end.
	 *
	 * @param a the matrix.
	 * @param f the number.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void insert_mat(mat* a, float f) {
		u32 i;
		for(i = 0; i < n_rows(a); i++) {
			entry2(a, i, (u32)n_cols(a)) = f;
		}
		for(i = 0; i <= n_cols(a); i++) {
			entry2(a, (u32)n_rows(a), i) = f;
		}
		n_rows(a)++;
		n_cols(a)++;
	}


	/**
	 * Add a new row at the end.
	 *
	 * @param a the matrix.
	 * @param v the new vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool insert_row(mat* a, vec* v) {
		u32 i;
		if(n_cols(a) != n_ents(v)) {
			printf("(insert_row) not aligned!\n");
			return false;
		}
		for(i = 0; i < n_cols(a); i++) {
			entry2(a, (u32)n_rows(a), i) = entry(v, i);
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
	__host__ __device__ bool insert_col(mat* a, vec* v) {
		u32 i;
		if(n_rows(a) != n_ents(v)) {
			printf("(insert_col) not aligned!\n");
			return false;
		}
		for(i = 0; i < n_rows(a); i++) {
			entry2(a, i, (u32)n_cols(a)) = entry(v, i);
		}
		return true;
	}


	/**
	 * Remove the last entry.
	 *
	 * @param v the vector.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool shed(vec* v) {
		if(n_ents(v) <= 1) {
			printf("(shed: vector) too small!\n");
			return false;
		}
		n_ents(v)--;
		return true;
	}


	/**
	 * Remove the last entry.
	 *
	 * @param x the index.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool shed(uvec* x) {
		if(n_ents(x) <= 1) {
			printf("(shed: index) too small!\n");
			return false;
		}
		n_ents(x)--;
		return true;
	}


	/**
	 * Remove the last n entries.
	 *
	 * @param x the index.
	 * @param n number of entries.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool shed(uvec* x, u32 n) {
		if(n == 0) {
			return true;
		}
		if(n_ents(x) < n) {
			printf("(shed: index) n too large!\n");
			return false;
		}
		n_ents(x) -= n;
		return true;
	}


	/**
	 * Remove the last row.
	 *
	 * @param a the matrix.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool shed_row(mat* a) {
		if(n_rows(a) == 0) {
			printf("(shed_row) empty!\n");
			return false;	
		}
		n_rows(a)--;
		return true;
	}


	/**
	 * Remove the last column.
	 *
	 * @param a the matrix.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool shed_col(mat* a) {
		if(n_cols(a) == 0) {
			printf("(shed_col) empty!\n");
			return false;
		}
		n_cols(a)--;
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
	__host__ __device__ void swap(vec* v, u32 i, u32 j) {
		float temp = entry(v, i);
		entry(v, i) = entry(v, j);
		entry(v, j) = temp;
	}


	/**
	 * Swap two entries.
	 *
	 * @param x the index.
	 * @param i the index of first entry.
	 * @param j the index of second entry.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool swap(uvec* x, u32 i, u32 j) {
		u32 temp = entry(x, i);
		entry(x, i) = entry(x, j);
		entry(x, j) = temp;
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
	__host__ __device__ void swap_row(mat* a, u32 i, u32 j) {
		u32 k;
		float temp;
		for(k = 0; k < n_cols(a); k++) {
			temp = entry2(a, i, k);
			entry2(a, i, k) = entry2(a, j, k);
			entry2(a, j, k) = temp;
		}
	}


	/**
	 * Swap two columns.
	 *
	 * @param a the matrix.
	 * @param i the index of first column.
	 * @param j the index of second column.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void swap_col(mat* a, u32 i, u32 j) {
		u32 k;
		float temp;
		for(k = 0; k < n_rows(a); k++) {
			temp = entry2(a, k, i);
			entry2(a, k, i) = entry2(a, k, j);
			entry2(a, k, j) = temp;
		}
	}


	/**
	 * Find the index of target element
	 *
	 * @param k the index.
	 * @param x the vector.
	 * @param i the element.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool find_index(u32* k, uvec* x, u32 i) {
		u32 j;
		for(j = 0; j < n_ents(x); j++) {
			if(entry(x, j) == i) {
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
	__host__ __device__ void find_min_index(u32* k, vec* v) {
		u32 j;
		float f = FLT_MAX;
		for(j = 0; j < n_ents(v); j++) {
			if(entry(v, j) < f) {
				f = entry(v, j);
				*k = j;
			}
		}
	}


	/**
	 * Sort a index in ascending order
	 *
	 * @param x the index.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ void sort_ascend(uvec* x) {
		u32 i, j, temp;
		for(i = n_ents(x)-1; i > 0; i--) {
			for(j = 0; j < i; j++) {
				if(entry(x, j) > entry(x, j+1)) {
					temp = entry(x, j);
					entry(x, j) = entry(x, j+1);
					entry(x, j+1) = temp;
				}
			}
		}
	}


	/**
	 * Sort a index in descending order
	 *
	 * @param x the index.
	 * @return whether this function has been executed successfully or not.
	 */
	__host__ __device__ bool sort_descend(uvec* x) {
		u32 i, j, temp;
		for(i = n_ents(x)-1; i > 0; i--) {
			for(j = 0; j < i; j++) {
				if(entry(x, j) < entry(x, j+1)) {
					temp = entry(x, j);
					entry(x, j) = entry(x, j+1);
					entry(x, j+1) = temp;
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
	__host__ __device__ bool sort_index_ascend(uvec* z, vec* v) {
		u32 i, j, ztemp;
		float vtemp;
		if(n_rows(z) != n_ents(v)) {
			printf("(sort_index_ascend) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < n_ents(v); i++) {
			entry(z, i) = i;
		}
		if(n_ents(v) < 2) {
			return true;
		}
		for(i = n_ents(v)-1; i > 0; i--) {
			for(j = 0; j < i; j++) {
				if(entry(v, j) > entry(v, j+1)) {
					vtemp = entry(v, j);
					entry(v, j) = entry(v, j+1);
					entry(v, j+1) = vtemp;
					ztemp = entry(z, j);
					entry(z, j) = entry(z, j+1);
					entry(z, j+1) = ztemp;
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
	__host__ __device__ bool sort_index_descend(uvec* z, vec* v) {
		u32 i, j, ztemp;
		float vtemp;
		if(n_rows(z) != n_ents(v)) {
			printf("(sort_index_descend) not aligned!\n");
			return false;
		}
		for(u32 i = 0; i < n_ents(v); i++) {
			entry(z, i) = i;
		}
		if(n_ents(v) < 2) {
			return true;
		}
		for(i = n_ents(v)-1; i > 0; i--) {
			for(j = 0; j < i; j++) {
				if(entry(v, j) < entry(v, j+1)) {
					vtemp = entry(v, j);
					entry(v, j) = entry(v, j+1);
					entry(v, j+1) = vtemp;
					ztemp = entry(z, j);
					entry(z, j) = entry(z, j+1);
					entry(z, j+1) = ztemp;
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
	__host__ __device__ bool complement(uvec* z, uvec* x, u32 n) {
		if(n_ents(z) + n_ents(x) != n) {
			printf("(complement) not aligned!\n");
			return false;
		}
		u32 i, j, f;
		for(i = 0, j = 0, f = 0; j < n_rows(z); f++) {
			if(i < n_ents(x) && f == entry(x, i)) {
				i++;
			}
			else {
				entry(z, j) = f;
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
	__host__ __device__ bool set_difference(uvec* z, uvec* x, uvec* y) {
		if(n_rows(z) != n_ents(x)) {
			printf("(set_difference) not aligned!\n");
			return false;
		}
		u32 i, j, k;
		for(i = 0, j = 0, k = 0; i < n_ents(x) && j < y[0];) {
			if(entry(x, i) < entry(y, j)) {
				entry(z, k) = entry(x, i);
				i++;
				k++;
			} else if(entry(x, i) == entry(y, j)) {
				i++;
				j++;
			} else {
				j++;
			}
		}
		for(; i < n_ents(x); i++, k++) {
			entry(z, k) = entry(x, i);
		}
		while(k != n_rows(z)) {
			shed(z, n_rows(z) - k);
		}
		return true;
	}
}
