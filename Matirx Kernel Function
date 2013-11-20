#include <cstdint>

struct vec {		// the 
	uint32* n;		// the length
	double* e;		// the array of entries
};

struct mat {
	uint32 n_row;	// the number of rows
	uint32 n_col;	// the number of cols
	vec** col;		// the array of columns
};

// Note that *(m->col[i]->n) = m->n_row for all i, where m is a mat*

vec* add(const vec* a, const vec* b);						// calculate a+b
mat* add(const mat* a, const mat* b);						// calculate a+b

vec* sub(const vec* a, const vec* b);						// calculate a-b
mat* sub(const mat* a, const mat* b);						// calculate a-b

vec* mul(const vec* v, const double e);						// calculate e*v
mat* mul(const mat* m, const double e);						// calculate e*m
vec* mul(const mat* m, const vec* v);						// calculate m*v
vec* mul(const vec* v, const mat* m);						// calculate (v'*m)'
mat* mul(const vec* v, const vec* w);						// calculate v*w'

double inner(const vec* v);									// calculate sum(v.*v)
double inner(const vec* a, const vec* b);					// calculate sum(a.*b)
double inner(const mat* m);									// calculate sum(sum(m.*m))
double inner(const mat* a, const mat* b);					// calculate sum(sum(a.*b))

void extend(vec* v, const double e);						// add a new entry
void extend_col(mat* m, const uint32 i, const uint32 j);	// add a new col
void extend_row(mat* m, const uint32 i, const uint32 j);	// add a new row

void permute(vec* v, const double e);						// permute two entries
void permute_col(mat* m, const vec* v);						// permute two cols
void permute_row(mat* m, const vec* v);						// permute two rows
