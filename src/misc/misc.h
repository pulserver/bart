/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * Copyright 2017. University of Oxford.
 * Copyright 2017-2018. Damien Nguyen
 * Copyright 2019. Uecker Lab, University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef _MISC_H
#define _MISC_H

#include <stdlib.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdnoreturn.h>
#include <setjmp.h>

#include "misc/nested.h"
#include "misc/dimtypes.h"
#include "misc/format.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#define MIN(x, y) ({ __typeof(x) __x = (x); __typeof(y) __y = (y); (__x < __y) ? __x : __y; })
#define MAX(x, y) ({ __typeof(x) __x = (x); __typeof(y) __y = (y); (__x > __y) ? __x : __y; })
#define CLAMP(x, min, max) MAX(min, MIN(x, max))

#define MAKE_ARRAY(x, ...) ((__typeof__(x)[]){ x, __VA_ARGS__ })
#define ARRAY_SIZE(x)	(sizeof(x) / sizeof(x[0]))

#ifndef unreachable
#define unreachable() __builtin_trap()
#endif

#define SWAP(x, y) do { __auto_type temp = x; x = y; y = temp; } while (0)

#include "misc/cppwrap.h"

#ifndef alloc_size
#ifndef __clang__
#define alloc_size(x) [[gnu::alloc_size(x)]]
#else
#define alloc_size(x)
#endif
#endif

extern void* xmalloc(size_t s) alloc_size(1);
extern void* xrealloc(void *, size_t s) alloc_size(2);
extern char* xstrdup(const char* str);
extern void xfree(const void*);
extern void warn_nonnull_ptr(void*);

#define XMALLOC(x)	(x = xmalloc(sizeof(*x)))
#define XFREE(x)	(xfree(x), x = NULL)

#define _TYPE_ALLOC(T)		((T*)xmalloc(sizeof(T)))
#define TYPE_ALLOC(T)		_TYPE_ALLOC(__typeof__(T))
// #define TYPE_CHECK(T, x)	({ T* _ptr1 = 0; __typeof(x)* _ptr2 = _ptr1; (void)_ptr2; (x);  })

#ifndef __clang__
#define _PTR_ALLOC(T, x)										\
	T* x __attribute__((cleanup(warn_nonnull_ptr))) = xmalloc(sizeof(T))
#else
#define _PTR_ALLOC(T, x)										\
	T* x = xmalloc(sizeof(T))
#endif

#define _CONCAT(A, B) A ## B
#define CONCAT(A, B) _CONCAT(A, B)
//FIXME _STRINGIFY is defined in mpi_portable_platform.h
#define STRINGIFY_HELPER(x) # x
#define STRINGIFY(x) STRINGIFY_HELPER(x)


#define PTR_ALLOC(T, x)		_PTR_ALLOC(__typeof__(T), x)
#define PTR_FREE(x)		XFREE(x)
#define PTR_PASS(x)		({ __typeof__(x) __tmp = (x); (x) = NULL; __tmp; })

#define ARR_CLONE(T, x)		({ PTR_ALLOC(T, __tmp2); memcpy(*__tmp2, x, sizeof(T)); *PTR_PASS(__tmp2); })

extern int parse_cfl(_Complex float res[1], const char* str);
extern int parse_double(double res[1], const char* str);
extern int parse_long(bart_dim_t res[1], const char* str);
extern int parse_longlong(long long res[1], const char* str);
extern int parse_ulonglong(unsigned long long res[1], const char* str);
extern int parse_int(int res[1], const char* str);
#ifndef __cplusplus
extern noreturn void error(const char* str, ...) __attribute__((format(BART_PRINTF,1,2)));
#else
extern __attribute__((noreturn, format(BART_PRINTF,1,2))) void error(const char* str, ...);
#endif

// A dimension or stride handed to a library that takes an int (BLAS,
// LAPACK, cuFFT); an error rather than a truncation when it does not fit.
extern int checked_int(bart_dim_t x);


#ifdef USE_DWARF
#undef assert
#define assert(expr) do { if (!(expr)) error("Assertion '" #expr "' failed in %s:%d\n",  __FILE__, __LINE__); } while (0)
#endif

struct error_jumper_s {

	bool initialized;
	jmp_buf buf;
};

extern struct error_jumper_s error_jumper;	// FIXME should not be extern

extern int error_catcher(int fun(int argc, char* argv[__VLA(argc)]), int argc, char* argv[__VLA(argc)]);

extern int bart_printf(const char* fmt, ...) __attribute__((format(BART_PRINTF,1,2)));

extern void debug_print_bits(int dblevel, int D, bart_flags_t bitmask);

extern void print_dims(int D, const bart_dim_t dims[__VLA(D)]);
extern void debug_print_dims(int dblevel, int D, const bart_dim_t dims[__VLA(D)]);

#ifdef REDEFINE_PRINTF_FOR_TRACE
#define debug_print_dims(...) \
	debug_print_dims_trace(__FUNCTION__, __FILE__, __LINE__, __VA_ARGS__)
#endif

extern void debug_print_dims_trace(const char* func_name,
				   const char* file,
				   int line,
				   int dblevel,
				   int D,
				   const bart_dim_t dims[__VLA(D)]);

typedef CLOSURE_TYPE(int, (int a, int b)) quicksort_cmp_t;

/* avoid mingw-gcc warning: "bound argument 1 value -2147483648
 * is negative for a variable length array (...)",
 * triggered by quicksort usage in num/delayed.c:2052
 */
#ifdef _WIN32
extern void quicksort(int N, int ord[], quicksort_cmp_t cmp);
#else
extern void quicksort(int N, int ord[__VLA(N)], quicksort_cmp_t cmp);
#endif
#define quicksort(N, ord, cmp) quicksort(N, ord, CLOSURE(quicksort_cmp_t, cmp))

extern float quickselect(float *arr, int n, int k);
extern float quickselect_complex(_Complex float *arr, int n, int k);

extern void print_long(int D, const bart_dim_t arr[__VLA(D)]);
extern void print_float(int D, const float arr[__VLA(D)]);
extern void print_int(int D, const int arr[__VLA(D)]);
extern void print_complex(int D, const _Complex float arr[__VLA(D)]);

extern int bitcount(bart_flags_t flags);

extern const char* command_line;
extern char* stdin_command_line;
extern char* serialize_command_line(int argc, char* argv[__VLA(argc)]);
extern void save_command_line(int argc, char* argv[__VLA(argc)]);

extern bool safe_isnanf(float x);
extern bool safe_isfinite(float x);

extern bart_dim_t io_calc_size(int D, const bart_dim_t dims[__VLA(D?:1)], size_t size);

extern char* ptr_printf(const char* fmt, ...) __attribute__((format(BART_PRINTF,1,2)));
extern void ptr_append_printf(const char** prefix, const char* fmt, ...) __attribute__((format(BART_PRINTF,2,3)));
extern char* ptr_vprintf(const char* fmt, va_list ap);
extern char* ptr_print_dims(int D, const bart_dim_t dims[__VLA(D)]);

extern char* construct_filename(int D, const bart_dim_t loopdims[__VLA(D)], const bart_dim_t pos[__VLA(D)], const char* prefix, const char* ext);


#define DEG2RAD(d) ((d) * M_PI / 180.)
#define RAD2DEG(r) ((r) / M_PI * 180.)

#ifdef _WIN32
#define ffs(x)  __builtin_ffs((int)x)
#define ffsl(x) __builtin_ffsl(x)
#endif


#include "misc/cppwrap.h"

#endif // _MISC_H

