
#ifdef USE_GPU
#include "misc/dimtypes.h"
#include "num/gpu_compat_runtime.h"

extern int cuda_get_max_threads(const void* func);

extern dim3 getBlockSize3(const bart_dim_t dims[3], int threads);
extern dim3 getGridSize3(const bart_dim_t dims[3], int threads);

extern dim3 getBlockSize3(const bart_dim_t dims[3], const void* func);
extern dim3 getGridSize3(const bart_dim_t dims[3], const void* func);

extern dim3 getBlockSize(bart_dim_t N, int threads);
extern dim3 getGridSize(bart_dim_t N, int threads);

extern dim3 getBlockSize(bart_dim_t N, const void* func);
extern dim3 getGridSize(bart_dim_t N, const void* func);

#define getBlockSize3S(dims, func) ({				\
	static int _threads = -1; 				\
	static const void* _func = NULL;			\
	const bart_dim_t _dims[3] = { dims[0], dims[1], dims[2] };	\
	if (-1 == _threads) {					\
		_func = (const void*)func;			\
		_threads = cuda_get_max_threads((_func));	\
	} else {						\
		assert(_func == (const void*)func);		\
	}							\
	getBlockSize3(_dims, _threads); })

#define getGridSize3S(dims, func) ({				\
	const bart_dim_t _dims[3] = { dims[0], dims[1], dims[2] };	\
	static int _threads = -1; 				\
	static const void* _func = NULL;			\
	if (-1 == _threads) {					\
		_func = (const void*)func;			\
		_threads = cuda_get_max_threads((_func));	\
	} else {						\
		assert(_func == (const void*)func);		\
	}							\
	getGridSize3(_dims, _threads); })

#endif
