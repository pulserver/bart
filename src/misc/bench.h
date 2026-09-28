
#include "misc/dimtypes.h"
#include "misc/nested.h"
#include <stdbool.h>

typedef void CLOSURE_TYPE(bench_f)(void);

void run_bench(bart_dim_t rounds, bool print, bool sync_gpu, bench_f fun);

