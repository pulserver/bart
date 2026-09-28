
#include "misc/nested.h"

typedef CLOSURE_TYPE(void, (void)) bench_f;

void run_bench(bart_dim_t rounds, bool print, bool sync_gpu, bench_f fun);

