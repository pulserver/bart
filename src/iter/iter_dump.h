
#ifndef _ITER_DUMP_H
#define _ITER_DUMP_H

#include "misc/dimtypes.h"

struct typeid_s;
struct nlop_s;

struct iter_dump_s;
typedef void (*iter_dump_fun_t)(const struct iter_dump_s* data, bart_dim_t epoch, bart_dim_t NI, const float* x[NI]);
typedef void (*iter_dump_free_t)(const struct iter_dump_s* data);

struct iter_dump_s {

	const struct typeid_s* TYPEID;
	iter_dump_fun_t fun;
	iter_dump_free_t free;
	const char* base_filename;
};

const struct iter_dump_s* iter_dump_default_create(const char* base_filename, bart_dim_t save_mod, bart_dim_t NI, _Bool save_flag[NI], int D[NI], const bart_dim_t* dims[NI]);

void iter_dump(const struct iter_dump_s* data, bart_dim_t epoch, bart_dim_t NI, const float* x[const NI]);
void iter_dump_free(const struct iter_dump_s* data);


#endif // _ITER_DUMP_H
