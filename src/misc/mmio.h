
#include "misc/cppwrap.h"

extern _Bool mmio_file_locking;

extern _Bool stream_create_binary_outputs;

#include <stddef.h>
extern _Bool mpi_shared_files;
extern bart_flags_t cfl_loop_rand_flags;
extern _Bool strided_cfl_loop;
extern void init_cfl_loop_desc(int D, const bart_dim_t loop_dims[__VLA(D)], bart_dim_t start_dims[__VLA(D)], bart_flags_t flags, int omp_threads, int index);
extern void set_cfl_loop_index(bart_dim_t index);
extern bart_dim_t get_cfl_loop_index(void);
extern _Bool cfl_loop_desc_active(void);
extern void cfl_loop_desc_set_inactive(void);
extern _Bool cfl_loop_omp(void);
extern int cfl_loop_worker_id(void);
extern int cfl_loop_num_workers(void);
extern bart_dim_t cfl_loop_desc_total(void);
extern bart_dim_t calc_size_cfl_loop(int D, const bart_dim_t dims[__VLA(D)], size_t size);
extern bart_flags_t cfl_loop_get_flags(void);
extern int cfl_loop_get_rank(void);
extern void cfl_loop_get_dims(int D, bart_dim_t dims[__VLA(D)]);
extern void cfl_loop_get_pos(int D, bart_dim_t pos[__VLA(D)]);

extern void* private_raw(size_t* size, const char* name);
extern void unmap_raw(const void* data, size_t size);

#ifndef MEMONLY_CFL
extern _Complex float* shared_cfl(int D, const bart_dim_t dims[__VLA(D)], const char* name);
extern _Complex float* private_cfl(int D, const bart_dim_t dims[__VLA(D)], const char* name);
#endif /* !MEMONLY_CFL */
extern void unmap_cfl(int D, const bart_dim_t dims[__VLA(D)], const _Complex float* x);
extern void unmap_shared_cfl(int D, const bart_dim_t dims[D], const _Complex float* x);

extern _Complex float* anon_cfl(const char* name, int D, const bart_dim_t dims[__VLA(D)]);
extern _Complex float* create_cfl(const char* name, int D, const bart_dim_t dimensions[__VLA(D)]);
extern _Complex float* create_async_cfl(const char* name, const bart_flags_t flags, int D, const bart_dim_t dimensions[__VLA(D)]);
extern _Complex float* create_cfl_sameplace(const char* name, int D, const bart_dim_t dimensions[__VLA(D)], const void* ref);
extern _Complex float* load_cfl(const char* name, int D, bart_dim_t dimensions[__VLA(D)]);
extern _Complex float* load_async_cfl(const char* name, int D, bart_dim_t dimensions[__VLA(D)]);
extern _Complex float* load_shared_cfl(const char* name, int D, bart_dim_t dimensions[__VLA(D)]);
extern _Complex float* load_cfl_sameplace(const char* name, int D, bart_dim_t dimensions[__VLA(D)], const void* ref);

extern void create_multi_cfl(const char* name, int N, int D[__VLA(N)], const bart_dim_t* dimensions[__VLA(N)], _Complex float* args[__VLA(N)]);
extern int load_multi_cfl(const char* name, int N_max, int D_max, int D[__VLA(N_max)], bart_dim_t dimensions[__VLA(N_max)][D_max], _Complex float* args[__VLA(N_max)]);
extern void unmap_multi_cfl(int N, int D[__VLA(N)], const bart_dim_t* dimensions[__VLA(N)], _Complex float* args[__VLA(N)]);

extern float* create_coo(const char* name, int D, const bart_dim_t dimensions[__VLA(D)]);
extern float* load_coo(const char* name, int D, bart_dim_t dimensions[__VLA(D)]);
extern _Complex float* create_zcoo(const char* name, int D, const bart_dim_t dimensions[__VLA(D)]);
extern _Complex float* load_zcoo(const char* name, int D, bart_dim_t dimensions[__VLA(D)]);
extern _Complex float* create_zra(const char* name, int D, const bart_dim_t dims[__VLA(D)]);
extern _Complex float* load_zra(const char* name, int D, bart_dim_t dims[__VLA(D)]);
extern _Complex float* create_zshm(const char* name, int D, const bart_dim_t dims[__VLA(D)]);
extern _Complex float* load_zshm(const char* name, int D, bart_dim_t dims[__VLA(D)]);

#ifdef __EMSCRIPTEN__
extern int wasm_fd_offset;
void wasm_close_fds(void);
#endif

#include "misc/cppwrap.h"
