#include "misc/dimtypes.h"
#include <stddef.h>

#include "misc/cppwrap.h"

enum delayed_md_fun_type {
	delayed_op_type_z3op,
	delayed_op_type_3op,
	delayed_op_type_z3opd,
	delayed_op_type_3opd,
	delayed_op_type_z2op,
	delayed_op_type_2op,
	delayed_op_type_z2opd,
	delayed_op_type_2opd,
	delayed_op_type_z2opf,
	delayed_op_type_2opf,
};

struct list_s;
struct vptr_fun_data_s;
typedef void (*vptr_fun_t)(struct vptr_fun_data_s* data, int N, int D, const bart_dim_t* dims[__VLA(N)], const bart_stride_t* strs[__VLA(N)], void* args[__VLA(N)]);

extern void exec_vptr_fun_delayed(vptr_fun_t fun, struct vptr_fun_data_s* data, int N, int D, bart_flags_t lflags, bart_flags_t wflags, bart_flags_t rflags, const bart_dim_t* dims[__VLA(N)], const bart_stride_t* strs[__VLA(N)], void* ptr[__VLA(N)], size_t sizes[__VLA(N)], bool resolve);

extern void debug_delayed_queue(int dl, struct list_s* ops_queue, bool nested);
extern void delayed_compute(const void* ptr);

extern bool is_delayed(const void* ptr);
extern void delayed_alloc(const void* ptr, int N, const bart_dim_t dims[__VLA(N)], size_t size);
extern void delayed_free(const void* ptr, int N, const bart_dim_t dims[__VLA(N)], size_t size);

extern bool delayed_queue_copy(int D, const bart_dim_t dim[__VLA(D)], const bart_stride_t ostr[__VLA(D)], void* optr, const bart_stride_t istr[__VLA(D)], const void* iptr, size_t size);
extern bool delayed_queue_circ_shift(int D, const bart_dim_t dimensions[D], const bart_dim_t center[D], const bart_stride_t str1[D], void* dst, const bart_stride_t str2[D], const void* src, size_t size);
extern bool delayed_queue_clear(int D, const bart_dim_t dim[__VLA(D)], const bart_stride_t str[__VLA(D)], void* ptr, size_t size);
extern bool delayed_queue_make_op(enum delayed_md_fun_type type, size_t offset, int D, const bart_dim_t dim[__VLA(D)], int N, const bart_stride_t* strs[__VLA(N)], const void* ptr[__VLA(N)], const size_t sizes[__VLA(N)]);

//extern for testing, dont use!

struct queue_s;
struct delayed_op_s;

extern struct queue_s* get_global_queue(void);
extern void release_global_queue(struct queue_s* queue);
extern void delayed_queue_exec(struct queue_s* queue);
extern void delayed_compute_debug(const char* name);

extern void queue_set_compute(struct queue_s*, bool compute);
extern struct list_s* get_delayed_op_list(struct queue_s* queue);
extern void delayed_optimize_queue(struct list_s* ops_queue);
extern void delayed_optimize_queue_looping(struct list_s* ops_queue);

extern bool delayed_op_is_alloc(const struct delayed_op_s* op);
extern bool delayed_op_is_free(const struct delayed_op_s* op);
extern bool delayed_op_is_copy(const struct delayed_op_s* op);
extern bool delayed_op_is_clear(const struct delayed_op_s* op);
extern bool delayed_op_is_chain(const struct delayed_op_s* op);

extern void debug_mpeak_queue(int dl, struct list_s* ops_queue, bool node);
extern bart_dim_t compute_mpeak(struct list_s* ops_queue, bool node);
extern bart_dim_t compute_mchange(struct list_s* ops_queue, bool node);


#include "misc/cppwrap.h"

