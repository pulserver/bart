
#include "misc/dimtypes.h"
#include "misc/cppwrap.h"


struct multiplace_array_s;
extern void multiplace_free(const struct multiplace_array_s* ptr);
extern const void* multiplace_read(struct multiplace_array_s* ptr, const void* ref);
extern struct multiplace_array_s* multiplace_move2(int D, const bart_dim_t dimensions[__VLA(D)], const bart_stride_t strides[__VLA(D)], size_t size, const void* ptr);
extern struct multiplace_array_s* multiplace_move(int D, const bart_dim_t dimensions[__VLA(D)], size_t size, const void* ptr);
extern struct multiplace_array_s* multiplace_move_F(int D, const bart_dim_t dimensions[__VLA(D)], size_t size, const void* ptr);
extern struct multiplace_array_s* multiplace_move_wrapper(int D, const bart_dim_t dimensions[__VLA(D)], size_t size, const void* ptr);


#include "misc/cppwrap.h"
