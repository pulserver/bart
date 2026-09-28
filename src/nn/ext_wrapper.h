#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

struct nlop_s;
extern const struct nlop_s* nlop_external_graph_create(const char* path, int OO, const int DO[__VLA2(OO)], const bart_dim_t* odims[__VLA2(OO)], int II, const int DI[__VLA(II)], const bart_dim_t* idims[__VLA(II)], bool init_gpu, const char* tf_key);

#include "misc/cppwrap.h"