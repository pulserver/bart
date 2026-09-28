#include "misc/dimtypes.h"


struct nlop_s;
struct nlop_s* nlop_tenmul_create2(int N, const bart_dim_t dims[N], const bart_stride_t dstr[N],
		const bart_stride_t istr1[N], const bart_stride_t istr[N]);


struct nlop_s* nlop_tenmul_create(int N, const bart_dim_t odim[N], const bart_dim_t idim1[N], const bart_dim_t idims2[N]);

extern _Bool nlop_tenmul_der_available(const struct nlop_s* op, int index);
