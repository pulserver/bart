
#include "misc/dimtypes.h"
#include "misc/mri.h"

extern void scc(const bart_dim_t out_dims[DIMS], complex float* out_data, const bart_dim_t caldims[DIMS], const complex float* cal_data);
extern void gcc(const bart_dim_t out_dims[DIMS], complex float* out_data, const bart_dim_t caldims[DIMS], const complex float* cal_data);
extern void ecc(const bart_dim_t out_dims[DIMS], complex float* out_data, const bart_dim_t caldims[DIMS], const complex float* cal_data);
extern void align_ro(const bart_dim_t dims[DIMS], complex float* odata, const complex float* idata);
extern void cc_align_mat(const bart_dim_t dims[DIMS], complex float* aligned, const complex float* in, const complex float* reference);

