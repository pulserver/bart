
#ifdef __cplusplus
extern "C" {
#endif

#include "misc/dimtypes.h"
#include "misc/mri.h"

struct linop_s;

extern struct linop_s* linop_sampling_create(const bart_dim_t dims[DIMS], const bart_dim_t pat_dims[DIMS], const _Complex float* pattern);

extern struct linop_s* sense_init(bart_flags_t shared_img_flags, const bart_dim_t max_dims[DIMS], bart_flags_t sens_flags, const _Complex float* sens);
extern struct linop_s* maps_create(bart_flags_t shared_img_flags, const bart_dim_t max_dims[DIMS], 
			bart_flags_t sens_flags, const _Complex float* sens);
extern struct linop_s* maps2_create(const bart_dim_t coilim_dims[DIMS], const bart_dim_t maps_dims[DIMS], const bart_dim_t img_dims[DIMS], const _Complex float* maps);


#ifdef __cplusplus
}
#endif


