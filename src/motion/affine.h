#include "misc/dimtypes.h"

struct nlop_s;

const struct nlop_s* nlop_affine_chain_FF(const struct nlop_s* A, const struct nlop_s* B);
const struct nlop_s* nlop_affine_prepend_FF(const struct nlop_s* A, _Complex float* B);
const struct nlop_s* nlop_affine_append_FF(_Complex float* A, const struct nlop_s* B);
const struct nlop_s* nlop_affine_to_grid_F(const struct nlop_s* affine, const bart_dim_t sdims[3], const bart_dim_t mdims[3]);

void affine_init_id(_Complex float* dst);

const struct nlop_s* nlop_affine_translation_2D(void);
const struct nlop_s* nlop_affine_translation_3D(void);

const struct nlop_s* nlop_affine_rotation_2D(void);
const struct nlop_s* nlop_affine_rotation_3D(void);

const struct nlop_s* nlop_affine_rigid_2D(void);
const struct nlop_s* nlop_affine_rigid_3D(void);

const struct nlop_s* nlop_affine_2D(void);
const struct nlop_s* nlop_affine_3D(void);

extern void affine_debug(int dl, const _Complex float* A);

extern void affine_interpolate(int ord, const _Complex float* affine, const bart_dim_t _odims[3], _Complex float* dst, const bart_dim_t _idims[3], const _Complex float* src);

extern const struct nlop_s* nlop_affine_compute_pos(int dim, int N, const bart_dim_t sdims[N], const bart_dim_t mdims[N], const struct nlop_s* affine);

extern void affine_reg(_Bool gpu, _Bool cubic, _Complex float* affine, const struct nlop_s* trafo, bart_dim_t sdims[3], const _Complex float* img_static, const _Complex float* msk_static, bart_dim_t mdims[3], const _Complex float* img_moving, const _Complex float* msk_moving,
			int N, float sigma[N], float factor[N]);


