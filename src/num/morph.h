
#include "misc/dimtypes.h"
#include "num/conv.h"

extern _Complex float* md_structuring_element_cube(int N, bart_dim_t dims[N], int radius, bart_flags_t flags, const void* ref);
extern _Complex float* md_structuring_element_ball(int N, bart_dim_t dims[N], int radius, bart_flags_t flags, const void* ref);
extern _Complex float* md_structuring_element_cross(int N, bart_dim_t dims[N], int radius, bart_flags_t flags, const void* ref);

extern void md_erosion(int D, const bart_dim_t mask_dims[D], _Complex float* mask, const bart_dim_t dims[D], _Complex float* out, const _Complex float* in, enum conv_type ctype);
extern void md_dilation(int D, const bart_dim_t mask_dims[D], _Complex float* mask, const bart_dim_t dims[D], _Complex float* out, const _Complex float* in, enum conv_type ctype);
extern void md_opening(int D, const bart_dim_t mask_dims[D], _Complex float* mask, const bart_dim_t dims[D], _Complex float* out, const _Complex float* in, enum conv_type ctype);
extern void md_closing(int D, const bart_dim_t mask_dims[D], _Complex float* mask, const bart_dim_t dims[D], _Complex float* out, const _Complex float* in, enum conv_type ctype);

extern _Complex float* md_label_simple_connection(int N, bart_dim_t dims[N], float radius, bart_flags_t flags);
extern bart_dim_t md_label(int N, const bart_dim_t dims[N], _Complex float* labels, const _Complex float* src, const bart_dim_t sdims[N], const _Complex float* structure);

void md_center_of_mass(int N_labels, int N, float com[N_labels][N], const bart_dim_t dims[N], const _Complex float* labels, const _Complex float* wgh);

void md_thinning_3D(int N, const bart_dim_t dims[N], _Complex float* dst, const _Complex float* src, const _Complex float* keep);
