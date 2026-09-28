
#ifndef _LINOPS_SOMEOPS_H
#define _LINOPS_SOMEOPS_H

#include "misc/dimtypes.h"
#include <stdbool.h>

#include "misc/cppwrap.h"

extern struct linop_s* linop_cdiag_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags, const _Complex float* diag);
extern struct linop_s* linop_rdiag_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags, const _Complex float* diag);
extern void linop_gdiag_set_diag(const struct linop_s* lop, int N, const bart_dim_t ddims[__VLA(N)], const _Complex float* diag);
extern void linop_gdiag_set_diag_F(const struct linop_s* lop, int N, const bart_dim_t ddims[__VLA(N)], const _Complex float* diag);
extern void linop_gdiag_set_diag_ref(const struct linop_s* lop, int N, const bart_dim_t ddims[__VLA(N)], const _Complex float* diag);

extern struct linop_s* linop_scale_create(int N, const bart_dim_t dims[N], const _Complex float scale);
extern struct linop_s* linop_zconj_create(int N, const bart_dim_t dims[N]);
extern struct linop_s* linop_zreal_create(int N, const bart_dim_t dims[N]);
extern struct linop_s* linop_flip_create(int N, const bart_dim_t dims[N], bart_flags_t flags);

extern struct linop_s* linop_identity_create(int N, const bart_dim_t dims[__VLA(N)]);
extern _Bool linop_is_identity(const struct linop_s* lop);

extern struct linop_s* linop_copy_block_create(int N, const bart_dim_t pos[__VLA(N)], const bart_dim_t odims[__VLA(N)], const bart_dim_t idims[__VLA(N)]);
extern struct linop_s* linop_resize_create(int N, const bart_dim_t out_dims[__VLA(N)], const bart_dim_t in_dims[__VLA(N)]);	// deprecated
extern struct linop_s* linop_resize_center_create(int N, const bart_dim_t out_dims[__VLA(N)], const bart_dim_t in_dims[__VLA(N)]);
extern struct linop_s* linop_expand_create(int N, const bart_dim_t out_dims[__VLA(N)], const bart_dim_t in_dims[__VLA(N)]);
extern struct linop_s* linop_reshape_create(int A, const bart_dim_t out_dims[__VLA(A)], int B, const bart_dim_t in_dims[__VLA(B)]);
extern struct linop_s* linop_reshape2_create(int N, bart_flags_t flags, const bart_dim_t out_dims[__VLA(N)], const bart_dim_t in_dims[__VLA(N)]);
extern struct linop_s* linop_extract_create(int N, const bart_dim_t pos[N], const bart_dim_t out_dims[N], const bart_dim_t in_dims[N]);
extern struct linop_s* linop_permute_create(int N, const int order[__VLA(N)], const bart_dim_t idims[N]);
extern struct linop_s* linop_transpose_create(int N, int a, int b, const bart_dim_t dims[N]);

extern struct linop_s* linop_slice_create(int N, bart_flags_t flags, const bart_dim_t pos[__VLA(N)], const bart_dim_t dims[__VLA(N)]);
extern struct linop_s* linop_slice_one_create(int N, int idx, bart_dim_t pos, const bart_dim_t dims[__VLA(N)]);

extern struct linop_s* linop_add_strided_create(int N, const bart_dim_t dims[__VLA(N)], const bart_stride_t ostrs[__VLA(N)], const bart_stride_t istrs[__VLA(N)],
					        int OO, const bart_dim_t odims[__VLA(OO)], int II, const bart_dim_t idims[__VLA(II)]);
extern struct linop_s* linop_hankelization_create(int N, const bart_dim_t dims[__VLA(N)], int dim, int window_dim, int window_size);


extern struct linop_s* linop_fft_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags);
extern struct linop_s* linop_ifft_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags);
extern struct linop_s* linop_fftc_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags);
extern struct linop_s* linop_ifftc_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flags);

extern struct linop_s* linop_fft_generic_create(int N, const bart_dim_t dims[N], bart_flags_t flags, bart_flags_t center_flags, bart_flags_t unitary_flags, bart_flags_t pre_flag, const _Complex float* pre_diag, bart_flags_t post_flag, const _Complex float* post_diag);
extern struct linop_s* linop_ifft_generic_create(int N, const bart_dim_t dims[N], bart_flags_t flags, bart_flags_t center_flags, bart_flags_t unitary_flags, bart_flags_t pre_flag, const _Complex float* pre_diag, bart_flags_t post_flag, const _Complex float* post_diag);

extern struct linop_s* linop_cdf97_create(int N, const bart_dim_t dims[__VLA(N)], bart_flags_t flag);

#ifndef _PADD_ENUMS
#define _PADD_ENUMS
enum PADDING { PAD_VALID, PAD_SAME, PAD_CYCLIC, PAD_SYMMETRIC, PAD_REFLECT, PAD_CAUSAL };
#endif
extern struct linop_s* linop_padding_create_onedim(int N, const bart_dim_t dims[N], enum PADDING pad_type, int pad_dim, bart_dim_t pad_for, bart_dim_t pad_after);
extern struct linop_s* linop_padding_create(int N, const bart_dim_t dims[N], enum PADDING pad_type, bart_dim_t pad_for[N], bart_dim_t pad_after[N]);

extern struct linop_s* linop_shift_create(int N, const bart_dim_t dims[__VLA(N)], int shift_dim, bart_dim_t shift, enum PADDING pad_type);

#ifndef _CONV_ENUMS
#define _CONV_ENUMS
enum conv_mode { CONV_SYMMETRIC, CONV_CAUSAL, CONV_ANTICAUSAL };
enum conv_type { CONV_CYCLIC, CONV_TRUNCATED, CONV_VALID, CONV_EXTENDED };
#endif

extern struct linop_s* linop_conv_create(int N, bart_flags_t flags, enum conv_type ctype, enum conv_mode cmode, const bart_dim_t odims[__VLA(N)],
                const bart_dim_t idims1[__VLA(N)], const bart_dim_t idims2[__VLA(N)], const _Complex float* src2);

extern struct linop_s* linop_conv_gaussian_create(int N, enum conv_type ctype, const bart_dim_t dims[__VLA(N)], const float sigma[__VLA(N)]);

extern struct linop_s* linop_matrix_create(int N, const bart_dim_t out_dims[__VLA(N)], const bart_dim_t in_dims[__VLA(N)], const bart_dim_t matrix_dims[__VLA(N)], const _Complex float* matrix);
extern struct linop_s* linop_matrix_altcreate(int N, const bart_dim_t out_dims[__VLA(N)], const bart_dim_t in_dims[__VLA(N)], int T_dim, int K_dim, const _Complex float* matrix);


extern struct linop_s* linop_matrix_chain(const struct linop_s* a, const struct linop_s* b);

extern struct linop_s* linop_hadamard_create(int N, const bart_dim_t in_dims[__VLA(N)], int hadamard_dim);

#include "misc/cppwrap.h"
#endif // _LINOPS_SOMEOPS_H
