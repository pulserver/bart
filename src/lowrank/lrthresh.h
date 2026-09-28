
#include "misc/dimtypes.h"
#include <complex.h>


#include "misc/mri.h"

#ifndef MAX_LEV
#define MAX_LEV 100
#endif

struct operator_p_s;


// Low rank thresholding for arbitrary block sizes
extern const struct operator_p_s* lrthresh_create(const bart_dim_t dims_lev[DIMS], bool randshift, bart_flags_t mflags, const bart_dim_t blkdims[MAX_LEV][DIMS], float lambda, bool noise, int remove_mean, bool overlapping_blocks);

// Returns nuclear norm using lrthresh operator
extern float lrnucnorm(const struct operator_p_s* op, const complex float* src);

// Generates multiscale block sizes
extern int multilr_blkdims(bart_dim_t blkdims[MAX_LEV][DIMS], bart_flags_t flags, const bart_dim_t dims[DIMS], int blkskip, int initblk);

// Generates locally low rank block size
extern int llr_blkdims(bart_dim_t blkdims[MAX_LEV][DIMS], bart_flags_t flags, const bart_dim_t dims[DIMS], int llrblk);

// Generates low rank plus sparse block size
extern int ls_blkdims(bart_dim_t blkdims[MAX_LEV][DIMS], const bart_dim_t dims[DIMS]);


extern void add_lrnoiseblk(int* level, bart_dim_t blkdims[MAX_LEV][DIMS], const bart_dim_t dims[DIMS]);

// Return the regularization parameter
extern float get_lrthresh_lambda(const struct operator_p_s* o);
