
#ifndef _MODELS_H
#define _MODELS_H
#include "misc/dimtypes.h"

extern double* stl_internal_tetrahedron(bart_dim_t dims[3]);
extern double* stl_internal_hexahedron(bart_dim_t dims[3]);
extern double* stl_internal_icosahedron(bart_dim_t dims[3]);
extern double* stl_subdivide_model(bart_dim_t dims_out[3], const bart_dim_t dims_in[3], const double* model);
extern double* stl_multiple_subdivide_model(int sub_divs, bart_dim_t dims_out[3], const bart_dim_t dims_in[3], const double *model);

#endif