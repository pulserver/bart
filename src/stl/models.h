
#ifndef _MODELS_H
#define _MODELS_H
#include "misc/dimtypes.h"

extern double* stl_internal_tetrahedron(bart_dim_t dims[3]);
extern double* stl_internal_hexahedron(bart_dim_t dims[3]);
extern double* stl_internal_icosahedron(bart_dim_t dims[3]);
extern double* stl_subdivide_model(long dims_out[3], const long dims_in[3], const double* model);
extern double* stl_multiple_subdivide_model(int sub_divs, long dims_out[3], const long dims_in[3], const double *model);

#endif