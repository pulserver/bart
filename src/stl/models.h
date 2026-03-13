
#ifndef _MODELS_H
#define _MODELS_H

extern double* stl_internal_tetrahedron(long dims[3]);
extern double* stl_internal_hexahedron(long dims[3]);
extern double* stl_internal_icosahedron(long dims[3]);
extern double* stl_subdivide_model(long dims_out[3], const long dims_in[3], const double* model);
extern double* stl_multiple_subdivide_model(int sub_divs, long dims_out[3], const long dims_in[3], const double *model);

#endif