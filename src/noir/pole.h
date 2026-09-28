#include "misc/dimtypes.h"


#include "num/lineseg.h"

struct pole_config_s {

	float diameter;
	float closing;
	int segments;
	float thresh;
	bart_flags_t avg_flag;
	int normal;
	bool espirit;
	float tol;
	float osx;
};

extern struct pole_config_s pole_config_default;

extern void compute_curl_map(struct pole_config_s conf, int N, const bart_dim_t curl_dims[N], int dim, _Complex float* curl_map, const bart_dim_t sens_dims[N], const _Complex float* sens);
extern void compute_curl_weighting(struct pole_config_s conf, int N, const bart_dim_t curl_dims[N], int dim, _Complex float* wgh_map, const bart_dim_t cim_dims[N], const _Complex float* cim);
extern void average_curl_map(int N, const bart_dim_t pmap_dims[N], _Complex float* red_curl_map, const bart_dim_t curl_dims[N], int dim, _Complex float* curl_map, _Complex float* wgh_map);

extern void sample_phase_pole_2D(int N, const bart_dim_t dims[N], _Complex float* dst, int D, const float pos[D][2][3]);
extern void sample_phase_pole_3D(int N, const bart_dim_t dims[N], _Complex float* dst, int D, const float pos[D][2][3], float tol);

struct lseg_s extract_phase_poles_2D(struct pole_config_s conf, int N, const long dims[N], const _Complex float* curl_map);
struct lseg_s extract_phase_poles_3D(struct pole_config_s conf, int N, const bart_dim_t dims[N], const _Complex float* curl_map, const bart_dim_t sens_dims[N], const _Complex float* sens);

extern bool phase_pole_correction(struct pole_config_s conf, int N, const bart_dim_t pmap_dims[N], _Complex float* phase, const bart_dim_t sens_dims[N], const _Complex float* sens);
extern bool phase_pole_correction_loop(struct pole_config_s conf, int N, bart_flags_t lflags, const bart_dim_t pmap_dims[N], _Complex float* phase, const bart_dim_t sens_dims[N], const _Complex float* sens);

extern void phase_pole_normalize(int N, const bart_dim_t pdims[N], _Complex float* phase, const bart_dim_t idims[N], const _Complex float* image);

extern float integrate_phase(int M, const float pos[M][3], int N, const float r[N][2][3], float tol);
