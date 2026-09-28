
#include "misc/dimtypes.h"
#include <complex.h>

struct bin_conf_s {

	int n_resp;
	int n_card;
	int mavg_window;
	int mavg_window_card;
	int cluster_dim;

	bart_dim_t resp_labels_idx[2];
	bart_dim_t card_labels_idx[2];

	const char* card_out;

	float offset_angle[2];

	bool amplitude;

};

extern const struct bin_conf_s bin_defaults;

extern int bin_quadrature(const bart_dim_t bins_dims[DIMS], float* bins,
			const bart_dim_t labels_dims[DIMS], complex float* labels,
			const struct bin_conf_s conf);
	
