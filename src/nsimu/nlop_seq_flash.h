#ifndef __NLOP_SEQ_FLASH_H
#define __NLOP_SEQ_FLASH_H
#include "misc/dimtypes.h"

struct nlop_s;
struct sim_config_s;

struct flash_config_s {

	int N;
	int npixels;
	int nparams;
	int excitations;
	bool inv;

	float m0;
	float r1;
	float r2;
	float b1;
	float b0;

	float TI ;
	float TE;
	float TR;
	float flip_angle;

	float rf_duration;
};

extern struct flash_config_s flash_config_default;

extern struct list_s* flash_kern_ops_create(struct sim_config_s sim, struct flash_config_s config);
extern struct list_s* flash_ops_create(struct sim_config_s sim, struct flash_config_s config);
extern struct list_s* ir_flash_ops_create(struct sim_config_s sim, struct flash_config_s config);
extern struct nlop_s* nlop_phy_create(int N, const bart_dim_t map_dims[N], const bart_dim_t out_dims[N], struct flash_config_s config, struct sim_config_s sim);

#endif

