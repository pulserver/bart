
#ifndef __NLOP_SEQ_H
#define __NLOP_SEQ_H

#ifndef DIMS
#define DIMS 16
#endif

struct pulse;
struct nlop_s;
struct rf_shape;

#define R1_IDX 0
#define R2_IDX 1
#define B1_IDX 2
#define B0_IDX 3

struct sim_config_s {

	int N;
	long mdims[DIMS];
	long pdims[DIMS];

	int MO_DIM;
	int MI_DIM;
	int PI_DIM;

	unsigned long spatial_flags;
	float voxel_size[3];

	float tol;
	bool hard_pulse_sim;
};

extern void sim_config_set_dims(struct sim_config_s* sim, int N, const long dims[N], int Nspins);
extern void sim_config_debug(int dl, struct sim_config_s* sim);

extern struct sim_config_s sim_config_default_cpu;
extern struct sim_config_s sim_config_default_gpu;

extern const struct nlop_s* nlop_pulse_create(struct sim_config_s sim, const struct pulse* pulse, float phase, float grad[3]);
extern const struct nlop_s* nlop_pulse_shape_create(struct sim_config_s sim, struct rf_shape* shape, float phase, float grad[3]);
extern const struct nlop_s* nlop_relax_create(struct sim_config_s sim, float t, float grad[3]);
extern const struct nlop_s* nlop_spoile_create(struct sim_config_s sim);
extern const struct nlop_s* nlop_adc_create(struct sim_config_s sim, long index, unsigned long sflags, float phase);

extern const struct nlop_s* nlop_phase_wrap_F(struct sim_config_s sim, const struct nlop_s* nlop, float phase);

extern const struct nlop_s* nlop_rotx_create(struct sim_config_s sim, float angle);
extern const struct nlop_s* nlop_roty_create(struct sim_config_s sim, float angle);
extern const struct nlop_s* nlop_rotz_create(struct sim_config_s sim, float angle);
extern const struct nlop_s* nlop_hard_pulse_create(struct sim_config_s sim, _Bool b1, float angle, float phase);

struct list_s;
extern const struct nlop_s* nlop_simu_jacobian_chain_create(struct sim_config_s sim, struct list_s* nlops);
extern const struct nlop_s* nlop_seq_from_blocks_create_F(struct sim_config_s sim, struct list_s* nlops);
extern const struct nlop_s* nlop_seq_from_blocks_jac_create_F(struct sim_config_s sim, struct list_s* nlops);

extern const struct nlop_s* sim_nlop_set_init(struct sim_config_s sim, const struct nlop_s* nlop);

struct stm_s;
extern struct stm_s* stm_create(struct sim_config_s sim, const struct nlop_s* nlop);
extern void stm_free(struct stm_s* x);
extern struct nlop_s* nlop_stm_create(struct stm_s* x);

extern const struct nlop_s* nlop_simu_stack_create(struct sim_config_s sim, const struct nlop_s* nlop, int stack_dim);

#endif
