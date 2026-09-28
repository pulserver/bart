
extern _Bool bart_use_gpu;
extern bart_flags_t bart_mpi_split_flags;
extern bart_flags_t bart_delayed_loop_flags;
extern bart_dim_t bart_delayed_loop_dims[16];
extern _Bool bart_delayed_computations;

extern void num_init(void);
extern void num_init_gpu_support(void);

extern void num_deinit_gpu(void);

extern void num_set_num_threads(int n);

extern void num_init_delayed(void);
extern void num_delayed_add_loop_dims(int dim);
