

struct nlop_s;
struct moba_conf_s;

extern float read_relax(float tr, float angle);

extern struct nlop_s* nlop_T1_phy_create(int N, const long out_dims[N], const long in_dims[N],
                const long TI_dims[N], const complex float* TI,  const struct moba_conf_s* config);

