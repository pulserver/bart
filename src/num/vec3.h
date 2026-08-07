
typedef float vec3_t[3];
typedef double vec3d_t[3];


extern void vec3_saxpy(vec3_t dst, const vec3_t src1, float alpha, const vec3_t src2);
extern void vec3_sub(vec3_t dst, const vec3_t src1, const vec3_t src2);
extern void vec3_add(vec3_t dst, const vec3_t src1, const vec3_t src2);
extern void vec3_copy(vec3_t dst, const vec3_t src);
extern void vec3_clear(vec3_t dst);
extern float vec3_sdot(const vec3_t a, const vec3_t b);
extern float vec3_norm(const vec3_t x);
extern void vec3_rot(vec3_t dst, const vec3_t src1, const vec3_t src2);
extern void vec3_smul(vec3_t dst, const vec3_t src, float alpha);

extern double vec3d_sdot(const vec3d_t x, const vec3d_t y);
extern double vec3d_norm(const vec3d_t x);
extern void vec3d_saxpy(vec3d_t o, const vec3d_t x, const double a, const vec3d_t y);
extern void vec3d_smul(vec3d_t o, const vec3d_t x, double a);
extern void vec3d_clear(vec3d_t x);
extern void vec3d_crossproduct(vec3d_t o, const vec3d_t v0, const vec3d_t v1);
extern void vec3d_rotax(vec3d_t o, const double theta, const vec3d_t ax, const vec3d_t x);
extern void vec3d_copy(vec3d_t o, const vec3d_t x);
extern double vec3d_angle(const vec3d_t x, const vec3d_t y);
extern void vec3d_rot(vec3d_t dst, const vec3d_t src1, const vec3d_t src2);
