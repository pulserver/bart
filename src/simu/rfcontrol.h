
struct puls_opt_pars {
	// Space discretization
	double a; 	// Domain border in m
	double z; 	// Half slice thickness in m
	int Nx; 	// Total number of spatial points
	double* xdis; 	// Spatial running variable
	float dx; 	// Spatial grid size

	// Time discretization
	float T; 	// Optimization time in ms
	int Nt; 	// Total number of temporal points
	float* tdis; 	// Temporal running variable
	float dt; 	// Temporal grid size
	int Nu; 	// Number of temporal control points

	// Model parameters
	float gamma; 	// Gyromagnetic ratio in MHz/T (= 42.57*2*pi)
	float T1; 	// Longitudinal relaxation time in ms
	float T2; 	// Transverse relaxation time in ms
	float r1;	// Longitudinal relaxation rate
	float r2;	// Transverse relaxation rate
	float phi; 	// Flip angle in degrees
	float B0; 	// Static magnetic field in mT
	float* M0; 	// Initial magnetization
	float M0c; 	// Normalized equilibrium magnetization
	float B1c; 	// Weighting factor for the RF amplitude (u * 1e3 * B1c) in muT
	float B1;	// Effective B1 field in mT
	float Gz; 	// Weighting factor for the z-gradient in mT
	int relax; 	// 0 = no relaxation, 1 = with relaxation

	float* u; 	// RF initial guess
	float* v;
	float* w;

	float alpha; 	// Control costs for u (SAR)

	float* Md; 	// Desired magnetization
	float* inslice;	// One slice in center
	float* outslice;
};

struct tr_pars { // TR-CG-Newton parameters
	int maxit;	// Maximum number of TR Newton iterations
	float reltol; 	// Relative tolerance for gradient norm in Newton
	float abstol; 	// Absolute tolerance for gradient norm in Newton
	float rho; 	// Initial trust-region radius
	float maxrad;	// Maximum trust-region radius
	float sig1;	// Parameter for trust-region update: decrease trad if dJa/dJm < sig1
	float sig2;	// Parameter for trust-region update: do not change if sig1 < dJa/dJm < sig2
	float sig3;	// Parameter for trust-region update: increase trad if dJa/dJm > sig3
	float q;	// Factor for radius change
	float cgtol;	// Desired reduction of residual in CG 
	int cgits;	// Maximum number of CG iterations
};

struct Xk_struct { 	// Data structure to store intermediate results for Hessian evaluation
	float* N; 	// State    [Nx][Nt - 1][3]
	float* P; 	// Adjoint  [Nx][Nt - 1][3]
	float* u; 	// Control  [Nt - 1]
};

void cn_bloch(int Nx, int Nt, float M[Nx][Nt][3], const float u[Nt - 1], const struct puls_opt_pars p);

void cn_adjoint(int Nx, int Nt, float P[Nx][Nt - 1][3], const float u[Nt - 1], 
	const float PT[Nx][3], const struct puls_opt_pars p);

float objfun(int Nu, float G[Nu], struct Xk_struct* Xk, const struct puls_opt_pars p, const float iu[Nu]);

void apply_Hess(int N, float Hdu[N], const struct puls_opt_pars p, const struct Xk_struct* Xk, const float idu[N]);

typedef CLOSURE_TYPE(void, (int N, float Hp[N], const float p[N])) tr_cg_fun1_t;
typedef CLOSURE_TYPE(float, (int N, const float x[N], const float y[N])) tr_cg_dot_t;

void tr_cg(int iter, float tol, float trad, float dt, float magic_power,
	int Nu, float du[Nu], const float g[Nu], tr_cg_fun1_t H_func);

void tr_newton(int Nu, float u[Nu], const struct puls_opt_pars p, const struct tr_pars np, float* u0);

