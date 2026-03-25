/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <stdlib.h>

#include "misc/nested.h"
#include "misc/debug.h"
#include "num/linalg.h"
#include "num/multind.h"
#include "num/ode.h"
#include "simu/bloch.h"

#include "rfcontrol.h"

/// @brief Solves Bloch equation using Crank-Nicolson scheme. Computes the magnetization vector M 
///	   starting from initial conditions M0 with RF pulse u,v and gradient w.
/// @param Nx Total number of spatial points
/// @param Nt Total number of temporal points
/// @param M Computed magnetization vector
/// @param u Control parameters
/// @param p Problem parameters
void cn_bloch(int Nx, int Nt, float M[Nx][Nt][3], const float u[Nt - 1], const struct puls_opt_pars p)
{
	float Mz[4] = { 0. };
	float Bz;

	// Loop over each voxel in the slice profile
	for (int z = 0; z < Nx; z++) {

		Bz = p.gamma * p.Gz * p.xdis[z];
		Mz[0] = p.M0[z];
		Mz[1] = p.M0[Nx + z];
		Mz[2] = p.M0[2 * Nx + z];

		vecf_copy(3, M[z][0], Mz);
 
		for (int k = 1; k < Nt; k++) {

			const float* gb = (const float[]){ u[k - 1] * p.B1, p.v[k - 1] * p.B1, p.w[k - 1] * Bz };

			NESTED(void, bloch_matrix_fun, (int N, float (*A)[N][N], float t))
			{
				(void)N;
				(void)t;

				bloch_matrix_ode((*A), p.r1, p.r2, gb);
				(*A)[2][3] = p.M0c * (*A)[2][3];
			};
			
			crank_nicolson(p.dt, 4, Mz, (k - 1) * p.dt, k * p.dt, bloch_matrix_fun);
			vecf_copy(3, M[z][k], Mz);
		}
	}
}

/// @brief Solves adjoint Bloch equation using adjoint Crank-Nicolson scheme. Computes the adjoint
///	   vector P starting from terminal conditions PT with RF pulse u,v and gradient w.
/// @param Nx Total number of spatial points
/// @param Nt Total number of temporal points
/// @param P Solution to the adjoint Bloch equation
/// @param u Control parameter
/// @param PT Terminal conditions for adjoint
/// @param p Problem parameters
void cn_adjoint(int Nx, int Nt, float P[Nx][Nt - 1][3], const float u[Nt - 1], 
	const float PT[Nx][3], const struct puls_opt_pars p)
{
	float Pz[4] = { 0. };
	float Bz;
	int end = Nt - 2;

	// Loop over each voxel in the slice profile
	for (int z = 0; z < Nx; z++) {

		Bz = p.gamma * p.Gz * p.xdis[z];

		// Initialize adjoint state
		vecf_copy(3, Pz, PT[z]);

		for (int k = end; k >= 0; k--) {

			int idx_kp1 = (k < end) ? k + 1 : k;
			const float* gb_kp1 = (const float[]){ -u[idx_kp1] * p.B1, -p.v[idx_kp1] * p.B1, -p.w[idx_kp1] * Bz };
			const float* gb_k = (const float[]){ -u[k] * p.B1, -p.v[k] * p.B1, -p.w[k] * Bz };

			NESTED(void, adjoint_matrix_fun, (int N, float (*Ak)[N][N], float (*Akp1)[N][N], float t))
			{
				(void)N;
				(void)t;

				bloch_matrix_ode((*Akp1), p.r1, p.r2, gb_kp1);
				bloch_matrix_ode((*Ak), p.r1, p.r2, gb_k);
			};

			crank_nicolson_adjoint(p.dt, 4, Pz, k * p.dt, (k + 1) * p.dt, adjoint_matrix_fun);
			vecf_copy(3, P[z][k], Pz);
		}
	}
}

/// @brief Computes the value J of the functional to be minimized together with the gradient G in the point u
///
/// @param Nu Number of temporal control points
/// @param G Gradient of the objective function (can be NULL if not needed)
/// @param Xk Contains information to evaluate the Hessian in u
/// @param p Problem parameters
/// @param iu Input control
/// @return functional J
float objfun(int Nu, float G[Nu], struct Xk_struct* Xk, const struct puls_opt_pars p, const float iu[Nu])
{
	// Zero padding of control to readout time
	float* u = xmalloc((size_t)(p.Nt - 1) * sizeof(float));
	memset(u, 0, (size_t)(p.Nt - 1) * sizeof(float));
	vecf_copy(p.Nu, u, iu);

	// Solve state equation
	float (*M)[p.Nt][3] = xmalloc((size_t)p.Nx * sizeof * M);
	cn_bloch(p.Nx, p.Nt, M, u, p);

	// Calculate residual at final time
	float (*res)[3] = xmalloc((size_t)p.Nx * sizeof * res);
	float res_norm_sqr = 0.;

	for (int z = 0; z < p.Nx; z++)
		for (int c = 0; c < 3; c++) {

			res[z][c] = M[z][p.Nt - 1][c] - p.Md[c * p.Nx + z];
			res_norm_sqr += res[z][c] * res[z][c];
		}

	// Objective function:
	//  1. term: penalizes deviation from the desired magnetization
	//  2. term: penalizes control energy (keeps RF pulse small)
	float u_norm_sqr = vecf_sdot((p.Nt - 1), u, u);
	float J = 0.5 * p.dx * res_norm_sqr + p.alpha / 2. * p.dt * u_norm_sqr;

	// Gradient
	if (G != NULL) {

		float (*P)[p.Nt - 1][3] = xmalloc((size_t)p.Nx * sizeof * P);
		float (*N)[p.Nt - 1][3] = xmalloc((size_t)p.Nx * sizeof * N);

		cn_adjoint(p.Nx, p.Nt, P, u, res, p);
		
		for (int z = 0; z < p.Nx; z++)
			for (int k = 0; k < p.Nt - 1; k++)
				for (int c = 0; c < 3; c++)
					N[z][k][c] = 0.5 * (M[z][k][c] + M[z][k + 1][c]);
		float sum;
		for (int i = 0; i < p.Nu; i++) {

			sum = 0.;
			for (int z = 0; z < p.Nx; z++)
				sum += N[z][i][2] * P[z][i][1] - N[z][i][1] * P[z][i][2];
			G[i] = p.alpha * u[i] + p.gamma * p.B1c * p.dx * sum;
		}

		// Hessian information
		if (Xk != NULL) {

			vecf_copy(p.Nt - 1, Xk->u, u);
			md_copy(3, (long[3]) { p.Nx, p.Nt - 1, 3 }, Xk->N, N, sizeof(float));
			md_copy(3, (long[3]) { p.Nx, p.Nt - 1, 3 }, Xk->P, P, sizeof(float));
		}

		xfree(P);
		xfree(N);
	}

	xfree(res);
	xfree(u);
	xfree(M);

	return J;
}

/// @brief Computes the action Hdu of the Hessian in direction du.
///
/// @param N Number of control points
/// @param Hdu Action of the Hessian in direction du
/// @param p Problem parameters
/// @param Xk The point in which the the Hessian is evaluated
/// @param idu Direction of the Hessian action
void apply_Hess(int N, float Hdu[N], const struct puls_opt_pars p, const struct Xk_struct* Xk, const float idu[N])
{
	float (*dP)[3] = xmalloc((size_t)(p.Nt - 1) * sizeof * dP);
	float (*dMz)[3] = xmalloc((size_t)p.Nt * sizeof * dMz);
	float (*dNz)[p.Nu] = xmalloc((size_t)3 * sizeof * dNz);

	float dPz[4];
	int end = p.Nt - 2;

	// Zero padding of control to readout time
	float* du = xmalloc((size_t)(p.Nt - 1) * sizeof(float));
	memset(du, 0, (size_t)(p.Nt - 1) * sizeof(float));
	vecf_copy(p.Nu, du, idu);

	memset(Hdu, 0, (size_t)p.Nu * sizeof(float));
	vecf_saxpy(p.Nu, Hdu, p.alpha, du); // Hdu = alpha * du

	// Loop over each voxel in the slice profile
	for (int z = 0; z < p.Nx; z++) {

		float Bz = p.gamma * p.Gz * p.xdis[z];
		float Mz[4] = { 0., 0., 0., 1. };

		// Initialize first time step
		vecf_zero(3, dMz[0]);

		// Step 1: Solve linearized state equation (forward)
		for (int k = 1; k < p.Nt; k++) {

			const float* gb = (const float[]){ Xk->u[k - 1] * p.B1, -p.v[k - 1] * p.B1, p.w[k - 1] * Bz };

			NESTED(void, lin_state_matrix_fun, (int N, float (*A)[N][N], float t))
			{
				(void)N;
				(void)t;

				bloch_matrix_ode((*A), p.r1, p.r2, gb);

				(*A)[1][3] = p.B1 * Xk->N[z * (p.Nt - 1) * 3 + (k - 1) * 3 + 2] * du[k - 1];
				(*A)[2][3] = -p.B1 * Xk->N[z * (p.Nt - 1) * 3 + (k - 1) * 3 + 1] * du[k - 1];
			};

			crank_nicolson(p.dt, 4, Mz, (k - 1) * p.dt, k * p.dt, lin_state_matrix_fun);
			vecf_copy(3, dMz[k], Mz);
		}

		for (int k = 0; k < p.Nu; k++)
			for (int i = 0; i < 3; i++)
				dNz[i][k] = 0.5 * (dMz[k][i] + dMz[k + 1][i]);

		// Terminal condition for adjoint
		vecf_copy(3, dPz, dMz[p.Nt - 1]);
		dPz[3] = 1.;

		// Step 2: Solve linearized adjoint equation (backward)
		for (int k = end; k >= 0; k--) {

			NESTED(void, lin_adjoint_matrix_fun, (int N, float (*Ak)[N][N], float (*Akp1)[N][N], float t))
			{
				(void)N;
				(void)t;

				int idx_kp1 = (k < end) ? k + 1 : k;
				float gb_kp1[3] = { -Xk->u[idx_kp1] * p.B1, -p.v[idx_kp1] * p.B1, -p.w[idx_kp1] * Bz };
				bloch_matrix_ode(*Akp1, p.r1, p.r2, gb_kp1);

				(*Akp1)[1][3] = -p.B1 * Xk->P[z * (p.Nt - 1) * 3 + idx_kp1 * 3 + 2] * du[idx_kp1];
				(*Akp1)[2][3] = p.B1 * Xk->P[z * (p.Nt - 1) * 3 + idx_kp1 * 3 + 1] * du[idx_kp1];

				float gb_k[3] = { -Xk->u[k] * p.B1, -p.v[k] * p.B1, -p.w[k] * Bz };
				bloch_matrix_ode(*Ak, p.r1, p.r2, gb_k);

				(*Ak)[1][3] = -p.B1 * Xk->P[z * (p.Nt - 1) * 3 + k * 3 + 2] * du[k];
				(*Ak)[2][3] = p.B1 * Xk->P[z * (p.Nt - 1) * 3 + k * 3 + 1] * du[k];
			};

			crank_nicolson_adjoint(p.dt, 4, dPz, k * p.dt, (k + 1) * p.dt, lin_adjoint_matrix_fun);
			vecf_copy(3, dP[k], dPz);
		}

		// Action of Hessian
		for (int i = 0; i < p.Nu; i++) {

			float sum = dNz[2][i] * Xk->P[z * (p.Nt - 1) * 3 + i * 3 + 1]
				- dNz[1][i] * Xk->P[z * (p.Nt - 1) * 3 + i * 3 + 2]
				+ dP[i][1] * Xk->N[z * (p.Nt - 1) * 3 + i * 3 + 2]
				- dP[i][2] * Xk->N[z * (p.Nt - 1) * 3 + i * 3 + 1];
			Hdu[i] += p.B1 * p.dx * sum;
		}
	}

	xfree(du);
	xfree(dP);
	xfree(dMz);
	xfree(dNz);
}

static float dist2bdy(int N, const float du[N], const float p[N], float trad, float CLOSURE_TYPE(ip)(int N, const float x[N], const float y[N]))
{
	// find distance to trust-region boundary from du in direction p
	float dd = 0., xd = 0., xx = 0.;

	dd = NESTED_CALL(ip, (N, p, p));
	xd = NESTED_CALL(ip, (N, du, p));
	xx = NESTED_CALL(ip, (N, du, du));

	float ss = trad * trad;
	float det = xd * xd + dd * (ss - xx);
	float tau = (ss - xx) / (xd + sqrtf(det));

	return tau;
}

/// @brief TR_CG iteration solves the Newton step HDU = -G  using Steihaug's trust-region 
///	   conjugate gradient method.
/// @param Nu Number of temporal control points
/// @param du Candidate step u computed by TR-CG
/// @param it Number of TR-CG iterations performed
/// @param g Gradient of the objective function
/// @param trad Radius of trust region
/// @param np Trust-region Newton method parameters
/// @param H_func Function that computes the action of the Hessian H on a given vector p s.t. Hp = H(p)
/// @param ip Inner product function: ip(x,y)
/// @return Convergence FLAG:
///		0: TRCG converged to the desired tolerance TOL within IT iterations
///		1: TRCG iterated MAXIT times but did not converge
///		2: TRCG terminated because the iterate left the trust region
///		3: TRCG terminated because negative curvature was encountered
int tr_cg(int Nu, float du[Nu], int* it, const float g[Nu], float trad, const struct tr_pars np,
	  void CLOSURE_TYPE(H_func)(int N, float Hp[N], const float p[N]),
	  float CLOSURE_TYPE(ip)(int N, const float x[N], const float y[N]))
{
	int flag;
	float pHp, tau, al, step_norm, nrk, beta;
	float* Hp = xmalloc((size_t)Nu * sizeof(float));
	float* temp = xmalloc((size_t)Nu * sizeof(float));
	float* r = xmalloc((size_t)Nu * sizeof(float));
	float* p = xmalloc((size_t)Nu * sizeof(float));

	vecf_zero(Nu, du);
	vecf_zero(Nu, r);
	vecf_saxpy(Nu, r, -1, g); // r = -g
	vecf_copy(Nu, p, r); 	  // p = -g

	// Compute initial residual norm
	float nr = NESTED_CALL(ip, (Nu, r, r));
	float nr0 = sqrtf(nr);

	*it = 1;

	while (1) {

		NESTED_CALL(H_func, (Nu, Hp, p));
		pHp = NESTED_CALL(ip, (Nu, p, Hp));

		// Check for negative curvature
		if (pHp < __FLT_MIN__) {

			tau = dist2bdy(Nu, du, p, trad, ip); 	// Go to boundary
			vecf_saxpy(Nu, du, tau, p); 		// du = du + tau * p
			flag = 3; 				// Negative curvature
			break;
		}

		al = nr / pHp;

		// Check if step is too large
		vecf_copy(Nu, temp, du);
		vecf_saxpy(Nu, temp, al, p);
		step_norm = NESTED_CALL(ip, (Nu, temp, temp));

		if (step_norm >= trad * trad) {

			tau = dist2bdy(Nu, du, p, trad, ip); 	// Go to boundary
			vecf_saxpy(Nu, du, tau, p); 		// du = du + tau * p
			flag = 2; 				// Step too large
			break;
		}

		vecf_saxpy(Nu, du, al, p);  // du = du + al * p
		vecf_saxpy(Nu, r, -al, Hp); // r = r - al * Hp

		nrk = NESTED_CALL(ip, (Nu, r, r));

		// Check convergence
		if (nrk < np.cgtol * powf(nr0, 1.3f)) { // Norm of residual small enough

			flag = 0; // Converged
			break;
		}
		else if (*it == np.cgits) { // Too many iterations, but not converged

			flag = 1; // Max iterations
			break;
		}

		beta = nrk / nr;
		vecf_sxpay(Nu, beta, p, r); // p = r + beta * p

		nr = nrk;
		(*it)++;
	}

	xfree(r);
	xfree(p);
	xfree(Hp);
	xfree(temp);

	return flag;
}

/// @brief The trust-region CG-Newton method computes the optimal control u. 
///	   The functional to be minimized is specified using the function handles OBJFUN, which 
///	   evaluates functional and gradient, and APPLY_HESS, which computes the action of the Hessian on a given direction. 
/// 
///        C. S. Aigner, C. Clason, A. Rund, and R. Stollberger, ‘Efficient high-resolution RF pulse 
///        design applied to simultaneous multi-slice excitation’, J. Magn. Reson., vol. 263, pp. 33–44, Feb. 2016.
/// 
/// @param Nu Number of temporal control points
/// @param u Optimal control u
/// @param p Problem parameters
/// @param np Trust-region Newton method parameters
/// @param u0 Initial guess for the control u
void tr_newton(int Nu, float u[Nu], const struct puls_opt_pars p, const struct tr_pars np, float* u0)
{
	NESTED(float, ip, (int N, const float x[N], const float y[N]))
	{
		float sum = 0.;
		for (int i = 0; i < N; i++)
			sum += p.dt * x[i] * y[i];
		return sum;
	};

	float* G = xmalloc((size_t)Nu * sizeof(float));
	struct Xk_struct Xk;
	Xk.N = md_alloc(3, (long[3]) { 3, p.Nx, p.Nt - 1 }, sizeof(float));
	Xk.P = md_alloc(3, (long[3]) { p.Nx, p.Nt - 1, 3 }, sizeof(float));
	Xk.u = md_alloc(1, (long[1]) { p.Nt - 1 }, sizeof(float));

	int it = 0;

	float J = objfun(p.Nu, G, &Xk, p, u0);
	float nrG0 = sqrtf(NESTED_CALL(ip, (p.Nu, G, G)));

	vecf_copy(p.Nu, u, u0);

	debug_printf(DP_DEBUG1, "it \tJ \t\t|g| \t\tflag \trho \t\tdJa/dJm \tcgits\n");
	debug_printf(DP_DEBUG1, "%d\t%1.3e\t%1.3e\n", it, J, nrG0);

	int flag, cgit;
	float dJa, dJm, Jratio, nrG;
	float* du = xmalloc((size_t)p.Nu * sizeof(float));
	float* udu = xmalloc((size_t)p.Nu * sizeof(float));
	float* Hmult_res = xmalloc((size_t)p.Nu * sizeof(float));
	float rho = np.rho;

	NESTED(void, Hmult, (int N, float Hdu[N], const float du[N]))
	{
		apply_Hess(N, Hdu, p, &Xk, du);
	};

	for (; it < np.maxit; it++) {

		// Minimize quadratic model
		flag = tr_cg(p.Nu, du, &cgit, G, rho, np, Hmult, ip);

		vecf_axpbz(p.Nu, udu, 1., u, 1., du); // udu = u + du

		// Test if control is updated
		dJa = J - objfun(p.Nu, NULL, NULL, p, udu);			// Actual reduction in J

		NESTED_CALL(Hmult, (p.Nu, Hmult_res, du));
		dJm = -(0.5 * NESTED_CALL(ip, (p.Nu, du, Hmult_res)) 		// Predicted reduction in J
			+ NESTED_CALL(ip, (p.Nu, G, du)));

		Jratio = dJa / dJm; 						// Ratio of real and predicted decrease

		if ((dJa > __FLT_MIN__) && (Jratio > np.sig1)) { 		// Actual reduction and mode good -> accept step

			vecf_copy(p.Nu, u, udu);
			J = objfun(p.Nu, G, &Xk, p, u);
		}
		else
			debug_printf(DP_DEBUG1, "R\n");

		// Test if radius is updated
		if ((dJa > __FLT_MIN__) && (fabsf(Jratio - 1) <= 1 - np.sig3))	// Step accepted, model good
			rho = fminf(rho * np.q, np.maxrad);		 	// Increase radius
		else if (dJa <= __FLT_MIN__)					// Step rejected, no decrease
			rho = 1. / np.q * rho;				 	// Decrease radius
		else if (Jratio < np.sig2)				 	// Model bad
			rho = 1. / np.q * rho;				 	// Decrease radius

		nrG = sqrtf(NESTED_CALL(ip, (p.Nu, G, G)));

		debug_printf(DP_DEBUG1, "%d\t%1.3e\t%1.3e\t%d\t%1.3e\t%1.3e\t%d\n", it + 1, J, nrG, flag, rho, Jratio, cgit);

		if ((nrG < np.reltol * nrG0) || (nrG < np.abstol))		// Tolerance reached
			break;
	}

	xfree(G);
	xfree(du);
	xfree(udu);
	xfree(Hmult_res);

	md_free(Xk.N);
	md_free(Xk.P);
	md_free(Xk.u);
}
