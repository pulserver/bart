#define tridiag(s) (s * (s + 1) / 2)

static void ODE_SUFFIX(runge_kutta_step)(ode_type h, int s, const ode_type a[tridiag(s)], 
	const ode_type b[s], const ode_type c[s - 1], int N, int K, ode_type k[K][N], 
	ode_type ynp[N], ode_type tmp[N], ode_type tn, const ode_type yn[N], 
	ODE_FUN f)
{
	ODE_VEC_COPY(N, ynp, yn);
	ODE_VEC_SAXPY(N, ynp, h * b[0], k[0]);

	for (int l = 0, t = 1; t < s; t++) {

		ODE_VEC_COPY(N, tmp, yn);

		for (int r = 0; r < t; r++, l++)
			ODE_VEC_SAXPY(N, tmp, h * a[l], k[r % K]);

		NESTED_CALL(f, (k[t % K], tn + h * c[t - 1], tmp));

		ODE_VEC_SAXPY(N, ynp, h * b[t], k[t % K]);
	}
}


ode_type ODE_SUFFIX(dormand_prince_step2)(ode_type h, int N, ode_type ynp[N], ode_type tn, 
	const ode_type yn[N], ode_type k[6][N], ODE_FUN f)
{
	const ode_type c[6] = { 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1., 1. };

	const ode_type a[tridiag(7)] = {
		1. / 5.,
		3. / 40.,	9. / 40.,
		44. / 45.,	-56. / 15.,	32. / 9.,
		19372. / 6561.,	-25360. / 2187., 64448. / 6561., -212. / 729.,
		9017. / 3168.,  -355. / 33.,	46732. / 5247.,	49. / 176.,	-5103. / 18656.,
		35. / 384.,	0.,		500. / 1113.,	125. / 192.,	-2187. / 6784.,	11. / 84.,
	};

	const ode_type b[7] = { 5179. / 57600., 0.,  7571. / 16695., 393. / 640., -92097. / 339200., 187. / 2100., 1. / 40. };

	ode_type tmp[N];
	ODE_SUFFIX(runge_kutta_step)(h, 7, a, b, c, N, 6, k, ynp, tmp, tn, yn, f);

	ODE_VEC_SAXPY(N, tmp, -1., ynp);
	return ODE_VEC_NORM(N, tmp);
}


ode_type ODE_SUFFIX(dormand_prince_scale)(ode_type tol, ode_type err)
{
#if 0
	ode_type sc = 0.75 * ODE_POW(tol / err, 1. / 5.);

	return (sc < 2.) ? sc : 2.;
#else
	ode_type sc = 1.25 * ODE_POW(err / tol, 1. / 5.);

	return 1. / ((sc > 1. / 2.) ? sc : (1. / 2.));
#endif
}


void (ODE_SUFFIX(ode_interval))(ode_type h, ode_type h_min, ode_type tol, int N, ode_type x[N], 
	ode_type st, ode_type end, ODE_FUN f)
{
	ode_type k[6][N];
	NESTED_CALL(f, (k[0], st, x));

	if (h > end - st)
		h = end - st;

	for (ode_type t = st; t < end; ) {

		ode_type ynp[N];
	repeat:
		;
		ode_type err = ODE_SUFFIX(dormand_prince_step2)(h, N, ynp, t, x, k, f);

		ode_type h_new = h * ODE_SUFFIX(dormand_prince_scale)(tol, err);

		if (0. < h_min)
			h_new = fmax(h_new, h_min);

		if (err > tol && (h_min <= 0. || h > h_min)) {

			h = h_new;
			NESTED_CALL(f, (k[0], t, x));	// recreate correct k[0] which has been overwritten
			goto repeat;
		}

		t += h;
		h = h_new;

		if (t + h > end)
			h = end - t;

		for (int i = 0; i < N; i++)
			x[i] = ynp[i];
	}
}


// the direct method for sensitivity analysis
// ode: d/dt y_i = f_i(y, t, p_j), y_i(0) = a_i
// d/dp_j y_i(0) = ...
// d/dt d/dp_j y_i(t) = d/dp_j f_i(y, t, p) = \sum_k d/dy_k f_i(y, t, p) * dy_k/dp_j + df_i / dp_j

struct ODE_SUFFIX(seq_data) {

	int N;
	int P;

	ODE_FUN f;
	ODE_FUN pdy;
	ODE_FUN pdp;
};

static void ODE_SUFFIX(seq)(const struct ODE_SUFFIX(seq_data)* data, ode_type* out, ode_type t, const ode_type* yn)
{
	int N = data->N;
	int P = data->P;

	NESTED_CALL(data->f, (out, t, yn));

	ode_type dy[N][N];
	NESTED_CALL(data->pdy, (&dy[0][0], t, yn));

	ode_type dp[P][N];
	NESTED_CALL(data->pdp, (&dp[0][0], t, yn));

	for (int i = 0; i < P; i++) {
		for (int j = 0; j < N; j++) {

			out[(1 + i) * N + j] = 0.;

			for (int k = 0; k < N; k++)
				out[(1 + i) * N + j] += dy[k][j] * yn[(1 + i) * N + k];

			out[(1 + i) * N + j] += dp[i][j];
		}
	}
}

void (ODE_SUFFIX(ode_direct_sa))(ode_type h, ode_type h_min, ode_type tol, int N, int P,
	ode_type x[P + 1][N],
	ode_type st, ode_type end,
	ODE_FUN f,
	ODE_FUN pdy,
	ODE_FUN pdp)
{
	struct ODE_SUFFIX(seq_data) data2 = { N, P, f, pdy, pdp };

	NESTED(void, seq2, (ode_type* out, ode_type t, const ode_type* yn))
	{
		ODE_SUFFIX(seq)(&data2, out, t, yn);
	};

	ODE_SUFFIX(ode_interval)(h, h_min, tol, N * (1 + P), &x[0][0], st, end, seq2);
}
