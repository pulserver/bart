void BLOCH_SUFFIX(bloch_matrix_ode)(bloch_type matrix[4][4], bloch_type r1, bloch_type r2, const bloch_type gb[3])
{
	bloch_type m0 = 1.;
	bloch_type m[4][4] = {
		{	-r2,		gb[2],		-gb[1],		0.	},
		{	-gb[2],		-r2,		gb[0],		0.	},
		{	gb[1],		-gb[0],		-r1,		m0 * r1 },
		{	0.,		0.,		0.,		0.	},
	};

	BLOCH_MAT_COPY(4, 4, matrix, m);
}

void BLOCH_SUFFIX(bloch_mcconnel_matrix_ode)(int P, bloch_type matrix[1 + P * 3][1 + P * 3], 
	const bloch_type r1[P], const bloch_type r2[P], const bloch_type k[P - 1],
	const bloch_type m0[P], const bloch_type Om[P], const bloch_type gb[3])
{
	// +1 for T1 relaxation term in homogeneous ODE representation
	int N = 1 + P * 3;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			matrix[i][j] = 0.;

	// copy 3 x 3 Bloch matrix

	for (int p = 0; p < P; p++) {

		bloch_type g[3];
		for (int i = 0; i < 3; i++)
			g[i] = gb[i];

		g[2] += Om[p];

		bloch_type m[4][4];
		BLOCH_SUFFIX(bloch_matrix_ode)(m, r1[p], r2[p], g);

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				matrix[3 * p + i][3 * p + j] = m[i][j];
	}

	// equilibrium

	for (int p = 0; p < P; p++)
		matrix[3 * p + 2][N - 1] = m0[p] * r1[p];

	// exchange  FIXME?: Additional pools only exchange with water pool
	for (int p = 0; p < P - 1; p++)
		for (int i = 0; i < 3; i++)
			matrix[i][i] -= k[p] * m0[p + 1];

	for (int p = 1; p < P; p++) {
		for (int i = 0; i < 3; i++) {

			matrix[p * 3 + i][p * 3 + i] -= k[p - 1] * m0[0];
			matrix[i][p * 3 + i] += k[p - 1] * m0[0];
			matrix[p * 3 + i][i] += k[p - 1] * m0[p];
		}
	}

}

// RIGHT-handed coordinate system with CLOCKWISE rotation
// dM/dt = MxB - ...
void BLOCH_SUFFIX(bloch_ode)(bloch_type out[3], const bloch_type in[3], bloch_type r1, bloch_type r2, const bloch_type gb[3])
{
	bloch_type m0 = 1.;
	BLOCH_VEC3_ROT(out, in, gb);
	out[0] -= in[0] * r2;
	out[1] -= in[1] * r2;
	out[2] -= (in[2] - m0) * r1;
}

void BLOCH_SUFFIX(bloch_pdy)(bloch_type out[3][3], const bloch_type in[3], bloch_type r1, bloch_type r2, const bloch_type gb[3])
{
	(void)in;

	BLOCH_VEC3_ROT(out[0], (bloch_type[3]){ 1., 0., 0. }, gb);
	out[0][0] -= r2;

	BLOCH_VEC3_ROT(out[1], (bloch_type[3]){ 0., 1., 0. }, gb);
	out[1][1] -= r2;

	BLOCH_VEC3_ROT(out[2], (bloch_type[3]){ 0., 0., 1. }, gb);
	out[2][2] -= r1;
}

void BLOCH_SUFFIX(bloch_b1_pdp)(bloch_type out[3][3], const bloch_type in[3], bloch_type r1, bloch_type r2, const bloch_type gb[3], complex bloch_type b1)
{
	(void)r1; (void)r2; (void)gb;

	bloch_type m0 = 1.;
	out[0][0] = 0.;
	out[0][1] = 0.;
	out[0][2] = -(in[2] - m0);
	out[1][0] = -in[0];
	out[1][1] = -in[1];
	out[1][2] = 0.;
	out[2][0] = in[2] * BLOCH_CIMAG(b1);
	out[2][1] = in[2] * BLOCH_CREAL(b1);
	out[2][2] = -BLOCH_CIMAG(b1) * in[0] - BLOCH_CREAL(b1) * in[1];
}


void BLOCH_SUFFIX(bloch_mcconnell_ode)(int P, bloch_type out[P * 3], const bloch_type in[P * 3], 
	bloch_type r1[P], bloch_type r2[P], bloch_type k[P - 1], bloch_type m0[P], 
	bloch_type Om[P], bloch_type gb[3])
{
	int N = 1 + P * 3;
	bloch_type m[N][N];

	// create Bloch-McConnell matrix
	BLOCH_SUFFIX(bloch_mcconnel_matrix_ode)(P, m, r1, r2, k, m0, Om, gb);

	bloch_type out_tmp[N];
	bloch_type in_tmp[N];

	for (int i = 0; i < N - 1; i++)
		in_tmp[i] = in[i];

	in_tmp[N - 1] = 1.;

	BLOCH_MAT_VECMUL(N, N, out_tmp, m, in_tmp);

	for (int i = 0; i < N - 1; i++)
		out[i] = out_tmp[i];
}

void BLOCH_SUFFIX(bloch_mcc_pdy)(int P, bloch_type out[P * 3][P * 3], const bloch_type /*in*/[P * 3],
	bloch_type r1[P], bloch_type r2[P], const bloch_type k[P - 1], const bloch_type m0[P], 
	const bloch_type Om[P], const bloch_type gb[3])
{
	bloch_type m[P * 3 + 1][P * 3 + 1];

	BLOCH_SUFFIX(bloch_mcconnel_matrix_ode)(P, m, r1, r2, k, m0, Om, gb);

	for (int i = 0; i < P * 3; i++)
		for (int j = 0; j < P * 3; j++)
			out[j][i] = m[i][j]; //transposition
}

void BLOCH_SUFFIX(bloch_mcc_b1_pdp)(int P, bloch_type out[P * 5 - 1][P * 3], 
	const bloch_type in[P * 3], bloch_type r1[P], bloch_type /*r2*/[P], 
	const bloch_type k[P - 1], const bloch_type m0[P], const bloch_type /*gb*/[3], 
	complex bloch_type b1)
{
	for (int i = 0; i < P * 5 - 1; i++)
		for (int j = 0; j < P * 3; j++)
			out[i][j] = 0.;

	assert(P > 1);

	for (int p = 0; p < P; p++) {

		// R1
		out[0 + p][2 + p * 3] = - (in[2 + p * 3] - m0[p]);
		// R2
		out[P + p][0 + p * 3] = - in[p * 3];
		out[P + p][1 + p * 3] = - in[1 + p * 3];
		// B1
		out[2 * P][0 + p * 3] = in[2 + p * 3] * cimag(b1);
		out[2 * P][1 + p * 3] = in[2 + p * 3] * crealf(b1);
		out[2 * P][2 + p * 3] = -cimagf(b1) * in[0 + p * 3] - crealf(b1) * in[1 + p * 3];
	}

	for (int p = 0; p < P - 1; p++) {

		// M0 water; special treatment because of exchange with all other pools
		out[2 * P + 1][0] += in[3 + p * 3] * k[p];
		out[2 * P + 1][1] += in[4 + p * 3] * k[p];
		out[2 * P + 1][2] += in[5 + p * 3] * k[p];
		out[2 * P + 1][3 + p * 3] = -in[3 + p * 3] * k[p];
		out[2 * P + 1][4 + p * 3] = -in[4 + p * 3] * k[p];
		out[2 * P + 1][5 + p * 3] = -in[5 + p * 3] * k[p];

		// M0 remaining
		out[2 * P + 2 + p][0] = -in[0] * k[p];
		out[2 * P + 2 + p][1] = -in[1] * k[p];
		out[2 * P + 2 + p][2] = -in[2] * k[p];
		out[2 * P + 2 + p][3 + p * 3] = in[0] * k[p];
		out[2 * P + 2 + p][4 + p * 3] = in[1] * k[p];
		out[2 * P + 2 + p][5 + p * 3] = r1[p + 1] + in[2] * k[p];

		// k; Pools only exchange with water pool
		out[3 * P + 1 + p][0] = -in[0] * m0[p + 1] + in[3 + p * 3] * m0[0];
		out[3 * P + 1 + p][1] = -in[1] * m0[p + 1] + in[4 + p * 3] * m0[0];
		out[3 * P + 1 + p][2] = -in[2] * m0[p + 1] + in[5 + p * 3] * m0[0];
		out[3 * P + 1 + p][3 + p * 3] = in[0] * m0[p + 1] - in[3 + p * 3] * m0[0];
		out[3 * P + 1 + p][4 + p * 3] = in[1] * m0[p + 1] - in[4 + p * 3] * m0[0];
		out[3 * P + 1 + p][5 + p * 3] = in[2] * m0[p + 1] - in[5 + p * 3] * m0[0];

		// Om
		out[4 * P + p][3 + p * 3] = in[4 + p * 3];
		out[4 * P + p][4 + p * 3] = -in[3 + p * 3];
	}

	// Missing term for M0 water
	out[2 * P + 1][2] += r1[0];
}
