static void SIM_SUFFIX(compute_fields)(struct sim_data* data, sim_type gb_eff[3], sim_type t)
{
	// Units: [gb] = rad/s
	gb_eff[0] = data->grad.gb[0];
	gb_eff[1] = data->grad.gb[1];
	gb_eff[2] = data->grad.gb[2];

	if (!data->seq.pulse_applied)
		return;

	struct pulse* ps = get_pulse(data);

	if (NULL == ps)
		return;

	complex sim_type w1 = SIM_CEXP(1.i * data->pulse.phase) * pulse_eval(ps, t);

	// Definition from Bernstein et al., Handbook of MRI Pulse Sequences, p. 26f
	// dM/dt = M x (e_x*B1*cos(phi) - e_y*B1*sin(phi) + e_z*B0) - ...
	gb_eff[0] = SIM_CREAL(w1);
	gb_eff[1] = -SIM_CIMAG(w1);
}


void SIM_SUFFIX(rf_pulse)(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float stm_matrix[P * N][P * N])
{
	data->seq.pulse_applied = true;

        // Single hard pulse is special case of homogeneously sampled sinc pulse
        if (0. == data->pulse.rf_end)
                data->seq.type = SIM_ROT;

        // Define effective z Gradient = Slice-selection gradient + off-resonance [rad/s]
	data->grad.gb[2] = data->grad.mom_sl + data->voxel.w;

	__block complex sim_type w1; // clang workaround (needs to be outside switch)
	
        switch (data->seq.type) {

		case SIM_ROT: {

			rot_pulse(data, N, P, xp); // Not supported for double-precision
			break;
		}
		case SIM_ODE: {

			sim_type xp_d[P][N];
			for (int i = 0; i < P; i++)
				for (int j = 0; j < N; j++)
					xp_d[i][j] = xp[i][j];

			sim_type gb_eff[3];
			void *gb_eff_p = gb_eff;	 // clang workaround

			sim_type r1_d_eff[5];
			sim_type *r1_d_eff_p = r1_d_eff; // clang workaround

			sim_type r2_d_eff[5];
			sim_type *r2_d_eff_p = r2_d_eff; // clang workaround

			sim_type m0_d_eff[5];
			sim_type *m0_d_eff_p = m0_d_eff; // clang workaround

			sim_type Om_d_eff[5];
			sim_type *Om_d_eff_p = Om_d_eff; // clang workaround

			sim_type k_d_eff[4];
			sim_type *k_d_eff_p = k_d_eff;	 // clang workaround

			// Copy voxel parameters to single/double precision arrays
			for (int i = 0; i < data->voxel.P; i++) {

				r1_d_eff_p[i] = data->voxel.r1[i];
				r2_d_eff_p[i] = data->voxel.r2[i];
				m0_d_eff_p[i] = data->voxel.m0[i];
				Om_d_eff_p[i] = data->voxel.Om[i];

				if (i < data->voxel.P - 1)
					k_d_eff_p[i] = data->voxel.k[i];
			}

			NESTED(void, call_fun, (sim_type* out, sim_type t, const sim_type* in))
			{
				sim_type *gb_eff = gb_eff_p;
				sim_type *r1_d_eff = r1_d_eff_p;
				sim_type *m0_d_eff = m0_d_eff_p;
				sim_type *Om_d_eff = Om_d_eff_p;
				sim_type *k_d_eff = k_d_eff_p;

				SIM_SUFFIX(compute_fields)(data, gb_eff, t);

				w1 = gb_eff[0] - 1.i * gb_eff[1];

				gb_eff[0] *= data->voxel.b1;
				gb_eff[1] *= data->voxel.b1;

				sim_type r2[data->voxel.P];

				if (MODEL_BMC == data->seq.model) {

					for (int i = 0; i < data->voxel.P; i++)
						r2[i] = data->voxel.r2[i];

					SIM_SUFFIX(bloch_mcconnell_ode)(data->voxel.P, out, in, 
						r1_d_eff, r2, k_d_eff, m0_d_eff, Om_d_eff, gb_eff);

				} else
					SIM_SUFFIX(bloch_ode)(out, in, data->voxel.r1[0], data->voxel.r2[0], gb_eff);
			};

			NESTED(void, call_pdy2, (sim_type* out, sim_type t, const sim_type* in))
			{
				sim_type *gb_eff = gb_eff_p;
				sim_type *r1_d_eff = r1_d_eff_p;
				sim_type *r2_d_eff = r2_d_eff_p;
				sim_type *m0_d_eff = m0_d_eff_p;
				sim_type *Om_d_eff = Om_d_eff_p;
				sim_type *k_d_eff = k_d_eff_p;
				(void)t;

				if (MODEL_BMC == data->seq.model)
					SIM_SUFFIX(bloch_mcc_pdy)(data->voxel.P, (sim_type (*)[N])out, in, 
						r1_d_eff, r2_d_eff, k_d_eff, m0_d_eff, Om_d_eff, gb_eff);
				else
					SIM_SUFFIX(bloch_pdy)((sim_type (*)[3])out, in, data->voxel.r1[0], data->voxel.r2[0], gb_eff);
			};

			NESTED(void, call_pdp2, (sim_type* out, sim_type t, const sim_type* in))
			{
				sim_type *gb_eff = gb_eff_p;
				sim_type *r1_d_eff = r1_d_eff_p;
				sim_type *r2_d_eff = r2_d_eff_p;
				sim_type *m0_d_eff = m0_d_eff_p;
				sim_type *k_d_eff = k_d_eff_p;
				(void)t;

				if (MODEL_BMC == data->seq.model)
					SIM_SUFFIX(bloch_mcc_b1_pdp)(data->voxel.P, (sim_type (*)[N])out, 
						in, r1_d_eff, r2_d_eff, k_d_eff, m0_d_eff, gb_eff, w1);
				else
					SIM_SUFFIX(bloch_b1_pdp)((sim_type(*)[3])out, in, data->voxel.r1[0], 
						data->voxel.r2[0], gb_eff, w1);
			};

			// Choose P-1 because ODE interface treats signal separate and 
			// P only describes the number of parameters
			SIM_SUFFIX(ode_direct_sa)(h, data->other.ode_h_min, tol, N, P - 1, xp_d, 
				data->pulse.rf_start, data->pulse.rf_end, call_fun, call_pdy2, call_pdp2);

			// Copy back to single precision
			for (int i = 0; i < P; i++)
				for (int j = 0; j < N; j++)
					xp[i][j] = xp_d[i][j];

			break;
		}
		case SIM_STM: {

			create_sim_matrix(data, P * N, stm_matrix, data->pulse.rf_start, data->pulse.rf_end, 0.); // Not supported for double-precision
			break;
		}
        }

        data->grad.gb[2] = 0.;
}

void SIM_SUFFIX(relaxation2)(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end, float stm_matrix[P * N][P * N], float r2spoil)
{
	data->seq.pulse_applied = false;

        // Single hard pulse is special case of homogeneously sampled sinc pulse
        if (0. == data->pulse.rf_end)
                data->seq.type = SIM_ROT;

        // Define effective z Gradient =Gradient Moments + off-resonance [rad/s]
        data->grad.gb[2] = data->grad.mom + data->voxel.w;
		
	__block complex sim_type w1; // clang workaround (needs to be outside switch)

        switch (data->seq.type) {

		case SIM_ROT: {

			hard_relaxation(data, N, P, xp, st, end, r2spoil);  // Not supported for double-precision
			break;
		}
		case SIM_ODE: {

			sim_type xp_d[P][N];
			for (int i = 0; i < P; i++)
				for (int j = 0; j < N; j++)
					xp_d[i][j] = xp[i][j];

			sim_type gb_eff[3];
			void *gb_eff_p = gb_eff;	 // clang workaround

			sim_type r1_d_eff[5];
			sim_type *r1_d_eff_p = r1_d_eff; // clang workaround

			sim_type m0_d_eff[5];
			sim_type *m0_d_eff_p = m0_d_eff; // clang workaround

			sim_type Om_d_eff[5];
			sim_type *Om_d_eff_p = Om_d_eff; // clang workaround

			sim_type k_d_eff[4];
			sim_type *k_d_eff_p = k_d_eff;	 // clang workaround

			// Copy voxel parameters to single/double precision arrays
			for (int i = 0; i < data->voxel.P; i++) {

				r1_d_eff_p[i] = data->voxel.r1[i];
				m0_d_eff_p[i] = data->voxel.m0[i];
				Om_d_eff_p[i] = data->voxel.Om[i];

				if (i < data->voxel.P - 1)
					k_d_eff_p[i] = data->voxel.k[i];
			}

			NESTED(void, call_fun, (sim_type* out, sim_type t, const sim_type* in))
			{
				sim_type *gb_eff = gb_eff_p;
				sim_type *r1_d_eff = r1_d_eff_p;
				sim_type *m0_d_eff = m0_d_eff_p;
				sim_type *Om_d_eff = Om_d_eff_p;
				sim_type *k_d_eff = k_d_eff_p;

				SIM_SUFFIX(compute_fields)(data, gb_eff, t);

				w1 = gb_eff[0] - 1.i * gb_eff[1];

				gb_eff[0] *= data->voxel.b1;
				gb_eff[1] *= data->voxel.b1;

				sim_type r2[data->voxel.P];

				if (MODEL_BMC == data->seq.model) {

					for (int i = 0; i < data->voxel.P; i++)
						r2[i] = data->voxel.r2[i] + r2spoil;

					SIM_SUFFIX(bloch_mcconnell_ode)(data->voxel.P, out, in, 
						r1_d_eff, r2, k_d_eff, m0_d_eff, Om_d_eff, gb_eff);

				} else
					SIM_SUFFIX(bloch_ode)(out, in, data->voxel.r1[0], 
						data->voxel.r2[0] + r2spoil, gb_eff);
			};

			NESTED(void, call_pdy2, (sim_type* out, sim_type t, const sim_type* in))
			{
				sim_type *gb_eff = gb_eff_p;
				sim_type *r1_d_eff = r1_d_eff_p;
				sim_type *m0_d_eff = m0_d_eff_p;
				sim_type *Om_d_eff = Om_d_eff_p;
				sim_type *k_d_eff = k_d_eff_p;
				(void)t;

				sim_type r2[data->voxel.P];

				if (MODEL_BMC == data->seq.model) {

					for (int i = 0; i < data->voxel.P; i++)
						r2[i] = data->voxel.r2[i] + r2spoil;

					SIM_SUFFIX(bloch_mcc_pdy)(data->voxel.P, (sim_type (*)[N])out, 
						in, r1_d_eff, r2, k_d_eff, m0_d_eff, Om_d_eff, gb_eff);
				} else
					SIM_SUFFIX(bloch_pdy)((sim_type(*)[3])out, in, data->voxel.r1[0], 
						data->voxel.r2[0] + r2spoil, gb_eff);
			};

			NESTED(void, call_pdp2, (sim_type* out, sim_type t, const sim_type* in))
			{
				sim_type *gb_eff = gb_eff_p;
				sim_type *r1_d_eff = r1_d_eff_p;
				sim_type *m0_d_eff = m0_d_eff_p;
				sim_type *k_d_eff = k_d_eff_p;
				(void)t;

				sim_type r2[data->voxel.P];

				if (MODEL_BMC == data->seq.model) {

					for (int i = 0; i < data->voxel.P; i++)
						r2[i] = data->voxel.r2[i] + r2spoil;

					SIM_SUFFIX(bloch_mcc_b1_pdp)(data->voxel.P, 
						(sim_type (*)[N])out, in, r1_d_eff, r2, k_d_eff, m0_d_eff, gb_eff, w1);

				} else
					SIM_SUFFIX(bloch_b1_pdp)((sim_type(*)[3])out, in, 
						data->voxel.r1[0], data->voxel.r2[0] + r2spoil, gb_eff, w1);
			};

			// Choose P-1 because ODE interface treats signal separate and P only describes the number of parameters
			SIM_SUFFIX(ode_direct_sa)(h, data->other.ode_h_min, tol, N, P - 1, xp_d, 
				st, end, call_fun, call_pdy2, call_pdp2);

			// Copy back to single precision
			for (int i = 0; i < P; i++)
				for (int j = 0; j < N; j++)
					xp[i][j] = xp_d[i][j];

			break;
		}
		case SIM_STM: {

			create_sim_matrix(data, P * N, stm_matrix, st, end, r2spoil);  // Not supported for double-precision
			break;
		}
	}

	data->grad.gb[2] = 0.;
}
