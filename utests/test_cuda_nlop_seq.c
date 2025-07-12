/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/mri.h"

#include "misc/misc.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "simu/bloch.h"
#include "simu/pulse.h"
#include "simu/simulation.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nsimu/nlop_seq.h"
#include "nlops/nltest.h"

#include "utest.h"




static bool test_cuda_nlop_pulse(void)
{
	struct pulse_sinc ps = pulse_sinc_defaults;
	pulse_sinc_init(&ps, 0.001, 90., 0., 4., ps.alpha);

	long dims[16] = { [0 ... 15] = 1 };
	int N = ARRAY_SIZE(dims);

	struct sim_config_s sim = sim_config_default_cpu;
	sim_config_set_dims(&sim, N, dims, 1);

	float grad[3] = { 0., 0., 0. };

	const struct nlop_s* nlop_cpu = nlop_pulse_create(sim, CAST_UP(&ps), 0.1, grad);
	const struct nlop_s* nlop_gpu = nlop_gpu_wrapper_F(nlop_pulse_create(sim, CAST_UP(&ps), 0.1, grad));

	bool ok = true;

	for (int i = 0; i < 10; i++)
		ok = ok && compare_nlops(nlop_cpu, nlop_gpu, true, true, true, 1.e-5);

	nlop_free(nlop_cpu);
	nlop_free(nlop_gpu);

	return ok;
}

UT_GPU_REGISTER_TEST(test_cuda_nlop_pulse);

