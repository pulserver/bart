/* Copyright 2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Daniel Mackner
 *
 * References:
 *
 * Layton K, Kroboth S, Jia F, Littin S, Yu H, Leupold J,
 * Nielsen JF, Stöcker T, Zaitsev M. Pulseq: A rapid and
 * hardware-independent pulse sequence prototyping framework.
 * Magn Reson Med 2017; 77:1544-1552.
 */

#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <assert.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "seq/config.h"
#include "seq/kernel.h"
#include "seq/seq.h"
#include "seq/pulseq.h"

#define VEC_LEN(x) ({ auto _x = (x); (_x ? _x->len : 0); })

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char* help_str = "Read/write pulseq files.";

int main_pulseq(int argc, char *argv[argc])
{
	const char* seq_file = NULL;
	const char* grad_file = NULL;
	const char* adc_file = NULL;
	const char* pulse_file = NULL;
	const char* shapes_file = NULL;
	const char* events_file = NULL;

	struct arg_s args[] = {

		ARG_INOUTFILE(true, &seq_file, "pulseq-file"),
		ARG_OUTFILE(false, &grad_file, "gradients"),
		ARG_OUTFILE(false, &adc_file, "adcs"),
		ARG_OUTFILE(false, &pulse_file, "pulse"),
	};


	bool write = false;

	const struct opt_s opts[] = {

		OPT_SET('w', &write, "write"),
		OPT_INOUTFILE('F', &shapes_file, "file", "RF Shapes file"),
		OPT_INOUTFILE('E', &events_file, "file", "Events file"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	complex float* shapes = NULL;
	complex float* events = NULL;

	bart_dim_t shape_dims[DIMS];
	bart_dim_t event_dims[DIMS];


	FILE *fp = fopen(seq_file, (write) ? "w+" : "r");

	if ((write && (NULL == fp)) || !fp)
		error("Opening .seq-file failed.\n");

	struct pulseq ps;

	if (write) {

		if ((NULL == shapes_file) || (NULL == events_file))
			error("RF Shapes/Sequence Events files must be provided for writing Pulseq files.\n");

		shapes = load_cfl(shapes_file, DIMS, shape_dims);
		events = load_cfl(events_file, DIMS, event_dims);

		struct seq_config conf = seq_config_defaults_flash;

		pulseq_init(&ps, &conf);

		struct rf_shape rf_shapes[shape_dims[TIME_DIM]];
		seq_pulse_shapes_from_cfl(shape_dims[TIME_DIM], rf_shapes, DIMS, shape_dims, shapes);
		pulse_shapes_to_pulseq(&ps, shape_dims[TIME_DIM], rf_shapes);

		bart_dim_t pos[DIMS] = { };

		bart_stride_t strs[DIMS];
		md_calc_strides(DIMS, strs, event_dims, 1);


		double start_block = 0.;
		struct seq_event seq_events[event_dims[PHS1_DIM]];

		double tr = seq_events_cfl_find_tr(DIMS, event_dims, events);
		if (0 > tr)
			debug_printf(DP_WARN, "Could not find TR in seq events! SEQ_BLOCK_KERNEL_IMAGE not possible.\n");

		do {

			int E = seq_events_from_cfl(event_dims[PHS1_DIM], seq_events, &start_block, DIMS, event_dims, events + md_calc_offset(DIMS, strs, pos));

			enum seq_block mode = SEQ_BLOCK_UNDEFINED;
			if (seq_events_is_image_block(E, seq_events) && (0 < tr))
				mode = SEQ_BLOCK_KERNEL_IMAGE;

			events_to_pulseq(&ps, mode, tr, shape_dims[TIME_DIM], rf_shapes, E, seq_events);

		} while (md_next(DIMS, event_dims, TIME_FLAG, pos));

		pulseq_writef(fp, &ps);
		fclose(fp);

	} else {

		error("Pulseq read not implemented - use \"-w\"");
	}
	return 0;
}

