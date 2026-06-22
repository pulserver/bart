/* Copyright 2013-2016. The Regents of the University of California.
 * Copyright 2015-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2021-2026. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013, 2015 Martin Uecker
 * 2015 Jonathan Tamir
 */

#include <complex.h>
#include <string.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/io.h"
#include "misc/stream.h"


#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] =
	"Join input files along {dimensions}. All other dimensions must have the same size.\n"
	"\t Example 1: join 0 slice_001 slice_002 slice_003 full_data\n"
	"\t Example 2: join 0 `seq -f \"slice_%%03g\" 0 255` full_data";



int main_join(int argc, char* argv[argc])
{
	int count = 0;
	int dim = -1;
	const char** in_files = NULL;
	const char* out_file = NULL;
	bool stream = false;

	struct arg_s args[] = {

		ARG_INT(true, &dim, "dimension"),
		ARG_TUPLE(true, &count, 1, { OPT_INFILE, sizeof(char*), &in_files, "input" }),
		ARG_INOUTFILE(true, &out_file, "output"),
	};


	bool append = false;

	const struct opt_s opts[] = {

		OPT_SET('a', &append, "append - only works for cfl files!"),
		OPT_SET('s', &stream, "Stream output along input stream dimensions."),
		// stream dimensions of output:
		//   if at least one input is a stream AND all input stream flags contain MD_BIT(join dimension)
		//   then stream along flags present in all inputs.
		//   All stream inputs must have same flags.
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;

	assert(dim < N);

	if (append) {

		if (stream)
			error("Streaming and appending are incompatible!");

		count += 1;

		assert(count > 1);

		int len = (int)strlen(out_file);
		char buf[len + 5];
		strcpy(buf, out_file);
		strcat(buf, ".cfl");

		if (-1 == access(buf, F_OK)) {

			// make sure we do not have any other file format

			strcpy(buf, out_file);
			strcat(buf, ".coo");
			assert(-1 == access(buf, F_OK));

			strcpy(buf, out_file);
			strcat(buf, ".ra");
			assert(-1 == access(buf, F_OK));

			count--;
			append = false;
		}
	}

	long in_dims[count][N];
	const complex float* in_data[count];

	long offsets[count];
	long sum = 0;

	unsigned long any_flags = 0UL;
	unsigned long all_flags = ~0UL;

	// figure out size and stream flags of output
	for (int l = 0, i = 0; i < count; i++) {

		const char* name = NULL;

		if (append && (i == 0)) {

			name = out_file;

		} else {

			name = in_files[l++];
		}

		debug_printf(DP_DEBUG1, "loading %s\n", name);

		if (stream) {

			in_data[i] = load_async_cfl(name, N, in_dims[i]);
			stream_t stream_in = stream_lookup(in_data[i]);

			if (stream_in) {

				unsigned long iflags = stream_get_flags(stream_in);
				any_flags |= iflags;
				all_flags &= iflags;
			}


		} else {

			in_data[i] = load_cfl(name, N, in_dims[i]);
		}

		offsets[i] = sum;

		sum += in_dims[i][dim];

		for (int j = 0; j < N; j++)
			assert((dim == j) || (in_dims[0][j] == in_dims[i][j]));

		if (append && (i == 0))
			unmap_cfl(N, in_dims[i], in_data[i]);
	}

	unsigned long stream_out_flags = 0UL;

	if (stream && any_flags) {

		if (all_flags != any_flags)
			error("All input streams must contain the same set of flags.\n");

		if (!(all_flags & MD_BIT(dim)))
			error("Input streams must contain join dimension.\n");

		stream_out_flags = all_flags;

	} else {

		stream = false;
	}

	long out_dims[N];

	for (int i = 0; i < N; i++)
		out_dims[i] = in_dims[0][i];

	out_dims[dim] = sum;

	if (append) {

		if (md_check_dimensions(N - dim - 1, out_dims + dim + 1, 0UL)) {

			debug_printf(DP_INFO, "dim: %d; out_dims + dim + 1:\n", dim);
			debug_print_dims(DP_INFO, N - dim - 1, out_dims + dim + 1);
			debug_printf(DP_WARN, "True appending impossible, dimensions after join-dim"
					      "are not trivial - rewriting appended file.\n");

			// keep a reference to original input data
			in_data[0] = load_cfl(out_file, N, in_dims[0]);
		} else {

			// here, io_close will prevent the subsequent create_cfl-call
			// from unlinking and recreating the output, thus keeping
			// it's contents (input 0) and appending to it
			io_close(out_file);
		}
	}

	complex float* out_data = NULL;

	if (stream)
		out_data = create_async_cfl(out_file, stream_out_flags, N, out_dims);
	else
		out_data = create_cfl(out_file, N, out_dims);

	stream_t stream_out = stream_lookup(out_data);

	if (append && md_check_dimensions(N - dim - 1, out_dims + dim + 1, 0UL)) {

		// fake append: rewriting the contents of the output
		md_copy2(N, in_dims[0], MD_STRIDES(N, out_dims, CFL_SIZE), out_data,
			MD_STRIDES(N, in_dims[0], CFL_SIZE), in_data[0], CFL_SIZE);
		unmap_cfl(N, in_dims[0], in_data[0]);
	}

#pragma omp parallel for num_threads(count)
	for (int i = 0; i < count; i++) {

		if (append && (0 == i))
			continue;

		stream_t stream_in = stream_lookup(in_data[i]);
		long counter = 0;

		// assert input slices are complete:
		assert((!stream) || ((!stream_in) || (stream_get_flags(stream_in) == stream_out_flags)));

		long opos[N];
		long ipos[N];
		md_singleton_strides(N, ipos);
		md_singleton_strides(N, opos);

		long slc_dims[DIMS];
		md_select_dims(DIMS, ~stream_out_flags, slc_dims, in_dims[i]);

		// join input slices in order of arrival:

		bool input_loop = stream && stream_in;
		do {

			if (input_loop && !stream_receive_serial(stream_in, DIMS, ipos, counter))
				break;

			md_copy_dims(N, opos, ipos);
			opos[dim] += offsets[i];

			md_move_block(N, slc_dims, opos, out_dims, out_data, ipos, in_dims[i], in_data[i], CFL_SIZE);

			if (stream)
				stream_sync_slice(stream_out, DIMS, out_dims, stream_out_flags, opos);

			counter++;

		} while (input_loop || md_next(N, in_dims[i], stream_out_flags, ipos)) ;

		debug_printf(DP_DEBUG1, "done copying file %d\n", i);

		unmap_cfl(N, in_dims[i], in_data[i]);
	}

	unmap_cfl(N, out_dims, out_data);

	xfree(in_files);

	return 0;
}

