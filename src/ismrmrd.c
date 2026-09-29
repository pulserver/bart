/* Copyright 2014-2016. The Regents of the University of California.
 * Copyright 2016-2022. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2021-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <limits.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/stream.h"

#include "ismrm/read.h"
#include "ismrm/xml_wrapper.h"


static const char help_str[] = "Import/Export ISMRM raw data files and streams.";


int main_ismrmrd(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* out_file = NULL;


	enum ismrm_write_opts { NO_ISMRM_WRITE=0, ISMRM_WRITE_CFL_IMG, ISMRM_WRITE_USHORT_IMG } write_ismrm = 0;
	bool stream = false;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(false, &out_file, "output"),
	};

	struct isrmrm_config_s config = ismrm_default_config;

	const struct opt_s opts[] = {

		OPTL_INT('m', "measurement", &(config.measurement), "", "select specific measurement (split by flag)"),
		OPTL_INT('o', "overwrite", &(config.overwriting_idx), "idx", "overwrite data idx times"),

		//OPTL_SELECT(0, "interleaved",enum ISMRMRD_SLICE_ORDERING, &(config.slice_ord),ISMRMRD_SLICE_INTERLEAVED, "interleaved slice ordering (1, 3, 5, 2, 4) / (1, 3, 2, 4)"),
		OPTL_SELECT(0, "interleaved-siemens",enum ISMRMRD_SLICE_ORDERING, &(config.slice_ord),ISMRMRD_SLICE_INTERLEAVED_SIEMENS, "interleaved slice ordering (1, 3, 5, 2, 4) / (2, 4, 1, 3)"),

		OPTL_SELECT('\0', "write-ismrm-cfl-img", enum ismrm_write_opts, &write_ismrm, ISMRM_WRITE_CFL_IMG, "Output ISMRM data as cfl image."),
		OPTL_SELECT('\0', "write-ismrm-ush-img", enum ismrm_write_opts, &write_ismrm, ISMRM_WRITE_USHORT_IMG, "Output ISMRM data as unsigned short image."),
		OPTL_SET('\0', "stream", &stream, "Use streaming protocols."),

	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);


	if (NULL == out_file) {

		assert(!stream);
		ismrm_print_xml(in_file);
		return 0;
	}

	long D = DIMS;
	long dims[D];
	md_set_dims(D, dims, 0);

	if (!stream) {

		if (write_ismrm)
			error("Writing ISMRMRD (without streaming) is not implemented.\n");

		ismrm_read_dims(in_file, &config, DIMS, dims);

		complex float* out = create_cfl(out_file, DIMS, dims);

		md_clear(DIMS, dims, out, CFL_SIZE);

		ismrm_read(in_file, &config, DIMS, dims, out);

		unmap_cfl(DIMS, dims, out);

		return 0;
	}

	long pos[D];
	md_set_dims(D, pos, 0);

	unsigned long flags = 0;

	complex float* bart_cfl = NULL;
	stream_t bart_stream = NULL;

	config.ismrm_cpp_state = ismrm_stream_open(write_ismrm ? out_file : in_file, write_ismrm);

	if (write_ismrm) {

		bart_cfl = load_async_cfl(in_file, D, dims);
		bart_stream = stream_lookup(bart_cfl);

		long count = 0;
		flags = ~(MD_BIT(0) | MD_BIT(1));

		assert(1 < dims[0] && 1 < dims[1]);

		do {
			if (bart_stream && !stream_receive_serial(bart_stream, D, pos, count++))
				break;

			if (bart_stream)
				stream_sync_slice(bart_stream, D, dims, flags, pos);

			complex float* src = bart_cfl + md_calc_offset(D, MD_STRIDES(D, dims, 1), pos);

			if (ISMRM_WRITE_CFL_IMG == write_ismrm) {

				ismrm_stream_write_cfl_image(&config, dims[0], dims[1], src);
			} else if (ISMRM_WRITE_USHORT_IMG) {

				unsigned short* buf = md_alloc(2, dims, sizeof(unsigned short));

				long* dimsp = dims;

				NESTED(void, sample_kernel, (const long ipos[]))
				{
					long offset = ipos[0] + dimsp[0] * ipos[1];
					float re = ((float*)(src + offset))[0] ;
					buf[offset] = CLAMP(re, 0., ((float)USHRT_MAX));
				};

				md_parallel_loop(2, dims, MD_BIT(0) | MD_BIT(1), sample_kernel);

				ismrm_stream_write_mag_image(&config, dims[0], dims[1], buf);

				md_free(buf);
			} else {

				error("Not implemented.\n");
			}
		} while (bart_stream || md_next(D, dims, flags, pos));
	} else {

		ismrm_stream_read_dims(&config, D, dims);

		for (int i = 1; i < D; i++)
			if (1 < dims[i] && i != COIL_DIM)
				flags |= MD_BIT(i);

		bart_cfl = create_async_cfl(out_file, flags, D, dims);

		bart_stream = stream_lookup(bart_cfl);

		while (ismrm_stream_read(&config, D, dims, pos, bart_cfl))
			if(bart_stream)
				stream_sync_slice(bart_stream, D, dims, flags, pos);
	}

	unmap_cfl(D, dims, bart_cfl);

	if (write_ismrm)
		ismrm_stream_write_text(&config, "BART Reconstruction finished.\n");

	ismrm_stream_close(config.ismrm_cpp_state);

	return 0;
}

