/* Copyright 2022. Uecker Lab. University Center Göttingen.
 * Copyright 2024-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdbool.h>
#include <stdexcept>

#include <cassert>
#include <climits>
#include <ext/stdio_filebuf.h>
#include <iostream>
#include <fstream>

#include "misc/misc.h"
#include "misc/debug.h"

#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/dataset.h"
#include "ismrmrd/meta.h"
#include "ismrmrd/serialization.h"
#include "ismrmrd/serialization_iostream.h"
#include "ismrmrd/xml.h"

#include "xml_wrapper.h"
#include "read.h"



static struct limit_s get_limit(ISMRMRD::Optional<ISMRMRD::Limit>& src)
{
	struct limit_s ret = ismrmrd_default_limit;

	if (src.is_present()) {

		ret.center = src.get().center;
		ret.min_hdr = src.get().minimum;
		ret.max_hdr = src.get().maximum;

		ret.size = ret.max_hdr + 1;
		ret.size_hdr = ret.max_hdr + 1;
	}

	return ret;
}


extern "C" void ismrm_read_encoding_limits(const char* filename, struct isrmrm_config_s* config)
{
	ISMRMRD::ISMRMRD_Dataset d;
	ismrmrd_init_dataset(&d, filename, "/dataset");
	ismrmrd_open_dataset(&d, false);

	const char* xml =  ismrmrd_read_header(&d);

	ismrm_read_encoding_limits_from_xml(xml, config);

	xfree(xml);

	ismrmrd_close_dataset(&d);
}


static void ismrm_read_encoding_limits_from_hdr(ISMRMRD::IsmrmrdHeader& h, struct isrmrm_config_s* config);


extern "C" void ismrm_read_encoding_limits_from_xml(const char* xml, struct isrmrm_config_s* config)
{
	ISMRMRD::IsmrmrdHeader h;
	deserialize(xml, h);
	ismrm_read_encoding_limits_from_hdr(h, config);
}


static void ismrm_read_encoding_limits_from_hdr(ISMRMRD::IsmrmrdHeader& h, struct isrmrm_config_s* config)
{
	if (config->idx_encoding >= (int)h.encoding.size())
		error("ISMRMD inconsistent number of encodings!\n");

	ISMRMRD::Encoding& encoding = h.encoding[config->idx_encoding];

	config->limits[ISMRMRD_READ_DIM].size		= encoding.encodedSpace.matrixSize.x;
	config->limits[ISMRMRD_COIL_DIM].size		= 1;

	if (h.acquisitionSystemInformation.is_present()) {

		if (h.acquisitionSystemInformation.get().receiverChannels.is_present()) {

			config->limits[ISMRMRD_COIL_DIM].size = h.acquisitionSystemInformation.get().receiverChannels.get();
			config->limits[ISMRMRD_COIL_DIM].size_hdr = h.acquisitionSystemInformation.get().receiverChannels.get();
		}
	}

	config->limits[ISMRMRD_PHS1_DIM] 		= get_limit(encoding.encodingLimits.kspace_encoding_step_1);
	config->limits[ISMRMRD_PHS1_DIM].size		= encoding.encodedSpace.matrixSize.y;
	config->limits[ISMRMRD_PHS1_DIM].size_hdr	= encoding.encodedSpace.matrixSize.y;

	config->limits[ISMRMRD_PHS2_DIM] 		= get_limit(encoding.encodingLimits.kspace_encoding_step_2);
	config->limits[ISMRMRD_PHS2_DIM].size 		= encoding.encodedSpace.matrixSize.z;
	config->limits[ISMRMRD_PHS2_DIM].size_hdr	= encoding.encodedSpace.matrixSize.z;

	config->limits[ISMRMRD_AVERAGE_DIM] 		= get_limit(encoding.encodingLimits.average);
	config->limits[ISMRMRD_SLICE_DIM] 		= get_limit(encoding.encodingLimits.slice);
	config->limits[ISMRMRD_CONTRAST_DIM] 		= get_limit(encoding.encodingLimits.contrast);
	config->limits[ISMRMRD_PHASE_DIM] 		= get_limit(encoding.encodingLimits.phase);
	config->limits[ISMRMRD_REPETITION_DIM]		= get_limit(encoding.encodingLimits.repetition);
	config->limits[ISMRMRD_SET_DIM] 		= get_limit(encoding.encodingLimits.set);
	config->limits[ISMRMRD_SEGMENT_DIM] 		= get_limit(encoding.encodingLimits.segment);
}


struct ismrm_cpp_state {

	std::istream* is;
	ISMRMRD::IStreamView* rs;
	ISMRMRD::ProtocolDeserializer* deserializer;

	std::ostream* os;
	ISMRMRD::OStreamView* ws;
	ISMRMRD::ProtocolSerializer* serializer;
};

extern "C" struct ismrm_cpp_state* ismrm_stream_open(const char* file, bool write)
{
	struct ismrm_cpp_state* ret = (struct ismrm_cpp_state*) malloc(sizeof *ret);
	ret->deserializer = NULL;
	ret->serializer = NULL;
	ret->rs = NULL;
	ret->ws = NULL;
	ret->is = NULL;

	if (write) {

		std::ostream* os;
		if (0 != strcmp("-", file)) {

			ret->os = new std::ofstream(file, std::ifstream::binary | std::ios::binary);
			os = ret->os;
		} else {

			os = &std::cout;
		}

		ret->ws = new ISMRMRD::OStreamView(*os);
		ret->serializer = new ISMRMRD::ProtocolSerializer(*ret->ws);
	} else {

		std::istream* is;
		if (0 != strcmp("-", file)) {

			ret->is = new std::ifstream(file, std::ifstream::binary | std::ios::binary);
			is = ret->is;
		} else {

			is = &std::cin;
		}

		ret->rs = new ISMRMRD::IStreamView(*is);
		ret->deserializer = new ISMRMRD::ProtocolDeserializer(*ret->rs);
	}

	return ret;
}


extern "C" void ismrm_stream_close(struct ismrm_cpp_state* s)
{
	if (NULL != s->deserializer)
		delete s->deserializer;
	// idk
	if (NULL != s->rs)
		delete s->rs;
	if (NULL != s->is)
		delete s->is;

	if (NULL != s->serializer) {

		s->serializer->close();
		delete s->serializer;
	}

	if (NULL != s->ws)
		delete s->ws;
	if (NULL != s->os)
		delete s->os;
}


extern "C" void ismrm_stream_read_meta(struct isrmrm_config_s* config)
{
	struct ismrm_cpp_state* s = config->ismrm_cpp_state;

	try {

		uint16_t type;

		while (ISMRMRD::ISMRMRD_MESSAGE_HEADER != (type = s->deserializer->peek())) {

			if (ISMRMRD::ISMRMRD_MESSAGE_CONFIG_FILE == type) {

				ISMRMRD::ConfigFile conf;
				s->deserializer->deserialize(conf);

			} else if (ISMRMRD::ISMRMRD_MESSAGE_CONFIG_TEXT == type) {

				ISMRMRD::ConfigText conf;
				s->deserializer->deserialize(conf);

			} else if (ISMRMRD::ISMRMRD_MESSAGE_TEXT == type) {

				ISMRMRD::TextMessage tm;
				s->deserializer->deserialize(tm);
				debug_printf(DP_WARN, "Message from ISMRM stream: %s", tm.message.c_str());

			} else {

				error("Unexpected message type: %d.\n", type);
			}
		}

		assert (type == ISMRMRD::ISMRMRD_MESSAGE_HEADER);

		ISMRMRD::IsmrmrdHeader hdr;
		s->deserializer->deserialize(hdr);

		ismrm_read_encoding_limits_from_hdr(hdr, config);

	} catch(std::runtime_error& e) {

		error("BART ISMRMRD Wrapper: Exception thrown: %s\n", e.what());
	}
}

static bool ignore_next_message(struct isrmrm_config_s* config)
{
	struct ismrm_cpp_state* s = config->ismrm_cpp_state;
	auto d = s->deserializer;

#define ismrm_l1types(m)\
	m(ISMRMRD::Acquisition, ISMRMRD::ISMRMRD_MESSAGE_ACQUISITION)			\
	m(ISMRMRD::Waveform,	ISMRMRD::ISMRMRD_MESSAGE_WAVEFORM)			\
	m(ISMRMRD::TextMessage,	ISMRMRD::ISMRMRD_MESSAGE_TEXT)

#define ismrm_img_types(m)\
	m(ISMRMRD::Image<unsigned short>,		ISMRMRD::ISMRMRD_USHORT)	\
	m(ISMRMRD::Image<short>,			ISMRMRD::ISMRMRD_SHORT)		\
	m(ISMRMRD::Image<unsigned int>,			ISMRMRD::ISMRMRD_UINT)		\
	m(ISMRMRD::Image<int>,				ISMRMRD::ISMRMRD_INT)		\
	m(ISMRMRD::Image<float>,			ISMRMRD::ISMRMRD_INT)		\
	m(ISMRMRD::Image<double>,			ISMRMRD::ISMRMRD_DOUBLE)	\
	m(ISMRMRD::Image<std::complex<float> >,		ISMRMRD::ISMRMRD_CXFLOAT)	\
	m(ISMRMRD::Image<std::complex<double> >,	ISMRMRD::ISMRMRD_CXDOUBLE)	\

#define ismrm_array_types(m)\
	m(ISMRMRD::NDArray<unsigned short>,		ISMRMRD::ISMRMRD_USHORT)	\
	m(ISMRMRD::NDArray<short>,			ISMRMRD::ISMRMRD_SHORT)		\
	m(ISMRMRD::NDArray<unsigned int>,		ISMRMRD::ISMRMRD_UINT)		\
	m(ISMRMRD::NDArray<int>,			ISMRMRD::ISMRMRD_INT)		\
	m(ISMRMRD::NDArray<float>,			ISMRMRD::ISMRMRD_INT)		\
	m(ISMRMRD::NDArray<double>,			ISMRMRD::ISMRMRD_DOUBLE)	\
	m(ISMRMRD::NDArray<std::complex<float> >,	ISMRMRD::ISMRMRD_CXFLOAT)	\
	m(ISMRMRD::NDArray<std::complex<double> >,	ISMRMRD::ISMRMRD_CXDOUBLE)	\

#define handle(type, type_no)	\
	if (type_id == type_no) {	\
		type x;			\
		d->deserialize(x);	\
		return true;		\
	}

	{
		//enum ISMRMRD::ISMRMRD_MESSAGE_ID
		uint16_t type_id = d->peek();
		ismrm_l1types(handle)
	}

	if (d->peek() == ISMRMRD::ISMRMRD_MESSAGE_IMAGE) {

		int type_id = d->peek_image_data_type();
		ismrm_img_types(handle)
	}

	if (d->peek() == ISMRMRD::ISMRMRD_MESSAGE_NDARRAY) {

		int type_id = d->peek_image_data_type();
		ismrm_img_types(handle)
	}

	return false;
}

extern "C" bart_dim_t ismrm_stream_read_acquisition(struct isrmrm_config_s* config, ISMRMRD::ISMRMRD_Acquisition* c_acq)
{
	struct ismrm_cpp_state* s = config->ismrm_cpp_state;

	try {

		uint16_t type;

		while (ISMRMRD::ISMRMRD_MESSAGE_ACQUISITION != (type = s->deserializer->peek())) {

			if (ISMRMRD::ISMRMRD_MESSAGE_CLOSE == type)
				return 0;

			if (ISMRMRD::ISMRMRD_MESSAGE_TEXT == type) {

				ISMRMRD::TextMessage tm;
				s->deserializer->deserialize(tm);
				debug_printf(DP_WARN, "Message from ISMRM stream: %s", tm.message.c_str());
				continue;
			}

			if(!ignore_next_message(config))
				error("BART ISMRMRD Wrapper: Unexpected Non-Acquisition message.\n");
		}

		assert(ISMRMRD::ISMRMRD_MESSAGE_ACQUISITION == type);
		ISMRMRD::Acquisition a;
		s->deserializer->deserialize(a);

		c_acq->head = a.getHead();

		size_t data_size = a.getDataSize();
		c_acq->data = (complex_float_t*)xmalloc(data_size);
		memcpy(c_acq->data, a.getDataPtr(), data_size);

		if (LONG_MAX < data_size)
			error("BART ISMRMRD Wrapper: Too large acquisition.\n");

		return (bart_dim_t)data_size;

	} catch(std::runtime_error& e) {

		error("BART ISMRMRD Wrapper: Exception thrown: %s\n", e.what());
	}
}

template<typename T>
static void ismrm_set_metadata(ISMRMRD::Image<T>& img)
{
	//FIXME
	float fov[3] = { 200, 200, 5 };
	float pos[3] = { 0, 0, 0 };
	float table_pos[3] = { 0, 0, -1591 };
	float read_dir[3] = { -1, 0, 0 };
	float phase_dir[3] = { 0, 1, 0 };
	float slice_dir[3] = { 0, 0, 1 };

	img.setSlice(0);
	img.setFieldOfView(fov[0], fov[1], fov[2]);
	img.setPosition(pos[0], pos[1], pos[2]);
	img.setPatientTablePosition(table_pos[0], table_pos[1], table_pos[2]);
	img.setReadDirection(read_dir[0], read_dir[1], read_dir[2]);
	img.setPhaseDirection(phase_dir[0], phase_dir[1], phase_dir[2]);
	img.setSliceDirection(slice_dir[0], slice_dir[1], slice_dir[2]);

	ISMRMRD::MetaContainer meta;

	meta.append("ImageRowDir", read_dir[0]);
	meta.append("ImageRowDir", read_dir[1]);
	meta.append("ImageRowDir", read_dir[2]);

	meta.append("ImageColumnDir", phase_dir[0]);
	meta.append("ImageColumnDir", phase_dir[1]);
	meta.append("ImageColumnDir", phase_dir[2]);
	meta.append("Keep_image_geometry", "0");

	try {

		std::stringstream meta_string_stream;
		ISMRMRD::serialize(meta, meta_string_stream);

		img.setAttributeString(meta_string_stream.str().c_str());

	} catch(std::runtime_error& e) {

		error("BART ISMRMRD Wrapper: Exception thrown: %s\n", e.what());
	}
}

template<typename T>
static void ismrm_send_img(const ISMRMRD::Image<T> &img, const struct isrmrm_config_s* config)
{
	try {

		config->ismrm_cpp_state->serializer->serialize(img);
	} catch(std::runtime_error& e) {

		error("BART ISMRMRD Wrapper: Exception thrown: %s\n", e.what());
	}
}

extern "C" void ismrm_stream_write_cfl_image(struct isrmrm_config_s* config, long size0, long size1, _Complex float* buf)
{
	ISMRMRD::Image<std::complex<float> > img(size0, size1);

	std::complex<float>* ptr = img.getDataPtr();
	memcpy(ptr, buf, size0 * size1 * sizeof(float) * 2);

	img.setImageType(ISMRMRD::ISMRMRD_IMTYPE_COMPLEX);

	ismrm_set_metadata(img);
	ismrm_send_img(img, config);
}

extern "C" void ismrm_stream_write_mag_image(struct isrmrm_config_s* config, long size0, long size1, unsigned short* buf)
{
	ISMRMRD::Image<unsigned short> img(size0, size1);

	unsigned short* ptr = img.getDataPtr();
	memcpy(ptr, buf, size0 * size1 * sizeof(unsigned short));

	img.setImageType(ISMRMRD::ISMRMRD_IMTYPE_IMAG);

	ismrm_set_metadata(img);
	ismrm_send_img(img, config);
}

extern "C" void ismrm_stream_write_text(struct isrmrm_config_s* config, const char* text)
{
	try {
		ISMRMRD::TextMessage tm;
		tm.message = text;
		config->ismrm_cpp_state->serializer->serialize(tm);
	}
	catch(std::runtime_error& e) {
		error("BART ISMRMRD Wrapper: Exception thrown: %s\n", e.what());
	}

}
