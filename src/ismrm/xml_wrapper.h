
struct isrmrm_config_s;

#include "misc/cppwrap.h"

#include "ismrmrd/ismrmrd.h"

#ifdef __cplusplus
#define ISMRMRD_NS(x) ISMRMRD::x
#else
#define ISMRMRD_NS(x) x
#endif

struct ismrm_cpp_state;

extern struct ismrm_cpp_state* ismrm_stream_open(const char* file, _Bool write);
extern void ismrm_stream_close(struct ismrm_cpp_state* s);

extern void ismrm_read_encoding_limits(const char* filename, struct isrmrm_config_s* encoding);
extern void ismrm_read_encoding_limits_from_xml(const char* xml, struct isrmrm_config_s* config);

extern void ismrm_stream_read_meta(struct isrmrm_config_s* config);
extern long ismrm_stream_read_acquisition(struct isrmrm_config_s* config, ISMRMRD_NS(ISMRMRD_Acquisition)* c_acq);

extern void ismrm_stream_write_cfl_image(struct isrmrm_config_s* config, long size0, long size1, _Complex float* buf);
extern void ismrm_stream_write_mag_image(struct isrmrm_config_s* config, long size0, long size1, unsigned short* buf);

extern void ismrm_stream_write_text(struct isrmrm_config_s* config, const char* text);

#include "misc/cppwrap.h"
