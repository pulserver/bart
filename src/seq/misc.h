
#ifndef _SEQ_MISC_H
#define _SEQ_MISC_H
#include "misc/dimtypes.h"

#define ERROR_LIST \
	X(ERROR_SAMPLE_RF, -11)				\
	X(ERROR_SEQ_BLOCK, -12)				\
	X(ERROR_SETTING_ASL, -51)			\
	X(ERROR_SETTING_CONTRAST, -100)			\
	X(ERROR_SETTING_BSSFP, -101)			\
	X(ERROR_SETTING_DIM, -102)			\
	X(ERROR_PREP_SCANS, -103)			\
	X(ERROR_MAG_PREP, -104)				\
	X(ERROR_CEST_TRIGGER, -105)			\
	X(ERROR_TRIGGER, -106)				\
	X(ERROR_SETTING_ORDER, -110)			\
	X(ERROR_SETTING_PEMODE, -120)			\
	X(ERROR_SETTING_RAGA_AL, -121)			\
	X(ERROR_SETTING_SPOKES_RAGA, -122)		\
	X(ERROR_SETTING_SPOKES_EVEN, -123)		\
	X(ERROR_SETTING_ASYM_MECO, -124)		\
	X(ERROR_SETTING_ASYM_ECHO, -125)		\
	X(ERROR_PREP_GRAD_SLI, -201)			\
	X(ERROR_PREP_GRAD_SLI_REPH, -202)		\
	X(ERROR_PREP_GRAD_PE3D_ENC, -203)		\
	X(ERROR_PREP_GRAD_PE3D_REW, -204)		\
	X(ERROR_SLI_TIMING, -221)			\
	X(ERROR_PREP_GRAD_RO_DEPH, -301)		\
	X(ERROR_PREP_GRAD_RO_RO, -302)			\
	X(ERROR_PREP_GRAD_RO_BLIP, -303)		\
	X(ERROR_PREP_GRAD_RO_REPH, -304)		\
	X(ERROR_BLIP_TIMING, -310)			\
	X(ERROR_RO_TIMING, -321)			\
	X(ERROR_ROT_ANGLE, -341)			\
	X(ERROR_MAX_GRAD_RO_SLI, -351)			\
	X(ERROR_PREP_GRAD_SP_READ, -401)		\
	X(ERROR_PREP_GRAD_SP_SLICE, -402)		\
	X(ERROR_MAX_GRAD_SPOILER_READ, -451)		\
	X(ERROR_MAX_GRAD_SPOILER, -452)			\
	X(ERROR_END_FLAT_KERNEL, -901)

enum seq_error {
#define X(name, val) name = val,
    ERROR_LIST
#undef X
};

static inline const char *error_string(enum seq_error e) {
    switch (e) {
#define X(name, val) case name: return #name;
        ERROR_LIST
#undef X
        default: return "ERROR_UNKNOWN";
    }
}


struct seq_config;

extern double slice_amplitude(const struct seq_config* seq);
extern double slice_momentum_to_rephase(const struct seq_config* seq);
extern double ro_amplitude(const struct seq_config* seq);

extern double round_up_raster(double time, double raster_time);

struct grad_trapezoid;
struct grad_limits;
extern int gradient_prepare_with_timing(struct grad_trapezoid* grad, double moment, const struct seq_config* seq);

extern bart_dim_t get_slices(const struct seq_config* seq);

#endif // _SEQ_MISC_H

