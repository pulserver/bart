#ifndef __SEQ_UI_ENUMS_H
#define __SEQ_UI_ENUMS_H

#include "seq/custom_ui.h"

#define SEQ_CUSTOM_UI_IDX_LONG(M)	\
	M(PE_MODE)			\
	M(CONTRAST)			\
	/* bool */			\
	M(RECO)				\
	M(SMS)				\
	/* long */			\
	M(CMD)				\
	/* long array */		\
	M(TINY)				\
	M(PREP_SCANS)			\
	M(RF_DURATION_US)		\
	M(INIT_DELAY)			\
	M(INVERSIONS)			\
	M(INV_DELAY)			\
	M(MB_FACTOR)			\
	M(RAGA_ALIGNED_FLAGS)		\
	M(CEST_SATURATION)		\
	M(CEST_OFFSET_TYPE)		\
	M(CEST_SAT_PULSES)		\
	M(ASL_MODE)

enum custom_idx_long {
#define enum_entry(name) SEQ_UI_IDX_LONG_##name,
SEQ_CUSTOM_UI_IDX_LONG(enum_entry)
#undef enum_entry
}; // max 64


#define SEQ_CUSTOM_UI_IDX_DOUBLE(M)	\
	M(BWTP)				\
	M(ASYM_ECHO)			\
	M(CEST_OFFSET_PAUSE_S)		\
	M(CEST_SAT_PULSE_PAUSE_MS)	\
	M(CEST_GAUSS_duration_MS)	\
	M(CEST_GAUSS_FA)		\
	M(CEST_OC_B1_SCALING)		\
	M(CEST_OFFSET_FIRST_PPM)	\
	M(CEST_OFFSET_LAST_PPM)		\
	M(CEST_OFFSET_INCREMENT_PPM)	\
	M(ASL_LD)			\
	M(ASL_PLD)

enum custom_idx_double {
#define enum_entry(name) SEQ_UI_IDX_DOUBLE_##name,
SEQ_CUSTOM_UI_IDX_DOUBLE(enum_entry)
#undef enum_entry
}; // max 16

#endif // __SEQ_UI_ENUMS_H
