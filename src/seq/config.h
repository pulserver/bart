
#ifndef _SEQ_CONFIG_H
#define _SEQ_CONFIG_H

#include "misc/cppwrap.h"

#include "misc/mri.h"

#include "seq/gradient.h"
#include "seq/helpers.h"
#include "seq/seq.h"

#define SEQ_FLAGS (PHS1_FLAG|PHS2_FLAG|COEFF_FLAG|COEFF2_FLAG|TIME_FLAG|TIME2_FLAG|SLICE_FLAG|AVG_FLAG|BATCH_FLAG|CSHIFT_FLAG)

enum seq_type {

	SEQ_TYPE_FLASH = 1, // BOOST v0.1
	SEQ_TYPE_MINIFLASH,
};

extern const int seq_loop_order_avg_inner[DIMS];
extern const int seq_loop_order_avg_outer[DIMS];
extern const int seq_loop_order_multislice[DIMS];
extern const int seq_loop_order_asl[DIMS];

extern void seq_copy_order(struct seq_config* seq);

enum flash_contrast {

	SEQ_CONTRAST_NO_SPOILING = 0,
	SEQ_CONTRAST_RF_RANDOM,
	SEQ_CONTRAST_RF_SPOILED
};


enum pe_mode {

	SEQ_PEMODE_TURN = 2,
	SEQ_PEMODE_RAGA,
	SEQ_PEMODE_MEMS_HYB,
	SEQ_PEMODE_CARTESIAN,
	SEQ_PEMODE_CARTESIAN_LINEAR
};


enum cest_saturation_type {

	SEQ_CEST_NONE,
	SEQ_CEST_GAUSS,
	SEQ_CEST_OC
};


enum cest_offset_type {

	SEQ_CEST_OFFSET_EQUIDISTANT,
	SEQ_CEST_OFFSET_PHANTOM,
	SEQ_CEST_OFFSET_INVIVO
};

enum asl_label_type { 

	SEQ_ASL_NONE,
	SEQ_ASL_PCASL,
};

struct seq_phys {

	double tr;
	double te;
	double te_delta;

	double dwell;
	double os;
	double asym_echo;

	enum flash_contrast contrast;
	double rf_duration;
	double flip_angle; // deg
	double bwtp;
};

struct seq_geom {

	double fov;
	double slice_thickness;
	double slab_os; // 1.0 no oversampling
	double shift[SEQ_MAX_SLICES][3]; // [ro, pe, slice]
	double rot[SEQ_MAX_SLICES][3][3];

	int baseres;

	int mb_factor;
	double sms_distance;
};

struct seq_enc {

	enum pe_mode pe_mode;
	int tiny;
	unsigned long aligned_flags;
	enum seq_order order;
	int is3D;
};

struct seq_magn {

	enum mag_prep mag_prep;
	double ti;
	long prep_scans;
	double init_delay;
	double inv_delay_time;
};

struct seq_sys {

	double gamma; // Hz/T
	double b0; // T
	struct grad_limits grad; // inv_slew_rate in s / (T/m), max_amplitude in T/m
	double coil_control_lead;
	double min_duration_ro_rf;
	double raster_grad;
	double raster_rf;
	double raster_dwell;
};


struct seq_trigger {

	// Physiological triggering: scanner waits on event
	enum trigger_type type; 
	double delay_time;
	int pulses;

	// Trigger output: scanner notifies other hardware
	int trigger_out;
};

struct asl_pulse {

	double rf_duration;
	double flip_angle;
};

struct seq_asl {
				
	enum asl_label_type label_type;		// ASL labeling mode (currently only PCASL is supported)
	double ld;				// Labeling duration for PCASL in s
	double pld;				// Post-labeling delay in s
	double ampl_grad_sli;			// Amplitude of slice-selection gradient during labeling in T/m
	double pulse_spacing;			// Spacing between two consecutive hanning pulses in s
	struct asl_pulse hanning;		// Hanning pulse parameters
	int label_slice_index;			// Chronological index of labeling slice
};

struct seq_cest {

	enum cest_saturation_type sat_type;
	long sat_pulses;
	double sat_pulse_pause;
	double gauss_pulse_duration;
	double gauss_pulse_fa;
	double oc_pulse_b1_scaling;
	enum cest_offset_type offset_type;
	double offset_first;
	double offset_last;
	double offset_increment;
	double offset_pause;
};


struct seq_config {

	enum seq_type seq_type;

	struct seq_phys phys;
	struct seq_geom geom;
	struct seq_enc enc;
	struct seq_magn magn;
	struct seq_trigger trigger;
	struct seq_cest cest;
	struct seq_asl asl;
	struct seq_sys sys; 

	int order[DIMS];
	long loop_dims[DIMS];
};

extern const struct seq_config seq_config_defaults_flash;
extern const struct seq_config seq_config_defaults_miniflash;

#include "misc/cppwrap.h"

#endif	// _SEQ_CONFIG_H

