#include <math.h>
#include <stdio.h>

#include "num/multind.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/kernel.h"
#include "seq/flash.h"
#include "seq/seq.h"
#include "seq/helpers.h"
#include "seq/opts.h"

#include "utest.h"

#define FLASH_EVENTS 14
#define FLASH_EVENTS_MECO 68
#define FLASH_EVENTS_MECO_64 581
#define FLASH_EVENTS_SPOILED 22


static bool test_command(void)
{
	struct seq_config seq = seq_config_defaults;
	seq.geom.baseres = 250;

	if (!read_config_from_str(&seq, 200, "bart seq --BR 200 --FOV 0.305\0"))
		return false;

	if (200 != seq.geom.baseres)
		return false;

	if (0.305 != seq.geom.fov)
		return false;

	return true;
}

UT_REGISTER_TEST(test_command);

static bool test_print_command(void)
{
	struct seq_config conf = seq_config_defaults;

	const bool print_debug = false;
	char config_info_tmp[7852];

	if (print_debug) {

		seq_print_info_config(7852, config_info_tmp, &conf);
		printf("%s\n", config_info_tmp);
	}

	struct seq_opts seq_opts = seq_opts_defaults;

	char buf[5000];
	int ctr = seq_cmdline_print(5000, buf, &conf, &seq_opts);

	if (0 > ctr)
		return false;

	struct seq_config conf_ref = seq_config_defaults;

	if (0 != memcmp(&conf, &conf_ref, sizeof(struct seq_config)))
		return false;

	if (print_debug) {

		seq_print_info_config(7852, config_info_tmp, &conf);
		printf("%s\n", config_info_tmp);
	}

	return true;
}

UT_REGISTER_TEST(test_print_command);


static bool test_print(void)
{
	struct seq_config seq = seq_config_defaults;

	static char tooltip[7852]; // 8192 (defined in sequence) - 340 (already used)
	int a = seq_print_info_config(7852, tooltip, &seq);

	if (0 > a)
		 return false;

	return true;
}

UT_REGISTER_TEST(test_print);


static bool test_flash_events(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = SEQ_BLOCK_KERNEL_NOISE;
	E = flash(E, ev, &seq_state, &seq);

	if ((FLASH_EVENTS - 1) != E) // no rf
		return false;

	int e_adc = events_idx(0, SEQ_EVENT_ADC, E, ev);
	if (0 == (ev[e_adc].adc.flags & SEQ_ADC_FLAG_ADJ))
		return false;

	seq_state.mode = SEQ_BLOCK_KERNEL_DUMMY;
	E = flash(E, ev, &seq_state, &seq);
	if (FLASH_EVENTS != E)
		return false;

	e_adc = events_idx(0, SEQ_EVENT_ADC, E, ev);
	if (0 == (ev[e_adc].adc.flags & SEQ_ADC_FLAG_DUMMY))
		return false;

	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	E = flash(E, ev, &seq_state, &seq);

	if (FLASH_EVENTS != E)
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_events);


static bool test_flash_te(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	E = flash(E, ev, &seq_state, &seq);

	if (FLASH_EVENTS != E)
		return false;

	double te[seq.loop_dims[TE_DIM]];
	events_get_te(seq.loop_dims[TE_DIM], te, E, ev);
	if (0 != (seq.phys.te - te[0]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_te);


static bool test_flash_te_meco(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;
	seq.enc.pe_mode = SEQ_PEMODE_MEMS_HYB;
	seq.loop_dims[TE_DIM] = 5;
	seq.loop_dims[PHS1_DIM] = 7;

	seq.enc.tiny = 2;
	seq.loop_dims[TE_DIM] = 7;
	seq.phys.tr = 13.8E-3;
	seq.phys.te =  1.8E-3;
	seq.phys.te_delta =  1.8E-3;
	seq.geom.fov = 220E-3;
	seq.geom.baseres = 220;
	seq.phys.dwell = 5.4E-6;
	seq.phys.rf_duration = 400E-6;
	seq.phys.bwtp = 1.;
	seq.geom.slice_thickness = 5.E-3;

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	E = flash(E, ev, &seq_state, &seq);

	if (FLASH_EVENTS_MECO != E)
		return false;

	double te[seq.loop_dims[TE_DIM]];
	events_get_te(seq.loop_dims[TE_DIM], te, E, ev);

	for (int i = 0; i < seq.loop_dims[TE_DIM]; i++)
 		if (1E-5 * UT_TOL < fabs(seq.phys.te + i * seq.phys.te_delta  - te[i]))
			return false;

	return true;
}

UT_REGISTER_TEST(test_flash_te_meco);


static bool test_flash_mom1(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	E = flash(E, ev, &seq_state, &seq);

	if (FLASH_EVENTS != E)
		return false;

	int e_adc = events_idx(0, SEQ_EVENT_ADC, E, ev);
	int e_rf = events_idx(0, SEQ_EVENT_PULSE, E, ev);

	double mom_rf[3];
	moment_sum(mom_rf, ev[e_rf].mid, E, ev);

	double mom[3];
	moment_sum(mom, ev[e_adc].mid, E, ev);
	if (1E-5 * UT_TOL < (fabs(mom[0] - mom_rf[0]) + fabs(mom[1] - mom_rf[1]) + fabs(mom[2] - mom_rf[2])))
		return false;

	moment_sum(mom, ev[e_adc].start, E, ev);
	if (1E-5 * UT_TOL < fabs(mom[2] - mom_rf[2]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_mom1);


static bool test_flash_mom1b(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;
	seq.phys.te = 2E-3;
	seq.phys.dwell = 4.3E-6;

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	E = flash(E, ev, &seq_state, &seq);

	if (FLASH_EVENTS != E)
		return false;

	int e_adc = events_idx(0, SEQ_EVENT_ADC, E, ev);
	int e_rf = events_idx(0, SEQ_EVENT_PULSE, E, ev);

	double mom_rf[3];
	moment_sum(mom_rf, ev[e_rf].mid, E, ev);

	double mom[3];
	moment_sum(mom, ev[e_adc].mid, E, ev);
	if (1E-5 * UT_TOL < (fabs(mom[0] - mom_rf[0]) + fabs(mom[1] - mom_rf[1]) + fabs(mom[2] - mom_rf[2])))
		return false;

	moment_sum(mom, ev[e_adc].start, E, ev);
	if (1E-5 * UT_TOL < fabs(mom[2] - mom_rf[2]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_mom1b);


static bool test_flash_mom1c(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;
	seq.phys.te = 2E-3;
	seq.phys.dwell = 4.1E-6;

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	E = flash(E, ev, &seq_state, &seq);

	if (FLASH_EVENTS != E)
		return false;

	int e_adc = events_idx(0, SEQ_EVENT_ADC, E, ev);
	int e_rf = events_idx(0, SEQ_EVENT_PULSE, E, ev);

	double mom_rf[3];
	moment_sum(mom_rf, ev[e_rf].mid, E, ev);

	double mom[3];
	moment_sum(mom, ev[e_adc].mid, E, ev);
	if (1E-5 * UT_TOL < (fabs(mom[0] - mom_rf[0]) + fabs(mom[1] - mom_rf[1]) + fabs(mom[2] - mom_rf[2])))
		return false;

	moment_sum(mom, ev[e_adc].start, E, ev);
	if (1E-5 * UT_TOL < fabs(mom[2] - mom_rf[2]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_mom1c);


static bool test_flash_mom_meco(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;
	seq.enc.pe_mode = SEQ_PEMODE_MEMS_HYB;
	seq.loop_dims[TE_DIM] = 5;
	seq.loop_dims[PHS1_DIM] = 7;

	seq.enc.tiny = 2;
	seq.loop_dims[TE_DIM] = 7;
	seq.phys.tr = 13.8E-3;
	seq.phys.te =  1.8E-3;
	seq.phys.te_delta =  1.8E-3;
	seq.geom.fov = 220E-3;
	seq.geom.baseres = 220;
	seq.phys.dwell = 5.4E-6;
	seq.phys.rf_duration = 400E-6;
	seq.phys.bwtp = 1.;
	seq.geom.slice_thickness = 5.E-3;


	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	E = flash(E, ev, &seq_state, &seq);

	if (FLASH_EVENTS_MECO != E)
		return false;

	int e_rf = events_idx(0, SEQ_EVENT_PULSE, E, ev);

	double mom_rf[3];
	moment_sum(mom_rf, ev[e_rf].mid, E, ev);

	double mom[3];

	for (int i = 0; i < seq.loop_dims[TE_DIM]; i++) {

		int e_adc = events_idx(i, SEQ_EVENT_ADC, E, ev);

		moment_sum(mom, ev[e_adc].mid, E, ev);
		if (1E-5 * UT_TOL < (fabs(mom[0] - mom_rf[0]) + fabs(mom[1] - mom_rf[1]) + fabs(mom[2] - mom_rf[2])))
			return false;

		moment_sum(mom, ev[e_adc].start, E, ev);
		if (1E-5 * UT_TOL < fabs(mom[2] - mom_rf[2]))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_flash_mom_meco);


static bool test_flash_momentum_meco64(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;
	seq.enc.pe_mode = SEQ_PEMODE_MEMS_HYB;


	seq.loop_dims[TE_DIM] = 64;
	seq.phys.tr = 10;
	seq.phys.te =  3.E-3;
	seq.geom.slice_thickness = 5.E-3;

	long loops[DIMS] =  { [0 ... DIMS - 1] = 1 };
	loops[READ_DIM] = 10;
	loops[PHS1_DIM] = 10;
	loops[PHS2_DIM] = 10;
	
	long pos[DIMS] = { };

	do {

		seq.phys.dwell = 5.E-6 + pos[READ_DIM] * 1.E-7;
		seq.phys.rf_duration = (890 + 2 * pos[PHS1_DIM]) * 1E-6;
		seq.phys.te_delta =  (100. + 0.11 * pos[PHS2_DIM]) * 1E-3;

		int E = 2048;
		struct seq_event ev[E];

		seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
		E = flash(E, ev, &seq_state, &seq);

		if (FLASH_EVENTS_MECO_64 != E)
			return false;

		int e_rf = events_idx(0, SEQ_EVENT_PULSE, E, ev);

		double mom_rf[3];
		moment_sum(mom_rf, ev[e_rf].mid, E, ev);

		double mom[3];

		for (int i = 0; i < seq.loop_dims[TE_DIM]; i++) {

			int e_adc = events_idx(i, SEQ_EVENT_ADC, E, ev);

			moment_sum(mom, ev[e_adc].mid, E, ev);
			if (1E-5 * UT_TOL < (fabs(mom[0] - mom_rf[0]) + fabs(mom[1] - mom_rf[1]) + fabs(mom[2] - mom_rf[2])))
				return false;

			moment_sum(mom, ev[e_adc].start, E, ev);
			if (1E-5 * UT_TOL < fabs(mom[2] - mom_rf[2]))
				return false;
		}

	} while (md_next(DIMS, loops, 1 | 2 | 4, pos));

	return true;
}

UT_REGISTER_TEST(test_flash_momentum_meco64);


static bool test_flash_mom2(void)
{
	struct seq_state seq_state = { 0 };
	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	struct seq_config seq = seq_config_defaults;

	int E = 200;
	struct seq_event ev[E];

	E = flash(E, ev, &seq_state, &seq);

	if (FLASH_EVENTS != E)
		return false;

	const int samples = lround(1.E6 * seq.phys.tr);
	float m0[samples][3];

	seq_compute_moment0(samples, m0, 1.E-6, E, ev);

	long adc_mid = 1.E6 * ev[events_idx(0, SEQ_EVENT_ADC, E, ev)].mid;

	if (UT_TOL < fabs(m0[adc_mid][0] + m0[adc_mid - 1][0]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_mom2);


static bool test_flash_mom_spoiled(void)
{
	// we are always expecting same moment at end of imaging block
	const double expected_moments[3] = { 2.798404E-5, 0., 1.9401939E-5 };

	struct seq_state seq_state = { 0 };
	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	struct seq_config seq = seq_config_defaults;

	seq.phys.tr = 7E-3;
	seq.phys.contrast = SEQ_CONTRAST_RF_SPOILED;
	seq.loop_dims[PHS1_DIM] = 3;
	seq.loop_dims[TIME_DIM] = 3;

	seq_ui_interface_loop_dims(0, &seq, DIMS, seq.loop_dims);

	int E = 200;
	struct seq_event ev[E];

	for (int i = 0; i < seq.loop_dims[PHS1_DIM]; i++) {

		E = flash(E, ev, &seq_state, &seq);

		if (FLASH_EVENTS_SPOILED != E)
			return false;

		double mom_end[3];
		moment_sum(mom_end, seq.phys.tr, E, ev);

		for (int j = 0; j < 3; j++)
			if (1E-5 * UT_TOL < (fabs(mom_end[j] - expected_moments[j])))
				return false;
	}

	return true;
}

UT_REGISTER_TEST(test_flash_mom_spoiled);


static bool test_flash_freq1(void)
{
	struct seq_state seq_state = { 0 };
	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;

	struct seq_config seq = seq_config_defaults;
	seq.loop_dims[PHS1_DIM] = 3;
	seq.geom.shift[0][0] = 10.E-3;

	//  gamma * shift * ro_amplitude * sin(proj_angle)
	double modulation = seq.geom.shift[0][0] / (seq.geom.fov * seq.phys.dwell);
	const double expected_angle[3] = { 0. , 4. * M_PI / 3. , 2. * M_PI / 3. };

	int E = 200;
	struct seq_event ev[E];

	for (int i = 0; i < seq.loop_dims[PHS1_DIM]; i++) {

		seq_state.pos[PHS1_DIM] = i;
		E = flash(E, ev, &seq_state, &seq);

		struct seq_event ev_adc = ev[events_idx(0, SEQ_EVENT_ADC, E, ev)];

		if (3. * UT_TOL < fabs(ev_adc.adc.freq - modulation * sin(expected_angle[i])))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_flash_freq1);


static bool test_flash_freq2(void)
{
	struct seq_state seq_state = { 0 };
	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;

	struct seq_config seq = seq_config_defaults;
	seq.loop_dims[PHS1_DIM] = 3;
	seq.geom.shift[0][1] = 20.E-3;

	//  gamma * shift * ro_amplitude * cos(proj_angle)
	double modulation = seq.geom.shift[0][1] / (seq.geom.fov * seq.phys.dwell);
	const double expected_angle[3] = { 0. , 4. * M_PI / 3. , 2. * M_PI / 3. };

	int E = 200;
	struct seq_event ev[E];

	for (int i = 0; i < seq.loop_dims[PHS1_DIM]; i++) {

		seq_state.pos[PHS1_DIM] = i;
		E = flash(E, ev, &seq_state, &seq);

		struct seq_event ev_adc = ev[events_idx(0, SEQ_EVENT_ADC, E, ev)];

		if (3. * UT_TOL < fabs(ev_adc.adc.freq - modulation * cos(expected_angle[i])))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_flash_freq2);


static bool test_flash_freq3(void)
{
	struct seq_state seq_state = { 0 };
	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;

	struct seq_config seq = seq_config_defaults;
	seq.geom.shift[0][2] = 20.E-3;

	//  gamma * shift * slice_amplitude
	const double expected_freq = seq.geom.shift[0][2] * seq.phys.bwtp /
					(seq.phys.rf_duration * seq.geom.slice_thickness);

	int E = 200;
	struct seq_event ev[E];

	E = flash(E, ev, &seq_state, &seq);

	struct seq_event ev_rf = ev[events_idx(0, SEQ_EVENT_PULSE, E, ev)];

	if (UT_TOL < fabs(ev_rf.pulse.freq - expected_freq))
		return false;
		
	return true;
}

UT_REGISTER_TEST(test_flash_freq3);


static bool test_flash_phase(void)
{
	struct seq_state seq_state = { };
	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;

	struct seq_config seq = seq_config_defaults;

	seq.loop_dims[PHS1_DIM] = 999;

	seq_ui_interface_loop_dims(0, &seq, DIMS, seq.loop_dims);

	seq.geom.shift[0][0] = 10.E-3;
	seq.geom.shift[0][1] = 20.E-3;
	seq.geom.shift[0][2] = 30.E-3;

	int E = 200;
	struct seq_event ev[E];

	for (int i = 0; i < 999; i++) {

		seq_state.pos[PHS1_DIM] = i;
		E = flash(E, ev, &seq_state, &seq);

		struct seq_event ev_rf = ev[events_idx(0, SEQ_EVENT_PULSE, E, ev)];
		struct seq_event ev_adc = ev[events_idx(0, SEQ_EVENT_ADC, E, ev)];
		
		if (UT_TOL < fabs(ev_adc.adc.phase - ev_rf.pulse.phase))
			return false;
	}
		
	return true;
}

UT_REGISTER_TEST(test_flash_phase);


static bool test_raga_spokes(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;

	const int expected_spokes = 8;
	const int slices = 3;

	seq.loop_dims[PHS1_DIM] = 5;
	seq.loop_dims[SLICE_DIM] = slices;
	seq.loop_dims[TIME_DIM] = expected_spokes;

	seq_ui_interface_loop_dims(0, &seq, DIMS, seq.loop_dims);

	const int max_E = 200;
	struct seq_event ev[max_E];

	int ctr = 0;
	int E = 0;
	long last_raga_idx = -1;

	do {

		E = seq_block(max_E, ev,  &seq_state, &seq);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		last_raga_idx = ev[events_idx(0, SEQ_EVENT_ADC, E, ev)].adc.pos[PHS1_DIM];

		if (SEQ_BLOCK_KERNEL_IMAGE == seq_state.mode)
			ctr++;

	} while (seq_continue(&seq_state, &seq));

	if (ctr != slices * expected_spokes)
		return false;

	if (4 != last_raga_idx) // bart raga -m 3 5 indices; PHS2_DIM = 2, SLICE_DIM = 2 
		return false;

	return true;
}

UT_REGISTER_TEST(test_raga_spokes);

static bool test_raga_spokes_full(void)
{
	struct seq_state seq_state = { 0 };;
	struct seq_config seq = seq_config_defaults;

	const int spk = 377;
	seq.loop_dims[PHS1_DIM] = spk;
	seq.loop_dims[TIME_DIM] = spk;

	seq_ui_interface_loop_dims(0, &seq, DIMS, seq.loop_dims);

	const int max_E = 200;
	struct seq_event ev[max_E];

	int ctr = 0;
	do {

		int E = seq_block(max_E, ev,  &seq_state, &seq);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (SEQ_BLOCK_KERNEL_IMAGE == seq_state.mode)
			ctr++;

	} while (seq_continue(&seq_state, &seq));

	if (ctr != spk)
		return false;
	return true;
}

UT_REGISTER_TEST(test_raga_spokes_full);

