#include <math.h>
#include <stdio.h>

#include "num/multind.h"

#include "seq/adc_rf.h"
#include "seq/config.h"
#include "seq/event.h"
#include "seq/kernel.h"
#include "seq/seq.h"
#include "seq/helpers.h"
#include "seq/misc.h"
#include "seq/opts.h"

#include "seq/miniflash.h"

#include "utest.h"

#define MINIFLASH_EVENTS 18
#define MINIFLASH_EVENTS_CENTER 14

static bool test_miniflash_events(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults_miniflash;
	seq_copy_order(&seq);

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	E = miniflash(E, ev, &seq_state, &seq);

	if (MINIFLASH_EVENTS != E)
		return false;

	return true;
}

UT_REGISTER_TEST(test_miniflash_events);


static bool test_miniflash_te(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults_miniflash;
	seq_copy_order(&seq);
	seq_copy_order(&seq);

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	E = miniflash(E, ev, &seq_state, &seq);

	if (MINIFLASH_EVENTS != E)
		return false;

	double te[seq.loop_dims[TE_DIM]];
	events_get_te(seq.loop_dims[TE_DIM], te, E, ev);
	if (0 != (seq.phys.te - te[0]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_miniflash_te);


static bool test_miniflash_mom1(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults_miniflash;
	seq_copy_order(&seq);

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	seq_state.pos[PHS1_DIM] = 128;
	E = miniflash(E, ev, &seq_state, &seq);

	if (MINIFLASH_EVENTS_CENTER != E)
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

UT_REGISTER_TEST(test_miniflash_mom1);


static bool test_miniflash_mom1b(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults_miniflash;
	seq_copy_order(&seq);
	seq.phys.dwell = 4.3E-6;

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	E = miniflash(E, ev, &seq_state, &seq);

	if (MINIFLASH_EVENTS != E)
		return false;

	int e_adc = events_idx(0, SEQ_EVENT_ADC, E, ev);
	int e_rf = events_idx(0, SEQ_EVENT_PULSE, E, ev);

	double mom_rf[3];
	moment_sum(mom_rf, ev[e_rf].mid, E, ev);

	double mom[3];
	moment_sum(mom, ev[e_adc].mid, E, ev);

	// don't check y-moment, because we are not in k-space center for line 0
	if (1E-5 * UT_TOL < (fabs(mom[0] - mom_rf[0]) /*+ fabs(mom[1] - mom_rf[1])*/ + fabs(mom[2] - mom_rf[2])))
		return false;

	moment_sum(mom, ev[e_adc].start, E, ev);
	if (1E-5 * UT_TOL < fabs(mom[2] - mom_rf[2]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_miniflash_mom1b);


static bool test_miniflash_mom1c(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults_miniflash;
	seq_copy_order(&seq);
	seq.phys.dwell = 4.1E-6;

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	seq_state.pos[PHS1_DIM] = 128;
	E = miniflash(E, ev, &seq_state, &seq);

	if (MINIFLASH_EVENTS_CENTER != E)
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

UT_REGISTER_TEST(test_miniflash_mom1c);


static bool test_miniflash_mom2(void)
{
	struct seq_state seq_state = { 0 };
	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	struct seq_config seq = seq_config_defaults_miniflash;
	seq_copy_order(&seq);

	int E = 200;
	struct seq_event ev[E];

	E = miniflash(E, ev, &seq_state, &seq);

	if (MINIFLASH_EVENTS != E)
		return false;

	const int samples = lround(1.E6 * seq.phys.tr);
	float m0[samples][3];

	seq_compute_moment0(samples, m0, 1.E-6, E, ev);

	bart_dim_t adc_mid = 1.E6 * ev[events_idx(0, SEQ_EVENT_ADC, E, ev)].mid;
	if (UT_TOL < fabs(m0[adc_mid][0] + m0[adc_mid - 1][0]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_miniflash_mom2);


static bool test_miniflash_mom_spoiled(void)
{
	struct seq_state seq_state = { 0 };
	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;
	struct seq_config seq = seq_config_defaults_miniflash;
	seq_copy_order(&seq);

	// we are always expecting same moment at end of imaging block
	const double expected_moments[3] = { 4.1873147E-5 /* = 1.5 * ro_momentum */, 0., slice_momentum_to_rephase(&seq) };

	int E = 200;
	struct seq_event ev[E];

	for (int i = 0; i < seq.loop_dims[PHS1_DIM]; i++) {

		seq_state.pos[PHS1_DIM] = i;
		E = miniflash(E, ev, &seq_state, &seq);

		if ((MINIFLASH_EVENTS != E) && (seq.loop_dims[PHS1_DIM] / 2 == i && (MINIFLASH_EVENTS_CENTER != E)))
			return false;

		int e_rf = events_idx(0, SEQ_EVENT_PULSE, E, ev);
		double mom_rf[3];
		moment_sum(mom_rf, ev[e_rf].mid, E, ev);

		double mom_end[3];
		moment_sum(mom_end, seq.phys.tr, E, ev);

		if (1E-5 * UT_TOL < (fabs(mom_end[0] - expected_moments[0])))
			return false;

		if (1E-5 * UT_TOL < (fabs(mom_end[1] - expected_moments[1])))
			return false;

		if (1E-5 * UT_TOL < (fabs(mom_end[2] - mom_rf[2] - expected_moments[2])))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_miniflash_mom_spoiled);


static bool test_miniflash_shift_ro(void)
{
	struct seq_state seq_state = { 0 };
	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;

	struct seq_config seq = seq_config_defaults_miniflash;
	seq_copy_order(&seq);
	seq.geom.shift[0][0] = 10.E-3;

	double modulation = seq.geom.shift[0][0] / (seq.geom.fov * seq.phys.dwell);

	int E = 200;
	struct seq_event ev[E];

	for (int i = 0; i < seq.loop_dims[PHS1_DIM]; i++) {

		seq_state.pos[PHS1_DIM] = i;
		E = miniflash(E, ev, &seq_state, &seq);

		struct seq_event ev_adc = ev[events_idx(0, SEQ_EVENT_ADC, E, ev)];

		if (UT_TOL < fabs(ev_adc.adc.freq - modulation))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_miniflash_shift_ro);


static bool test_miniflash_shift_pe(void)
{
	struct seq_state seq_state = { 0 };
	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;

	struct seq_config seq = seq_config_defaults_miniflash;
	seq_copy_order(&seq);
	seq.phys.contrast = SEQ_CONTRAST_NO_SPOILING;
	seq.geom.shift[0][1] = 20.E-3;

	double phase_shift = 360. * seq.geom.shift[0][1] / seq.geom.fov;

	int E = 200;
	struct seq_event ev[E];

	for (int i = 0; i < seq.loop_dims[PHS1_DIM]; i++) {

		seq_state.pos[PHS1_DIM] = i;
		E = miniflash(E, ev, &seq_state, &seq);

		struct seq_event ev_adc = ev[events_idx(0, SEQ_EVENT_ADC, E, ev)];

		if (1E-5 * UT_TOL < fabs(ev_adc.adc.freq))
			return false;

		if (UT_TOL < fabs(ev_adc.adc.phase - phase_clamp((i - seq.loop_dims[PHS1_DIM] / 2) * phase_shift)))
			return false;
	}

	return true;
}

UT_REGISTER_TEST(test_miniflash_shift_pe);


static bool test_miniflash_shift_sl(void)
{
	struct seq_state seq_state = { 0 };
	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;

	struct seq_config seq = seq_config_defaults_miniflash;
	seq_copy_order(&seq);
	seq.geom.shift[0][2] = 20.E-3;

	//  gamma * shift * slice_amplitude
	const double expected_freq = seq.geom.shift[0][2] * seq.phys.bwtp /
					(seq.phys.rf_duration * seq.geom.slice_thickness);

	int E = 200;
	struct seq_event ev[E];

	E = miniflash(E, ev, &seq_state, &seq);

	struct seq_event ev_rf = ev[events_idx(0, SEQ_EVENT_PULSE, E, ev)];

	if (UT_TOL < fabs(ev_rf.pulse.freq - expected_freq))
		return false;
		
	return true;
}

UT_REGISTER_TEST(test_miniflash_shift_sl);


static bool test_miniflash_phase(void)
{
	struct seq_state seq_state = { };
	seq_state.mode = SEQ_BLOCK_KERNEL_IMAGE;

	struct seq_config seq = seq_config_defaults_miniflash;
	seq_copy_order(&seq);

	seq.geom.shift[0][0] = 10.E-3;
	seq.geom.shift[0][1] = 20E-3;
	seq.geom.shift[0][2] = 30.E-3;

	int E = 200;
	struct seq_event ev[E];

	double phase_shift = 360. * seq.geom.shift[0][1] / seq.geom.fov;

	for (int i = 0; i < seq.loop_dims[PHS1_DIM]; i++) {

		seq_state.pos[PHS1_DIM] = i;
		E = miniflash(E, ev, &seq_state, &seq);

		struct seq_event ev_rf = ev[events_idx(0, SEQ_EVENT_PULSE, E, ev)];
		struct seq_event ev_adc = ev[events_idx(0, SEQ_EVENT_ADC, E, ev)];

		if (1E-5 * UT_TOL < fabs(phase_clamp(ev_adc.adc.phase - ((i - seq.loop_dims[PHS1_DIM] / 2) * phase_shift) - ev_rf.pulse.phase)))
			return false;
	}
		
	return true;
}

UT_REGISTER_TEST(test_miniflash_phase);
