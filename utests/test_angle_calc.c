
#include <math.h>

#include "num/multind.h"

#include "misc/misc.h"

#include "seq/anglecalc.h"
#include "seq/config.h"

#include "utest.h"

static double angle_diff(double a, double b)
{
	return fabs(fmod(a - b, 2 * M_PI));
}


static bool test_get_rot_angle(void)
{
	double TOL = UT_TOL * 1e-5;

	struct seq_config seq = seq_config_defaults_flash;
	bart_dim_t pos[DIMS] = { 0 };
	md_copy_order(DIMS, seq.order, seq_loop_order_avg_outer);

	if (TOL < angle_diff(get_rot_angle(pos, &seq), 0.))
		return false;

	pos[PHS1_DIM] = 1;
	seq.loop_dims[PHS1_DIM] = 5;

	if (TOL < angle_diff(get_rot_angle(pos, &seq), 216. * (M_PI / 180.)))
		return false;

	seq.enc.pe_mode = SEQ_PEMODE_TURN;

	if (TOL < angle_diff(get_rot_angle(pos, &seq), (2./5.) * M_PI))
		return false;

	return true;
}


UT_REGISTER_TEST(test_get_rot_angle);


static bool test_cartesian_line(void)
{
	struct seq_config seq = seq_config_defaults_flash;
	bart_dim_t pos[DIMS] = { 0 };

	seq.enc.pe_mode = SEQ_PEMODE_CARTESIAN;

	seq.loop_dims[PHS1_DIM] = 8;
	const bart_dim_t good[8] = { 4, 3, 5, 2, 6, 1, 7, 0};

	int i = 0;
	do {
		if (good[i++] != cartesian_line(pos, &seq))
			return false;

	} while (md_next(DIMS, seq.loop_dims, PHS1_FLAG, pos));


	return true;
}

UT_REGISTER_TEST(test_cartesian_line);
