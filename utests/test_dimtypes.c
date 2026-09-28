/* Copyright 2026. Pulserver contributors.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Dimensions, strides and flag sets that do not fit in 32 bits.
 *
 * Everything here holds on a platform whose 'long' is 32 bits wide
 * (64-bit Windows) as much as on one whose 'long' is 64 bits, which is
 * what the fixed-width types are for.  The arrays are described but
 * never allocated: the arithmetic is what is tested.
 */

#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <unistd.h>

#include "num/multind.h"

#include "misc/misc.h"
#include "misc/io.h"

#include "utest.h"


_Static_assert(8 == sizeof(bart_dim_t), "dimensions are 64 bits");
_Static_assert(8 == sizeof(bart_stride_t), "strides are 64 bits");
_Static_assert(8 == sizeof(bart_flags_t), "flag sets are 64 bits");
_Static_assert((bart_flags_t)-1 > 0, "flag sets are unsigned");
_Static_assert((bart_stride_t)-1 < 0, "strides are signed");

// The element type of a temporary dimension array does not follow the
// width of whatever literal is written first.
_Static_assert(_Generic(MD_DIMS(1)[0], bart_dim_t: 1, default: 0), "MD_DIMS makes dimensions");
_Static_assert(8 == sizeof(MD_DIMS(1)[0]), "MD_DIMS makes 64-bit dimensions");
_Static_assert(_Generic(MD_BIT(0), bart_flags_t: 1, default: 0), "MD_BIT makes a flag set");

// A shift past bit 31 of a 32-bit type would be undefined.
_Static_assert(0 != MD_BIT(63), "MD_BIT reaches bit 63");
_Static_assert(UINT64_C(0x8000000000000000) == MD_BIT(63), "MD_BIT(63) is the top bit");


static bool test_stride_above_int32(void)
{
	// 65536 x 65536 x 2 complex floats: the third stride is 2^35 bytes.
	bart_dim_t dims[3] = { 1 << 16, 1 << 16, 2 };
	bart_stride_t strs[3];

	md_calc_strides(3, strs, dims, sizeof(complex float));

	UT_RETURN_ON_FAILURE(8 == strs[0]);
	UT_RETURN_ON_FAILURE(INT64_C(8) << 16 == strs[1]);
	UT_RETURN_ON_FAILURE(INT64_C(8) << 32 == strs[2]);
	UT_RETURN_ON_FAILURE(strs[2] > INT32_MAX);

	bart_dim_t pos[3] = { 3, 5, 1 };
	bart_stride_t off = md_calc_offset(3, strs, pos);

	UT_RETURN_ON_FAILURE(8 * 3 + (INT64_C(8) << 16) * 5 + (INT64_C(8) << 32) == off);

	// A negative stride of the same magnitude, as a flip makes.
	bart_stride_t nstrs[3] = { strs[0], strs[1], -strs[2] };

	UT_RETURN_ASSERT(-(INT64_C(8) << 32) + 8 * 3 + (INT64_C(8) << 16) * 5 == md_calc_offset(3, nstrs, pos));
}

UT_REGISTER_TEST(test_stride_above_int32);


static bool test_size_above_int32(void)
{
	// 3 * 2^32 elements.
	bart_dim_t dims[3] = { 1 << 20, 1 << 12, 3 };

	bart_dim_t size = md_calc_size(3, dims);

	UT_RETURN_ON_FAILURE(INT64_C(3) << 32 == size);
	UT_RETURN_ON_FAILURE(size > UINT32_MAX);

	// A flat index past 2^32 and back.
	bart_dim_t index = (INT64_C(2) << 32) + 12345;
	bart_dim_t pos[3];

	md_unravel_index(3, pos, ~UINT64_C(0), dims, index);

	UT_RETURN_ON_FAILURE(12345 == pos[0]);
	UT_RETURN_ON_FAILURE(0 == pos[1]);
	UT_RETURN_ON_FAILURE(2 == pos[2]);

	UT_RETURN_ASSERT(index == md_ravel_index(3, pos, ~UINT64_C(0), dims));
}

UT_REGISTER_TEST(test_size_above_int32);


static bool test_temporary_dims(void)
{
	// Each element of a temporary array is read at the width
	// md_calc_size reads it with.
	UT_RETURN_ON_FAILURE((INT64_C(3) << 33) == md_calc_size(2, MD_DIMS(INT64_C(1) << 33, 3)));
	UT_RETURN_ON_FAILURE((INT64_C(3) << 33) == md_calc_size(2, MD_DIMS(3, INT64_C(1) << 33)));
	UT_RETURN_ON_FAILURE(6 == md_calc_size(3, MD_DIMS(1, 2, 3)));

	// MAKE_ARRAY takes the type of its first element.
	UT_RETURN_ASSERT(6 == md_calc_size(3, MAKE_ARRAY((bart_dim_t)1, 2, 3)));
}

UT_REGISTER_TEST(test_temporary_dims);


static bool test_high_flags(void)
{
	enum { D = 64 };

	UT_RETURN_ON_FAILURE(MD_IS_SET(MD_BIT(40), 40));
	UT_RETURN_ON_FAILURE(!MD_IS_SET(MD_BIT(40), 8));	// 40 mod 32
	UT_RETURN_ON_FAILURE(0 == MD_CLEAR(MD_BIT(63), 63));
	UT_RETURN_ON_FAILURE(MD_BIT(63) == MD_SET(0, 63));

	UT_RETURN_ON_FAILURE(40 == md_min_idx(MD_BIT(40)));
	UT_RETURN_ON_FAILURE(63 == md_min_idx(MD_BIT(63)));
	UT_RETURN_ON_FAILURE(63 == md_max_idx(MD_BIT(63) | MD_BIT(2)));
	UT_RETURN_ON_FAILURE(-1 == md_min_idx(0));
	UT_RETURN_ON_FAILURE(64 == bitcount(~UINT64_C(0)));
	UT_RETURN_ON_FAILURE(2 == bitcount(MD_BIT(33) | MD_BIT(62)));

	bart_dim_t dims[D];
	md_singleton_dims(D, dims);
	dims[40] = 5;
	dims[63] = 7;

	UT_RETURN_ON_FAILURE((MD_BIT(40) | MD_BIT(63)) == md_nontriv_dims(D, dims));

	bart_dim_t odims[D];
	md_select_dims(D, MD_BIT(63), odims, dims);

	UT_RETURN_ON_FAILURE(1 == odims[40]);
	UT_RETURN_ON_FAILURE(7 == odims[63]);

	md_select_dims(D, ~MD_BIT(40), odims, dims);

	UT_RETURN_ON_FAILURE(1 == odims[40]);
	UT_RETURN_ON_FAILURE(7 == odims[63]);

	bart_stride_t strs[D];
	md_calc_strides(D, strs, dims, 1);

	UT_RETURN_ASSERT((MD_BIT(40) | MD_BIT(63)) == md_nontriv_strides(D, strs));
}

UT_REGISTER_TEST(test_high_flags);


static bool test_parse_above_int32(void)
{
	bart_dim_t val = 0;

	UT_RETURN_ON_FAILURE(0 == parse_long(&val, "8589934592"));
	UT_RETURN_ON_FAILURE(INT64_C(1) << 33 == val);

	UT_RETURN_ON_FAILURE(0 == parse_long(&val, "-8589934593"));
	UT_RETURN_ON_FAILURE(-(INT64_C(1) << 33) - 1 == val);

	char* str = ptr_print_dims(2, MD_DIMS(INT64_C(1) << 33, 5));
	bool ok = (NULL != strstr(str, "8589934592"));
	xfree(str);

	UT_RETURN_ASSERT(ok);
}

UT_REGISTER_TEST(test_parse_above_int32);


static bool test_cfl_header_above_int32(void)
{
	const bart_dim_t dims[4] = { INT64_C(3) << 32, 1, INT64_C(1) << 40, 2 };

	FILE* fp = tmpfile();

	UT_RETURN_ON_FAILURE(NULL != fp);

	int fd = fileno(fp);

	UT_RETURN_ON_FAILURE(0 < write_cfl_header(fd, NULL, 4, dims));
	UT_RETURN_ON_FAILURE(0 == lseek(fd, 0, SEEK_SET));

	char* file = NULL;
	char* cmd = NULL;
	bart_dim_t rdims[4];

	int r = read_cfl_header(fd, "test", &file, &cmd, 4, rdims);

	fclose(fp);
	xfree(file);
	xfree(cmd);

	UT_RETURN_ON_FAILURE(0 < r);

	UT_RETURN_ASSERT(md_check_equal_dims(4, dims, rdims, ~UINT64_C(0)));
}

UT_REGISTER_TEST(test_cfl_header_above_int32);
