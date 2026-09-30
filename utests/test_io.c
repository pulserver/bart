/* Copyright 2026. Pulserver contributors.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Headers written through the descriptor-level printf, many of them in one
 * process, as a program that runs BART's commands in-process writes them.
 */

#include <stdbool.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include "misc/misc.h"
#include "misc/io.h"

#ifdef _WIN32
#include "win/open_patch.h"
#endif

#include "utest.h"


static bool test_cfl_header_length_is_its_bytes(void)
{
	const char* name = "test_io_header.hdr";
	bart_dim_t dims[16] = { 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5 };

	int fd = open(name, O_RDWR|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR);

	UT_RETURN_ON_FAILURE(-1 != fd);

	int written = write_cfl_header(fd, NULL, 16, dims);

	off_t size = lseek(fd, 0, SEEK_END);

	close(fd);
	unlink(name);

	UT_RETURN_ASSERT((0 < written) && (size == written));
}

UT_REGISTER_TEST(test_cfl_header_length_is_its_bytes);


static bool test_many_cfl_headers_in_one_process(void)
{
#ifdef _WIN32
	const char* null_file = "nul";
#else
	const char* null_file = "/dev/null";
#endif
	bart_dim_t dims[16] = { 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5 };

	int fd = open(null_file, O_WRONLY);

	UT_RETURN_ON_FAILURE(-1 != fd);

	// Each header is a few dozen formatted writes, so this is more writes
	// than a C runtime holds streams.
	bool ok = true;

	for (int i = 0; ok && (i < 200); i++)
		ok = (0 < write_cfl_header(fd, NULL, 16, dims));

	close(fd);

	UT_RETURN_ASSERT(ok);
}

UT_REGISTER_TEST(test_many_cfl_headers_in_one_process);
