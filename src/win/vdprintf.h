/* Copyright 2021. Tamás Hakkel
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Tamás Hakkel <hakkelt@gmail.com>
 */

#ifndef _VDPRINTF_WINDOWS
#define _VDPRINTF_WINDOWS

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <io.h>

int vdprintf(int, const char*, va_list);

// Formats into memory and writes the bytes to the descriptor, as POSIX
// vdprintf does.  A stdio stream opened over the descriptor for each call
// could not be closed without closing the descriptor, and the C runtime
// holds a fixed number of streams per process.
int vdprintf(int fd, const char *format, va_list ap)
{
	char small[256];
	char* buf = small;

	va_list aq;
	va_copy(aq, ap);
	int len = vsnprintf(small, sizeof small, format, aq);
	va_end(aq);

	if (len < 0)
		return -1;

	if ((size_t)len >= sizeof small) {

		if (NULL == (buf = malloc((size_t)len + 1)))
			return -1;

		vsnprintf(buf, (size_t)len + 1, format, ap);
	}

	int done = 0;

	while (done < len) {

		int n = _write(fd, buf + done, (unsigned int)(len - done));

		if (n <= 0) {

			done = -1;
			break;
		}

		done += n;
	}

	if (small != buf)
		free(buf);

	return done;
}

#endif /* _VDPRINTF_WINDOWS */
