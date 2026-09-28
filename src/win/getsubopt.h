/* Copyright 2026. Pulserver contributors.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef _GETSUBOPT_WINDOWS
#define _GETSUBOPT_WINDOWS

// POSIX getsubopt, which the Windows C runtime does not provide.
extern int getsubopt(char** optionp, char* const* tokens, char** valuep);

#endif /* _GETSUBOPT_WINDOWS */
