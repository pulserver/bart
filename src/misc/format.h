/* Copyright 2026. Pulserver contributors.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef _MISC_FORMAT_H
#define _MISC_FORMAT_H

#include <stdio.h>

/* The printf dialect BART's variadic functions are checked against.
 * MinGW's GCC reads the archetype printf as Microsoft's msvcrt dialect,
 * which has no z, j or ll; <stdio.h> names the dialect its own
 * functions implement. */
#ifdef __MINGW_PRINTF_FORMAT
#define BART_PRINTF __MINGW_PRINTF_FORMAT
#else
#define BART_PRINTF printf
#endif

#endif
