/* Copyright 2026. Pulserver contributors.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifdef _WIN32

#include <string.h>

#include "win/getsubopt.h"

int getsubopt(char** optionp, char* const* tokens, char** valuep)
{
	char* option = *optionp;

	if ('\0' == *option)
		return -1;

	char* end = strchr(option, ',');

	if (NULL == end)
		end = option + strlen(option);

	char* equals = memchr(option, '=', (size_t)(end - option));
	size_t len = (size_t)((NULL != equals ? equals : end) - option);

	*optionp = ('\0' != *end) ? end + 1 : end;

	if ('\0' != *end)
		*end = '\0';

	for (int i = 0; NULL != tokens[i]; i++) {

		if ((0 == strncmp(option, tokens[i], len)) && ('\0' == tokens[i][len])) {

			*valuep = (NULL != equals) ? equals + 1 : NULL;
			return i;
		}
	}

	*valuep = option;
	return -1;
}

#endif /* _WIN32 */
