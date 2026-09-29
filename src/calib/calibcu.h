/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "misc/dimtypes.h"
#include "misc/cppwrap.h"

extern void eigenmapscu(const bart_dim_t dims[5], _Complex float* optr, _Complex float* eptr, const _Complex float* imgcov2, int num_orthiter);

#include "misc/cppwrap.h"

