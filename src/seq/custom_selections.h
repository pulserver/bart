/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "seq/custom_ui.h"

extern const struct selection_opt seqtype_opts[2];
extern const struct selection_opt pemode_opts[5];
extern const struct selection_opt contrast_opts[5];

const char* get_seqtype_str(enum seq_type type);
const char* get_pemode_str(enum pe_mode mode);
const char* get_contrast_str(enum flash_contrast contrast);