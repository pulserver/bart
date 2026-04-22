/* Collect header files which should be C++-compatible */

/* Convert _Complex float* to float* for C++ compatibility.
 * In C++ code, casts to/from std::complex<float>* must be used. */
#define _Complex

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/stream.h"
#include "seq/config.h"
#include "seq/ui_enums.h"
