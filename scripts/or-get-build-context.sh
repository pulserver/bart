#!/bin/bash

mkdir -p cuda-11-6-2; cd cuda-11-6-2;

wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda-repo-debian11-11-6-local_11.6.2-510.47.03-1_amd64.deb

wget http://ftp.at.debian.org/debian/pool/main/libu/liburcu/liburcu6_0.12.2-1_amd64.deb

cat - > cuda-11-patch-all <<EOF
diff -u b/math_functions.h  00:03:30.815134398 +0300 b/math_functions.h
--- b/math_functions.h  00:03:30.815134398 +0300
+++ b/math_functions.h	2026-06-19 15:04:33.328666789 +0000
@@ -2547,7 +2547,7 @@
  *
  * \note_accuracy_double
  */
-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 sinpi(double x) noexcept (true);
 /**
  * \ingroup CUDA_MATH_SINGLE
  * \brief Calculate the sine of the input argument
@@ -2570,7 +2570,7 @@
  *
  * \note_accuracy_single
  */
-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  sinpif(float x) noexcept (true);
 /**
  * \ingroup CUDA_MATH_DOUBLE
  * \brief Calculate the cosine of the input argument
@@ -2592,7 +2592,7 @@
  *
  * \note_accuracy_double
  */
-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double                 cospi(double x) noexcept (true);
 /**
  * \ingroup CUDA_MATH_SINGLE
  * \brief Calculate the cosine of the input argument
@@ -2614,7 +2614,7 @@
  *
  * \note_accuracy_single
  */
-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float                  cospif(float x) noexcept (true);
 /**
  * \ingroup CUDA_MATH_DOUBLE
  * \brief  Calculate the sine and cosine of the first input argument
@@ -11583,9 +11583,9 @@

 __MATH_FUNCTIONS_DECL__ float rcbrt(const float a);

-__MATH_FUNCTIONS_DECL__ float sinpi(const float a);
+__MATH_FUNCTIONS_DECL__ float sinpi(const float a) noexcept(true);

-__MATH_FUNCTIONS_DECL__ float cospi(const float a);
+__MATH_FUNCTIONS_DECL__ float cospi(const float a) noexcept(true);

 __MATH_FUNCTIONS_DECL__ void sincospi(const float a, float *const sptr, float *const cptr);

unchanged:
--- a/math_functions.hpp	2026-06-19 14:57:45.269438591 +0000
+++ b/math_functions.hpp	2026-06-19 15:05:09.388245113 +0000
@@ -770,12 +770,12 @@
   return rcbrtf(a);
 }

-__MATH_FUNCTIONS_DECL__ float sinpi(const float a)
+__MATH_FUNCTIONS_DECL__ float sinpi(const float a) noexcept(true)
 {
   return sinpif(a);
 }

-__MATH_FUNCTIONS_DECL__ float cospi(const float a)
+__MATH_FUNCTIONS_DECL__ float cospi(const float a) noexcept(true)
 {
   return cospif(a);
 }
@@ -2970,7 +2970,7 @@
   return t;
 }

-__func__(double sinpi(double a))
+__func__(double sinpi(double a)) noexcept(true)
 {
   int n;

@@ -3000,7 +3000,7 @@
   return a;
 }

-__func__(double cospi(double a))
+__func__(double cospi(double a)) noexcept(true)
 {
   int n;

@@ -3331,12 +3331,12 @@
   return static_cast<float>(rcbrt(static_cast<double>(a)));
 }

-__func__(float sinpif(const float a))
+__func__(float sinpif(const float a)) noexcept(true)
 {
   return static_cast<float>(sinpi(static_cast<double>(a)));
 }

-__func__(float cospif(const float a))
+__func__(float cospif(const float a)) noexcept(true)
 {
   return static_cast<float>(cospi(static_cast<double>(a)));
 }
EOF
