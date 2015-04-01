# Copyright (C) 2014 Josh Willis
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import multibench as _mb
from pycbc.types import zeros, complex64, complex128, float32, float64, Array
from pycbc import scheme as _scheme
from scipy.weave import inline
import numpy as _np
import ctypes
from pycbc import libutils

"""
This module contains various long-strings of code intended to be used by other
modules when they implement the correlation step of matched filtering. Most
of these will refer to various constants that mus be string-replaced before
the code can be compiled by weave.

All correlations are the multiplication, element-by-element, of two complex
vectors, using SIMD instructions where available. There is *NO* conjugation
of either vector performed; in applications, it is assumed that it is
simpler to conjugate one of the inputs (typically, the overwhitened FFT of
the data segment) and this shortens and simplifies somewhat the SIMD instructions.
At present, these codes also assume that all inputs are SIMD aligned and that
the lengths of various arrays are multiples of the SIMD length, though in
actual applications that will be relaxed.

All array arguments are single-precision real arrays, assuming that the complex
arrays are in interleaved storage formats.  Constants that are substituted before
compilation should accordingly be for the relevant lengths as real arrays.

The basic approaches implemented here are:

 (1) Read and write contiguous arrays (no striding, and no assumption of a
     "pre-transpose".  There are three variants:
     (a) Read from and write to memory with standard store instructions, and
         no attempt to bypass the cache.
     (b) Reads of inputs bypass the cache, but writes do not.
     (c) Read while bypassing the cache, and write two copies of the output,
         one of which bypasses the cache and the other of which does not.

 (2) Read "pre-transposed" arrays which have an input length and stride, and
     write to an output array (with a possibly different stride). Again,
     three variants:
     (a) Normal read and write, no cache bypass.
     (b) Read of input bypasses the cache, write does not.
     (c) Read while bypassing cache, and write two copies, one of which
         bypasses the cache and the other of which does not.

Though the code is designed to compile on machines lacking either of SSE3 or
AVX, the cache bypasses will not work without these instruction sets, and so
the various sub-options above may degenerate into one another.

The basic functions are vectorized by not parallelized; they are intended to
be called from other functions using OpenMP to parallelize.
"""

### The following is used by all of our codes, so we just define it once here

corr_common_support = """
#include <x86intrin.h>

#ifdef __AVX2__
#define _HAVE_AVX2 1
#else
#define _HAVE_AVX2 0
#endif

#ifdef __AVX__
#define _HAVE_AVX 1
#else
#define _HAVE_AVX 0
#endif

#ifdef __SSE4_1__
#define _HAVE_SSE4_1 1
#else
#define _HAVE_SSE4_1 0
#endif

#ifdef __SSE3__
#define _HAVE_SSE3 1
#else
#define _HAVE_SSE3 0
#endif

// The following ONLY works if s is a power of two
#define ROUND_DOWN(x, s) ((x) & ~((s)-1))

"""

### Basic correlations, no transpose

# In the following, string-substitute 'NLEN' with the length
# (as an array of floats) to be correlated.

corr_contig_nostream_support = corr_common_support + """
static inline void ccorrf_contig_nostream(float * __restrict in1,
                                            float * __restrict in2,
                                            float * __restrict out){

  int i;
  float *aptr, *bptr, *cptr;

  aptr = in1;
  bptr = in2;
  cptr = out;

#if _HAVE_AVX
  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;

  _mm256_zeroupper();

  // Main loop using AVX; unrolled once.
  for (i = 0; i < NLEN; i += 16){
      // Load everything into registers

      ymm0 = _mm256_load_ps(aptr);
      ymm3 = _mm256_load_ps(aptr+8);
      ymm1 = _mm256_load_ps(bptr);
      ymm4 = _mm256_load_ps(bptr+8);

      ymm2 = _mm256_movehdup_ps(ymm1);
      ymm1 = _mm256_moveldup_ps(ymm1);
      ymm1 = _mm256_mul_ps(ymm1, ymm0);
      ymm0 = _mm256_shuffle_ps(ymm0, ymm0, 0xB1);
      ymm2 = _mm256_mul_ps(ymm2, ymm0);
      ymm0 = _mm256_addsub_ps(ymm1, ymm2);

      ymm5 = _mm256_movehdup_ps(ymm4);
      ymm4 = _mm256_moveldup_ps(ymm4);
      ymm4 = _mm256_mul_ps(ymm4, ymm3);
      ymm3 = _mm256_shuffle_ps(ymm3, ymm3, 0xB1);
      ymm5 = _mm256_mul_ps(ymm5, ymm3);
      ymm3 = _mm256_addsub_ps(ymm4, ymm5);

      _mm256_store_ps(cptr, ymm0);
      _mm256_store_ps(cptr+8, ymm3);

      aptr += 16;
      bptr += 16;
      cptr += 16;
  }

  _mm256_zeroupper();

#elif _HAVE_SSE3

  __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

  // Main loop using SSE; unrolled once.
  for (i = 0; i < NLEN; i += 8){
      // Load everything into registers

      xmm0 = _mm_load_ps(aptr);
      xmm3 = _mm_load_ps(aptr+4);
      xmm1 = _mm_load_ps(bptr);
      xmm4 = _mm_load_ps(bptr+4);

      xmm2 = _mm_movehdup_ps(xmm1);
      xmm1 = _mm_moveldup_ps(xmm1);
      xmm1 = _mm_mul_ps(xmm1, xmm0);
      xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0xB1);
      xmm2 = _mm_mul_ps(xmm2, xmm0);
      xmm0 = _mm_addsub_ps(xmm1, xmm2);

      xmm5 = _mm_movehdup_ps(xmm4);
      xmm4 = _mm_moveldup_ps(xmm4);
      xmm4 = _mm_mul_ps(xmm4, xmm3);
      xmm3 = _mm_shuffle_ps(xmm3, xmm3, 0xB1);
      xmm5 = _mm_mul_ps(xmm5, xmm3);
      xmm3 = _mm_addsub_ps(xmm4, xmm5);

      _mm_store_ps(cptr, xmm0);
      _mm_store_ps(cptr+4, xmm3);

      aptr += 8;
      bptr += 8;
      cptr += 8;
  }

#else
// We don't have AVX or SSE3, so fall back to plain old C
  float re0, re1, im0, im1, ar0, ar1, ai0, ai1, br0, br1, bi0, bi1;

  for (i = 0; i < NLEN; i += 4){
    ar0 = *aptr;
    ai0 = *(aptr+1);
    ar1 = *(aptr+2);
    ai1 = *(aptr+3);
    br0 = *bptr;
    bi0 = *(bptr+1);
    br1 = *(bptr+2);
    bi1 = *(bptr+3);
 
    re0 = ar0*br0 - ai0*bi0;
    im0 = ar0*bi0 + ai0*br0;
    re1 = ar1*br1 - ai1*bi1;
    im1 = ar1*bi1 + ai1*br1;

    *cptr = re0;
    *(cptr+1) = im0;
    *(cptr+2) = re1;
    *(cptr+3) = im1;

    aptr += 4;
    bptr += 4;
    cptr += 4;
  }
#endif

}
"""

corr_contig_streamin_support = corr_common_support + """
static inline void ccorrf_contig_streamin(float * __restrict in1,
                                            float * __restrict in2,
                                            float * __restrict out){

  int i;
  float *aptr, *bptr, *cptr;

  aptr = in1;
  bptr = in2;
  cptr = out;

#if _HAVE_AVX
  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;
#ifndef __AVX2__
  // Without AVX2, we'll have to load a bunch of stuff using SSE4.1
  __m128 xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13;
#endif

  _mm256_zeroupper();

  // Main loop using AVX; unrolled once.
  for (i = 0; i < NLEN; i += 16){
      // Load everything into registers
#if _HAVE_AVX2
      // Non-temporal loads only become available in AVX2
      ymm0 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) aptr ));
      ymm3 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) (aptr+8) ));
      ymm1 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) bptr ));
      ymm4 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) (bptr+8) ));
#else
      // So we instead replicate using SSE4.1 non-temporal loads
      _mm256_zeroupper();
      xmm6 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) aptr ));
      xmm7 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+4) ));
      xmm8 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+8) ));
      xmm9 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+12) ));
      xmm10 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) bptr ));
      xmm11 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+4) ));
      xmm12 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+8) ));
      xmm13 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+12) ));
      _mm256_zeroupper();
      // Now blend these together
      ymm0 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm6), xmm7, 0x1);
      ymm3 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm8), xmm9, 0x1);
      ymm1 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm10), xmm11, 0x1);
      ymm4 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm12), xmm13, 0x1);
#endif

      ymm2 = _mm256_movehdup_ps(ymm1);
      ymm1 = _mm256_moveldup_ps(ymm1);
      ymm1 = _mm256_mul_ps(ymm1, ymm0);
      ymm0 = _mm256_shuffle_ps(ymm0, ymm0, 0xB1);
      ymm2 = _mm256_mul_ps(ymm2, ymm0);
      ymm0 = _mm256_addsub_ps(ymm1, ymm2);

      ymm5 = _mm256_movehdup_ps(ymm4);
      ymm4 = _mm256_moveldup_ps(ymm4);
      ymm4 = _mm256_mul_ps(ymm4, ymm3);
      ymm3 = _mm256_shuffle_ps(ymm3, ymm3, 0xB1);
      ymm5 = _mm256_mul_ps(ymm5, ymm3);
      ymm3 = _mm256_addsub_ps(ymm4, ymm5);

      _mm256_store_ps(cptr, ymm0);
      _mm256_store_ps(cptr+8, ymm3);

      aptr += 16;
      bptr += 16;
      cptr += 16;
  }

  _mm256_zeroupper();

#elif _HAVE_SSE3

  __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

  // Main loop using SSE; unrolled once.
  for (i = 0; i < NLEN; i += 8){
      // Load everything into registers
#ifdef __SSE4_1__
      xmm0 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) aptr ));
      xmm3 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+4) ));
      xmm1 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) bptr ));
      xmm4 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+4) ));
#else
      // If we don't have at least SS4.1, then this will NOT bypass the cache,
      // and will fall back to the same behavior as the non-streamed function
      xmm0 = _mm_load_ps(aptr);
      xmm3 = _mm_load_ps(aptr+4);
      xmm1 = _mm_load_ps(bptr);
      xmm4 = _mm_load_ps(bptr+4);
#endif
      xmm2 = _mm_movehdup_ps(xmm1);
      xmm1 = _mm_moveldup_ps(xmm1);
      xmm1 = _mm_mul_ps(xmm1, xmm0);
      xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0xB1);
      xmm2 = _mm_mul_ps(xmm2, xmm0);
      xmm0 = _mm_addsub_ps(xmm1, xmm2);

      xmm5 = _mm_movehdup_ps(xmm4);
      xmm4 = _mm_moveldup_ps(xmm4);
      xmm4 = _mm_mul_ps(xmm4, xmm3);
      xmm3 = _mm_shuffle_ps(xmm3, xmm3, 0xB1);
      xmm5 = _mm_mul_ps(xmm5, xmm3);
      xmm3 = _mm_addsub_ps(xmm4, xmm5);

      _mm_store_ps(cptr, xmm0);
      _mm_store_ps(cptr+4, xmm3);

      aptr += 8;
      bptr += 8;
      cptr += 8;
  }

#else
// We don't have AVX or SSE3, so fall back to plain old C
  float re0, re1, im0, im1, ar0, ar1, ai0, ai1, br0, br1, bi0, bi1;

  for (i = 0; i < NLEN; i += 4){
    ar0 = *aptr;
    ai0 = *(aptr+1);
    ar1 = *(aptr+2);
    ai1 = *(aptr+3);
    br0 = *bptr;
    bi0 = *(bptr+1);
    br1 = *(bptr+2);
    bi1 = *(bptr+3);
 
    re0 = ar0*br0 - ai0*bi0;
    im0 = ar0*bi0 + ai0*br0;
    re1 = ar1*br1 - ai1*bi1;
    im1 = ar1*bi1 + ai1*br1;

    *cptr = re0;
    *(cptr+1) = im0;
    *(cptr+2) = re1;
    *(cptr+3) = im1;

    aptr += 4;
    bptr += 4;
    cptr += 4;
  }
#endif

}
"""

corr_contig_streaminout_support = corr_common_support + """
static inline void ccorrf_contig_streaminout(float * __restrict in1,
                                               float * __restrict in2,
                                               float * __restrict outstream,
                                               float * __restrict outstore){
  int i;
  float *aptr, *bptr, *cptr, *dptr;

  aptr = in1;
  bptr = in2;
  cptr = outstream;
  dptr = outstore;

#if _HAVE_AVX
  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;
#ifndef __AVX2__
  // Without AVX2, we'll have to load a bunch of stuff using SSE4.1
  __m128 xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13;
#endif

  _mm256_zeroupper();

  // Main loop using AVX; unrolled once.
  for (i = 0; i < NLEN; i += 16){
      // Load everything into registers
#if _HAVE_AVX2
      // Non-temporal loads only become available in AVX2
      ymm0 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) aptr ));
      ymm3 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) (aptr+8) ));
      ymm1 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) bptr ));
      ymm4 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) (bptr+8) ));
#else
      // So we instead replicate using SSE4.1 non-temporal loads
      _mm256_zeroupper();
      xmm6 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) aptr ));
      xmm7 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+4) ));
      xmm8 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+8) ));
      xmm9 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+12) ));
      xmm10 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) bptr ));
      xmm11 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+4) ));
      xmm12 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+8) ));
      xmm13 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+12) ));
      _mm256_zeroupper();
      // Now blend these together
      ymm0 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm6), xmm7, 0x1);
      ymm3 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm8), xmm9, 0x1);
      ymm1 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm10), xmm11, 0x1);
      ymm4 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm12), xmm13, 0x1);
#endif

      ymm2 = _mm256_movehdup_ps(ymm1);
      ymm1 = _mm256_moveldup_ps(ymm1);
      ymm1 = _mm256_mul_ps(ymm1, ymm0);
      ymm0 = _mm256_shuffle_ps(ymm0, ymm0, 0xB1);
      ymm2 = _mm256_mul_ps(ymm2, ymm0);
      ymm0 = _mm256_addsub_ps(ymm1, ymm2);

      ymm5 = _mm256_movehdup_ps(ymm4);
      ymm4 = _mm256_moveldup_ps(ymm4);
      ymm4 = _mm256_mul_ps(ymm4, ymm3);
      ymm3 = _mm256_shuffle_ps(ymm3, ymm3, 0xB1);
      ymm5 = _mm256_mul_ps(ymm5, ymm3);
      ymm3 = _mm256_addsub_ps(ymm4, ymm5);

      _mm256_stream_ps(cptr, ymm0);
      _mm256_stream_ps(cptr+8, ymm3);
      _mm256_store_ps(dptr, ymm0);
      _mm256_store_ps(dptr+8, ymm3);

      aptr += 16;
      bptr += 16;
      cptr += 16;
      dptr += 16;
  }

  _mm256_zeroupper();

#elif _HAVE_SSE3

  __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

  // Main loop using SSE; unrolled once.
  for (i = 0; i < NLEN; i += 8){
      // Load everything into registers
#ifdef __SSE4_1__
      xmm0 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) aptr ));
      xmm3 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+4) ));
      xmm1 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) bptr ));
      xmm4 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+4) ));
#else
      // If we don't have at least SS4.1, then this will NOT bypass the cache,
      // and will fall back to the same behavior as the non-streamed function
      xmm0 = _mm_load_ps(aptr);
      xmm3 = _mm_load_ps(aptr+4);
      xmm1 = _mm_load_ps(bptr);
      xmm4 = _mm_load_ps(bptr+4);
#endif
      xmm2 = _mm_movehdup_ps(xmm1);
      xmm1 = _mm_moveldup_ps(xmm1);
      xmm1 = _mm_mul_ps(xmm1, xmm0);
      xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0xB1);
      xmm2 = _mm_mul_ps(xmm2, xmm0);
      xmm0 = _mm_addsub_ps(xmm1, xmm2);

      xmm5 = _mm_movehdup_ps(xmm4);
      xmm4 = _mm_moveldup_ps(xmm4);
      xmm4 = _mm_mul_ps(xmm4, xmm3);
      xmm3 = _mm_shuffle_ps(xmm3, xmm3, 0xB1);
      xmm5 = _mm_mul_ps(xmm5, xmm3);
      xmm3 = _mm_addsub_ps(xmm4, xmm5);

      _mm_stream_ps(cptr, xmm0);
      _mm_stream_ps(cptr+4, xmm3);
      _mm_store_ps(dptr, xmm0);
      _mm_store_ps(dptr+4, xmm3);

      aptr += 8;
      bptr += 8;
      cptr += 8;
      dptr += 8;
  }

#else
// We don't have AVX or SSE3, so fall back to plain old C
  float re0, re1, im0, im1, ar0, ar1, ai0, ai1, br0, br1, bi0, bi1;

  for (i = 0; i < NLEN; i += 4){
    ar0 = *aptr;
    ai0 = *(aptr+1);
    ar1 = *(aptr+2);
    ai1 = *(aptr+3);
    br0 = *bptr;
    bi0 = *(bptr+1);
    br1 = *(bptr+2);
    bi1 = *(bptr+3);
 
    re0 = ar0*br0 - ai0*bi0;
    im0 = ar0*bi0 + ai0*br0;
    re1 = ar1*br1 - ai1*bi1;
    im1 = ar1*bi1 + ai1*br1;

    *cptr = re0;
    *(cptr+1) = im0;
    *(cptr+2) = re1;
    *(cptr+3) = im1;
    *dptr = re0;
    *(dptr+1) = im0;
    *(dptr+2) = re1;
    *(dptr+3) = im1;


    aptr += 4;
    bptr += 4;
    cptr += 4;
    dptr += 4;
  }
#endif

}
"""

### "Pre-transposed" or strided correlations

# The following constants must be string-substituted
# before this code can be compiled by weave:
#     NITER:  How many steps of the in and out strides to take
#     NLEN: Length of input to read from each stride
#     ISTRIDE: Length of stride of input
#     OSTRIDE: Length of stride of (non-streamed) output

corr_strided_nostream_support = corr_common_support + """
static inline void ccorrf_strided_nostream(float * __restrict in1,
                                           float * __restrict in2,
                                           float * __restrict out){

  int i, j;
  float *aptr, *bptr, *cptr;

#if _HAVE_AVX
  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;

  _mm256_zeroupper();

  for (j = 0; j < NITER; j++){
     aptr = in1 + j * ISTRIDE;
     bptr = in2 + j * ISTRIDE;
     cptr = out + j * OSTRIDE;

     // Main loop using AVX; unrolled once.
     for (i = 0; i < NLEN; i += 16){
         // Load everything into registers

         ymm0 = _mm256_load_ps(aptr);
         ymm3 = _mm256_load_ps(aptr+8);
         ymm1 = _mm256_load_ps(bptr);
         ymm4 = _mm256_load_ps(bptr+8);

         ymm2 = _mm256_movehdup_ps(ymm1);
         ymm1 = _mm256_moveldup_ps(ymm1);
         ymm1 = _mm256_mul_ps(ymm1, ymm0);
         ymm0 = _mm256_shuffle_ps(ymm0, ymm0, 0xB1);
         ymm2 = _mm256_mul_ps(ymm2, ymm0);
         ymm0 = _mm256_addsub_ps(ymm1, ymm2);

         ymm5 = _mm256_movehdup_ps(ymm4);
         ymm4 = _mm256_moveldup_ps(ymm4);
         ymm4 = _mm256_mul_ps(ymm4, ymm3);
         ymm3 = _mm256_shuffle_ps(ymm3, ymm3, 0xB1);
         ymm5 = _mm256_mul_ps(ymm5, ymm3);
         ymm3 = _mm256_addsub_ps(ymm4, ymm5);

         _mm256_store_ps(cptr, ymm0);
         _mm256_store_ps(cptr+8, ymm3);

         aptr += 16;
         bptr += 16;
         cptr += 16;
     }
  }
  _mm256_zeroupper();

#elif _HAVE_SSE3

  __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

  for (j = 0; j < NITER; j++){
     aptr = in1 + j * ISTRIDE;
     bptr = in2 + j * ISTRIDE;
     cptr = out + j * OSTRIDE;

     // Main loop using SSE; unrolled once.
     for (i = 0; i < NLEN; i += 8){
         // Load everything into registers

         xmm0 = _mm_load_ps(aptr);
         xmm3 = _mm_load_ps(aptr+4);
         xmm1 = _mm_load_ps(bptr);
         xmm4 = _mm_load_ps(bptr+4);

         xmm2 = _mm_movehdup_ps(xmm1);
         xmm1 = _mm_moveldup_ps(xmm1);
         xmm1 = _mm_mul_ps(xmm1, xmm0);
         xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0xB1);
         xmm2 = _mm_mul_ps(xmm2, xmm0);
         xmm0 = _mm_addsub_ps(xmm1, xmm2);

         xmm5 = _mm_movehdup_ps(xmm4);
         xmm4 = _mm_moveldup_ps(xmm4);
         xmm4 = _mm_mul_ps(xmm4, xmm3);
         xmm3 = _mm_shuffle_ps(xmm3, xmm3, 0xB1);
         xmm5 = _mm_mul_ps(xmm5, xmm3);
         xmm3 = _mm_addsub_ps(xmm4, xmm5);

         _mm_store_ps(cptr, xmm0);
         _mm_store_ps(cptr+4, xmm3);

         aptr += 8;
         bptr += 8;
         cptr += 8;
     }
  }
#else
// We don't have AVX or SSE3, so fall back to plain old C
  float re0, re1, im0, im1, ar0, ar1, ai0, ai1, br0, br1, bi0, bi1;

  for (j = 0; j < NITER; j++){
     aptr = in1 + j * ISTRIDE;
     bptr = in2 + j * ISTRIDE;
     cptr = out + j * OSTRIDE;

     for (i = 0; i < NLEN; i += 4){
       ar0 = *aptr;
       ai0 = *(aptr+1);
       ar1 = *(aptr+2);
       ai1 = *(aptr+3);
       br0 = *bptr;
       bi0 = *(bptr+1);
       br1 = *(bptr+2);
       bi1 = *(bptr+3);
 
       re0 = ar0*br0 - ai0*bi0;
       im0 = ar0*bi0 + ai0*br0;
       re1 = ar1*br1 - ai1*bi1;
       im1 = ar1*bi1 + ai1*br1;

       *cptr = re0;
       *(cptr+1) = im0;
       *(cptr+2) = re1;
       *(cptr+3) = im1;

       aptr += 4;
       bptr += 4;
       cptr += 4;
     }
   }
#endif

}
"""

corr_strided_streamin_support = corr_common_support + """
static inline void ccorrf_strided_streamin(float * __restrict in1,
                                           float * __restrict in2,
                                           float * __restrict out){

  int i, j;
  float *aptr, *bptr, *cptr;

#if _HAVE_AVX
  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;
#ifndef __AVX2__
  // Without AVX2, we'll have to load a bunch of stuff using SSE4.1
  __m128 xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13;
#endif

  _mm256_zeroupper();

  for (j = 0; j < NITER; j++){
     aptr = in1 + j * ISTRIDE;
     bptr = in2 + j * ISTRIDE;
     cptr = out + j * OSTRIDE;

     // Main loop using AVX; unrolled once.
     for (i = 0; i < NLEN; i += 16){
         // Load everything into registers
#if _HAVE_AVX2
         // Non-temporal loads only become available in AVX2
         ymm0 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) aptr ));
         ymm3 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) (aptr+8) ));
         ymm1 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) bptr ));
         ymm4 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) (bptr+8) ));
#else
         // So we instead replicate using SSE4.1 non-temporal loads
         _mm256_zeroupper();
         xmm6 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) aptr ));
         xmm7 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+4) ));
         xmm8 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+8) ));
         xmm9 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+12) ));
         xmm10 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) bptr ));
         xmm11 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+4) ));
         xmm12 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+4) ));
         xmm13 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+4) ));
         _mm256_zeroupper();
         // Now blend these together
         ymm0 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm6), xmm7, 0x1);
         ymm3 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm8), xmm9, 0x1);
         ymm1 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm10), xmm11, 0x1);
         ymm4 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm12), xmm13, 0x1);
#endif
         ymm0 = _mm256_load_ps(aptr);
         ymm3 = _mm256_load_ps(aptr+8);
         ymm1 = _mm256_load_ps(bptr);
         ymm4 = _mm256_load_ps(bptr+8);

         ymm2 = _mm256_movehdup_ps(ymm1);
         ymm1 = _mm256_moveldup_ps(ymm1);
         ymm1 = _mm256_mul_ps(ymm1, ymm0);
         ymm0 = _mm256_shuffle_ps(ymm0, ymm0, 0xB1);
         ymm2 = _mm256_mul_ps(ymm2, ymm0);
         ymm0 = _mm256_addsub_ps(ymm1, ymm2);

         ymm5 = _mm256_movehdup_ps(ymm4);
         ymm4 = _mm256_moveldup_ps(ymm4);
         ymm4 = _mm256_mul_ps(ymm4, ymm3);
         ymm3 = _mm256_shuffle_ps(ymm3, ymm3, 0xB1);
         ymm5 = _mm256_mul_ps(ymm5, ymm3);
         ymm3 = _mm256_addsub_ps(ymm4, ymm5);

         _mm256_store_ps(cptr, ymm0);
         _mm256_store_ps(cptr+8, ymm3);

         aptr += 16;
         bptr += 16;
         cptr += 16;
     }
  }
  _mm256_zeroupper();

#elif _HAVE_SSE3

  __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

  for (j = 0; j < NITER; j++){
     aptr = in1 + j * ISTRIDE;
     bptr = in2 + j * ISTRIDE;
     cptr = out + j * OSTRIDE;

     // Main loop using SSE; unrolled once.
     for (i = 0; i < NLEN; i += 8){
         // Load everything into registers
#ifdef __SSE4_1__
         xmm0 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) aptr ));
         xmm3 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+4) ));
         xmm1 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) bptr ));
         xmm4 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+4) ));
#else
         // If we don't have at least SS4.1, then this will NOT bypass the cache,
         // and will fall back to the same behavior as the non-streamed function
         xmm0 = _mm_load_ps(aptr);
         xmm3 = _mm_load_ps(aptr+4);
         xmm1 = _mm_load_ps(bptr);
         xmm4 = _mm_load_ps(bptr+4);
#endif

         xmm2 = _mm_movehdup_ps(xmm1);
         xmm1 = _mm_moveldup_ps(xmm1);
         xmm1 = _mm_mul_ps(xmm1, xmm0);
         xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0xB1);
         xmm2 = _mm_mul_ps(xmm2, xmm0);
         xmm0 = _mm_addsub_ps(xmm1, xmm2);

         xmm5 = _mm_movehdup_ps(xmm4);
         xmm4 = _mm_moveldup_ps(xmm4);
         xmm4 = _mm_mul_ps(xmm4, xmm3);
         xmm3 = _mm_shuffle_ps(xmm3, xmm3, 0xB1);
         xmm5 = _mm_mul_ps(xmm5, xmm3);
         xmm3 = _mm_addsub_ps(xmm4, xmm5);

         _mm_store_ps(cptr, xmm0);
         _mm_store_ps(cptr+4, xmm3);

         aptr += 8;
         bptr += 8;
         cptr += 8;
     }
  }
#else
// We don't have AVX or SSE3, so fall back to plain old C
  float re0, re1, im0, im1, ar0, ar1, ai0, ai1, br0, br1, bi0, bi1;

  for (j = 0; j < NITER; j++){
     aptr = in1 + j * ISTRIDE;
     bptr = in2 + j * ISTRIDE;
     cptr = out + j * OSTRIDE;

     for (i = 0; i < NLEN; i += 4){
       ar0 = *aptr;
       ai0 = *(aptr+1);
       ar1 = *(aptr+2);
       ai1 = *(aptr+3);
       br0 = *bptr;
       bi0 = *(bptr+1);
       br1 = *(bptr+2);
       bi1 = *(bptr+3);
 
       re0 = ar0*br0 - ai0*bi0;
       im0 = ar0*bi0 + ai0*br0;
       re1 = ar1*br1 - ai1*bi1;
       im1 = ar1*bi1 + ai1*br1;

       *cptr = re0;
       *(cptr+1) = im0;
       *(cptr+2) = re1;
       *(cptr+3) = im1;

       aptr += 4;
       bptr += 4;
       cptr += 4;
     }
   }
#endif

}
"""

# The following function does *NOT* use OSTRIDE, but rather
# two different constants, OSTRIDE_STREAM and OSTRIDE_STORE,
# for the output strides of the third and fourth arguments,
# repsectively.

corr_strided_streaminout_support = corr_common_support + """
static inline void ccorrf_strided_streaminout(float * __restrict in1,
                                             float * __restrict in2,
                                             float * __restrict outstream,
                                             float * __restrict outstore){

  int i, j;
  float *aptr, *bptr, *cptr, *dptr;

#if _HAVE_AVX
  __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;
#ifndef __AVX2__
  // Without AVX2, we'll have to load a bunch of stuff using SSE4.1
  __m128 xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13;
#endif

  _mm256_zeroupper();

  for (j = 0; j < NITER; j++){
     aptr = in1 + j * ISTRIDE;
     bptr = in2 + j * ISTRIDE;
     cptr = outstream + j * OSTRIDE_STREAM;
     dptr = outstore + j * OSTRIDE_STORE;

     // Main loop using AVX; unrolled once.
     for (i = 0; i < NLEN; i += 16){
         // Load everything into registers
#if _HAVE_AVX2
         // Non-temporal loads only become available in AVX2
         ymm0 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) aptr ));
         ymm3 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) (aptr+8) ));
         ymm1 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) bptr ));
         ymm4 = _mm256_castsi256_ps( _mm256_stream_load_si256( (__m256i *) (bptr+8) ));
#else
         // So we instead replicate using SSE4.1 non-temporal loads
         _mm256_zeroupper();
         xmm6 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) aptr ));
         xmm7 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+4) ));
         xmm8 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+8) ));
         xmm9 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+12) ));
         xmm10 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) bptr ));
         xmm11 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+4) ));
         xmm12 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+4) ));
         xmm13 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+4) ));
         _mm256_zeroupper();
         // Now blend these together
         ymm0 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm6), xmm7, 0x1);
         ymm3 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm8), xmm9, 0x1);
         ymm1 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm10), xmm11, 0x1);
         ymm4 = _mm256_insertf128_ps( _mm256_castps128_ps256(xmm12), xmm13, 0x1);
#endif
         ymm0 = _mm256_load_ps(aptr);
         ymm3 = _mm256_load_ps(aptr+8);
         ymm1 = _mm256_load_ps(bptr);
         ymm4 = _mm256_load_ps(bptr+8);

         ymm2 = _mm256_movehdup_ps(ymm1);
         ymm1 = _mm256_moveldup_ps(ymm1);
         ymm1 = _mm256_mul_ps(ymm1, ymm0);
         ymm0 = _mm256_shuffle_ps(ymm0, ymm0, 0xB1);
         ymm2 = _mm256_mul_ps(ymm2, ymm0);
         ymm0 = _mm256_addsub_ps(ymm1, ymm2);

         ymm5 = _mm256_movehdup_ps(ymm4);
         ymm4 = _mm256_moveldup_ps(ymm4);
         ymm4 = _mm256_mul_ps(ymm4, ymm3);
         ymm3 = _mm256_shuffle_ps(ymm3, ymm3, 0xB1);
         ymm5 = _mm256_mul_ps(ymm5, ymm3);
         ymm3 = _mm256_addsub_ps(ymm4, ymm5);

         _mm256_stream_ps(cptr, ymm0);
         _mm256_stream_ps(cptr+8, ymm3);
         _mm256_store_ps(dptr, ymm0);
         _mm256_store_ps(dptr+8, ymm3);

         aptr += 16;
         bptr += 16;
         cptr += 16;
         dptr += 16;
     }
  }
  _mm256_zeroupper();

#elif _HAVE_SSE3

  __m128 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5;

  for (j = 0; j < NITER; j++){
     aptr = in1 + j * ISTRIDE;
     bptr = in2 + j * ISTRIDE;
     cptr = outstream + j * OSTRIDE_STREAM;
     dptr = outstore + j * OSTRIDE_STORE;

     // Main loop using SSE; unrolled once.
     for (i = 0; i < NLEN; i += 8){
         // Load everything into registers
#ifdef __SSE4_1__
         xmm0 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) aptr ));
         xmm3 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (aptr+4) ));
         xmm1 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) bptr ));
         xmm4 = _mm_castsi128_ps( _mm_stream_load_si128( (__m128i *) (bptr+4) ));
#else
         // If we don't have at least SS4.1, then this will NOT bypass the cache,
         // and will fall back to the same behavior as the non-streamed function
         xmm0 = _mm_load_ps(aptr);
         xmm3 = _mm_load_ps(aptr+4);
         xmm1 = _mm_load_ps(bptr);
         xmm4 = _mm_load_ps(bptr+4);
#endif

         xmm2 = _mm_movehdup_ps(xmm1);
         xmm1 = _mm_moveldup_ps(xmm1);
         xmm1 = _mm_mul_ps(xmm1, xmm0);
         xmm0 = _mm_shuffle_ps(xmm0, xmm0, 0xB1);
         xmm2 = _mm_mul_ps(xmm2, xmm0);
         xmm0 = _mm_addsub_ps(xmm1, xmm2);

         xmm5 = _mm_movehdup_ps(xmm4);
         xmm4 = _mm_moveldup_ps(xmm4);
         xmm4 = _mm_mul_ps(xmm4, xmm3);
         xmm3 = _mm_shuffle_ps(xmm3, xmm3, 0xB1);
         xmm5 = _mm_mul_ps(xmm5, xmm3);
         xmm3 = _mm_addsub_ps(xmm4, xmm5);

         _mm_stream_ps(cptr, xmm0);
         _mm_stream_ps(cptr+4, xmm3);
         _mm_store_ps(dptr, xmm0);
         _mm_store_ps(dptr+4, xmm3);

         aptr += 8;
         bptr += 8;
         cptr += 8;
         dptr += 8;
     }
  }
#else
// We don't have AVX or SSE3, so fall back to plain old C
  float re0, re1, im0, im1, ar0, ar1, ai0, ai1, br0, br1, bi0, bi1;

  for (j = 0; j < NITER; j++){
     aptr = in1 + j * ISTRIDE;
     bptr = in2 + j * ISTRIDE;
     cptr = outstream + j * OSTRIDE_STREAM;
     dptr = outstore + j * OSTRIDE_STORE;

     for (i = 0; i < NLEN; i += 4){
       ar0 = *aptr;
       ai0 = *(aptr+1);
       ar1 = *(aptr+2);
       ai1 = *(aptr+3);
       br0 = *bptr;
       bi0 = *(bptr+1);
       br1 = *(bptr+2);
       bi1 = *(bptr+3);
 
       re0 = ar0*br0 - ai0*bi0;
       im0 = ar0*bi0 + ai0*br0;
       re1 = ar1*br1 - ai1*bi1;
       im1 = ar1*bi1 + ai1*br1;

       *cptr = re0;
       *(cptr+1) = im0;
       *(cptr+2) = re1;
       *(cptr+3) = im1;

       *dptr = re0;
       *(dptr+1) = im0;
       *(dptr+2) = re1;
       *(dptr+3) = im1;

       aptr += 4;
       bptr += 4;
       cptr += 4;
       dptr += 4;
     }
   }
#endif

}
"""

### Now some actual code that just implements the different
### correlations in a parallelized fashion.

corr_contig_nostream_code = """
int k;

#pragma omp parallel for schedule(static, 1)
for (k = 0; k < NBLOCKS; k++){
  ccorrf_contig_nostream( &in1[k*SIZE_BLOCK], &in2[k*SIZE_BLOCK], &out[k*SIZE_BLOCK]);
}
"""

corr_contig_streamin_code = """
int k;

#pragma omp parallel for schedule(static, 1)
for (k = 0; k < NBLOCKS; k++){
  ccorrf_contig_streamin( &in1[k*SIZE_BLOCK], &in2[k*SIZE_BLOCK], &out[k*SIZE_BLOCK]);
}
"""

corr_contig_streaminout_code = """
int k;

#pragma omp parallel for schedule(static, 1)
for (k = 0; k < NBLOCKS; k++){
  ccorrf_contig_streaminout( &in1[k*SIZE_BLOCK], &in2[k*SIZE_BLOCK],
                               &outstream[k*SIZE_BLOCK], &outstore[k*SIZE_BLOCK]);
}
"""

corr_strided_nostream_code = """
int k;

#pragma omp parallel for schedule(static, 1)
for (k = 0; k < NBLOCKS; k++){
  ccorrf_strided_nostream( &in1[k*ISIZE], &in2[k*ISIZE], &out[k*OSIZE]);
}
"""

corr_strided_streamin_code = """
int k;

#pragma omp parallel for schedule(static, 1)
for (k = 0; k < NBLOCKS; k++){
  ccorrf_strided_streamin( &in1[k*ISIZE], &in2[k*ISIZE], &out[k*OSIZE]);
}
"""

corr_strided_streaminout_code = """
int k;

#pragma omp parallel for schedule(static, 1)
for (k = 0; k < NBLOCKS; k++){
  ccorrf_strided_streaminout( &in1[k*ISIZE], &in2[k*ISIZE],
                              &outstream[k*OSIZE_STREAM], &outstore[k*OSIZE_STORE]);
}
"""

omp_support = """
#include <omp.h>
"""

omp_libs = ['gomp']
omp_flags = ['-fopenmp']

# With 8192 *complex* elements, three copies will fit in L2 cache
max_chunk = 8192

def check_pow_two(n):
    return ( (n != 0) and ( (n & (n-1)) == 0) )

class BaseCorrProblem(_mb.MultiBenchProblem):
    def __init__(self, size, pad = 0):
        # We'll do some arithmetic with these, so sanity check first:
        if (size < 1):
            raise ValueError("size must be >= 1")
        if not check_pow_two(size):
            raise ValueError("Only power-of-two sizes supported")
        self.nsqrt = 2 ** int(_np.log2( size ) / 2)
        ntmp = size/self.nsqrt

        if self.nsqrt != ntmp:
            raise ValueError("Only supporting perfect square size at the moment")

        self.size = size
        self.pad = pad
        self.psize = (ntmp+pad)*ntmp
        self.nbatch = max_chunk/self.nsqrt

        self.i1 = zeros(self.size, dtype=complex64)
        self.i2 = zeros(self.size, dtype=complex64)
        # Note: the second output is the same size as
        # the *input*, because it is the streaming
        # destination for correlations that both
        # stream and store.
        self.o1 = zeros(self.psize, dtype=complex64)
        self.o2 = zeros(self.size, dtype=complex64)

class CorrNumpy(object):
    def __init__(self, in1, in2, out, ncorr):
        if (in1.dtype != _np.complex64 or in2.dtype != _np.complex64 or out.dtype != _np.complex64):
            raise ValueError("Only complex64 arrays permitted")
        if (len(in1) < ncorr ) or (len(in2) < ncorr ) or (len(out) < ncorr):
            raise ValueError("Arrays must have length at least equal to 'ncorr'")
        self.in1 = in1
        self.in2 = in2
        self.out = out
        self.ncorr = ncorr

    def execute(self):
        self.out[:self.ncorr] = self.in1[:self.ncorr]
        self.out[:self.ncorr] *= self.in2[:self.ncorr]

class CorrContiguousBase(object):
    """
    Create an object that holds the two input vectors in1 and in2,
    and the output vector out, and which when its 'execute()' method
    is called, will write into 'out' the element-by-element complex
    multiplication of the vectors 'in1' and 'in2', from the first
    element up to the 'ncorr'-th element.
    """
    def __init__(self, in1, in2, out, ncorr, verbose):
        if (in1.dtype != _np.complex64 or in2.dtype != _np.complex64 or out.dtype != _np.complex64):
            raise ValueError("Only complex64 arrays permitted")
        if (len(in1) < ncorr ) or (len(in2) < ncorr ) or (len(out) < ncorr):
            raise ValueError("Arrays must have length at least equal to 'ncorr'")
        # At the moment, we only process blocks whose length is divisble
        # by 'max_chunk'
        if ( (ncorr % max_chunk) != 0):
            raise ValueError("Correlation length must be divisble by 'max_chunk' = {0}".format(max_chunk))
        self.verbose = verbose
        self.in1 = in1
        self.in2 = in2
        self.out = out
        self.ncorr = ncorr
        # Note, all of the following assumes that the __init__ of the subclass is
        # called *before* this init
        #
        # First, support code. The factor of two is because our arrays will be seen
        # by the C-correlation function as real arrays rather than complex.
        self.support = self.support.replace('NLEN', str(max_chunk * 2))
        # Next, the function code. 
        self.nblocks = self.ncorr/max_chunk
        self.code = self.code.replace('NBLOCKS', str(self.nblocks))
        self.code = self.code.replace('SIZE_BLOCK', str(max_chunk * 2))

class CorrContiguousNoStreaming(CorrContiguousBase):
    def __init__(self, in1, in2, out, ncorr, verbose = 0):
        self.support = omp_support + corr_contig_nostream_support
        self.code = corr_contig_nostream_code
        super(CorrContiguousNoStreaming, self).__init__(in1, in2, out, ncorr, verbose = verbose)

    def execute(self):
        in1 = _np.array(self.in1, copy = False).view(dtype = float32)
        in2 = _np.array(self.in2, copy = False).view(dtype = float32)
        out = _np.array(self.out, copy = False).view(dtype = float32)
        inline(self.code, ['in1', 'in2', 'out'],
               extra_compile_args = ['-march=native -O3 -w'] + omp_flags,
               support_code = self.support, auto_downcast = 1, verbose = self.verbose,
               libraries = omp_libs)

class CorrContiguousStreamIn(CorrContiguousBase):
    def __init__(self, in1, in2, out, ncorr, verbose = 0):
        self.support = omp_support + corr_contig_streamin_support
        self.code = corr_contig_streamin_code
        super(CorrContiguousStreamIn, self).__init__(in1, in2, out, ncorr, verbose = verbose)

    def execute(self):
        in1 = _np.array(self.in1, copy = False).view(dtype = float32)
        in2 = _np.array(self.in2, copy = False).view(dtype = float32)
        out = _np.array(self.out, copy = False).view(dtype = float32)
        inline(self.code, ['in1', 'in2', 'out'],
               extra_compile_args = ['-march=native -O3 -w'] + omp_flags,
               support_code = self.support, auto_downcast = 1, verbose = self.verbose,
               libraries = omp_libs)

class CorrContiguousStreamInOut(CorrContiguousBase):
    def __init__(self, in1, in2, outstream, outstore, ncorr, verbose = 0):
        self.support = omp_support + corr_contig_streaminout_support
        self.code = corr_contig_streaminout_code
        super(CorrContiguousStreamInOut, self).__init__(in1, in2, outstream, ncorr, verbose = verbose)
        self.outstore = outstore

    def execute(self):
        in1 = _np.array(self.in1, copy = False).view(dtype = float32)
        in2 = _np.array(self.in2, copy = False).view(dtype = float32)
        outstream = _np.array(self.out, copy = False).view(dtype = float32)
        outstore = _np.array(self.outstore, copy = False).view(dtype = float32)
        inline(self.code, ['in1', 'in2', 'outstream', 'outstore'],
               extra_compile_args = ['-march=native -O3 -w'] + omp_flags,
               support_code = self.support, auto_downcast = 1, verbose = self.verbose,
               libraries = omp_libs)

class ContigNostream(BaseCorrProblem):
    def __init__(self, size):
        super(ContigNostream, self).__init__(size = size)
        self.instance = CorrContiguousNoStreaming(self.i1.data, self.i2.data, self.o2.data, size/2, verbose = 2)
        self.execute = self.instance.execute
        # Force compilation as the setup step
        self._setup = self.instance.execute


class ContigStreamIn(BaseCorrProblem):
    def __init__(self, size):
        super(ContigStreamIn, self).__init__(size = size)
        self.instance = CorrContiguousStreamIn(self.i1.data, self.i2.data, self.o2.data, size/2, verbose = 2)
        self.execute = self.instance.execute
        # Force compilation as the setup step
        self._setup = self.instance.execute


class ContigStreamInOut(BaseCorrProblem):
    def __init__(self, size):
        super(ContigStreamInOut, self).__init__(size = size)
        self.instance = CorrContiguousStreamInOut(self.i1.data, self.i2.data, self.o2.data,
                                                  self.o1.data, size/2, verbose = 2)
        self.execute = self.instance.execute
        # Force compilation as the setup step
        self._setup = self.instance.execute


# Classes for strided correlations

PAD_LEN = 8 # Keep everything aligned on cache-line boundary (64 bytes)

#     NITER:  How many steps of the in and out strides to take
#     NLEN: Length of input to read from each stride
#     ISTRIDE: Length of stride of input
#     OSTRIDE: Length of stride of (non-streamed) output

class CorrStridedBase(object):
    def __init__(self, in1, in2, out, ncorr, nistride, nostride, verbose = 0):        
        if (in1.dtype != _np.complex64 or in2.dtype != _np.complex64 or out.dtype != _np.complex64):
            raise ValueError("Only complex64 arrays permitted")
        if (nistride < ncorr) or (nostride < ncorr):
            raise ValueError("I/O strides must be >= correlation stride")
        if len(in1) != len(in2):
            raise ValueError("Input arrays must have the same length")
        if len(out) < len(in1):
            raise ValueError("Input arrays cannot be shorter than output arrays")
        # At the moment, we only process blocks whose length is divisble
        # by 'max_chunk'
        if ( (max_chunk % nistride) != 0):
            raise ValueError("'max_chunk' ({0}) must be divisble by input stride".format(max_chunk))
        self.verbose = verbose
        self.in1 = in1
        self.in2 = in2
        self.out = out
        self.ncorr = ncorr
        self.nistride = nistride
        self.nostride = nostride
        self.nbatch = max_chunk/self.nistride
        self.inlen = len(self.in1)
        # Note, all of the following assumes that the __init__ of the subclass is
        # called *before* this init
        #
        # First, support code. The factor of two is because our arrays will be seen
        # by the C-correlation function as real arrays rather than complex.
        self.support = self.support.replace('NLEN', str(self.ncorr * 2))
        self.support = self.support.replace('NITER', str(self.nbatch))
        self.support = self.support.replace('ISTRIDE', str(self.nistride * 2))
        self.support = self.support.replace('OSTRIDE', str(self.nostride * 2))
        # Next, the function code. 
        self.nblocks = self.inlen/(self.nbatch * self.nistride)
        self.code = self.code.replace('NBLOCKS', str(self.nblocks))
        self.code = self.code.replace('ISIZE', str(self.nistride * 2 * self.nbatch))
        self.code = self.code.replace('OSIZE', str(self.nostride * 2 * self.nbatch))

class CorrStridedNoStreaming(CorrStridedBase):
    def __init__(self, in1, in2, out, ncorr, nistride, nostride, verbose = 0):
        self.support = omp_support + corr_strided_nostream_support
        self.code = corr_strided_nostream_code
        super(CorrStridedNoStreaming, self).__init__(in1, in2, out, ncorr, 
                                                     nistride, nostride, verbose = verbose)

    def execute(self):
        in1 = _np.array(self.in1, copy = False).view(dtype = float32)
        in2 = _np.array(self.in2, copy = False).view(dtype = float32)
        out = _np.array(self.out, copy = False).view(dtype = float32)
        inline(self.code, ['in1', 'in2', 'out'],
               extra_compile_args = ['-march=native -O3 -w'] + omp_flags,
               support_code = self.support, auto_downcast = 1, verbose = self.verbose,
               libraries = omp_libs)

class CorrStridedStreamIn(CorrStridedBase):
    def __init__(self, in1, in2, out, ncorr, nistride, nostride, verbose = 0):
        self.support = omp_support + corr_strided_streamin_support
        self.code = corr_strided_streamin_code
        super(CorrStridedStreamIn, self).__init__(in1, in2, out, ncorr, 
                                                  nistride, nostride, verbose = verbose)

    def execute(self):
        in1 = _np.array(self.in1, copy = False).view(dtype = float32)
        in2 = _np.array(self.in2, copy = False).view(dtype = float32)
        out = _np.array(self.out, copy = False).view(dtype = float32)
        inline(self.code, ['in1', 'in2', 'out'],
               extra_compile_args = ['-march=native -O3 -w'] + omp_flags,
               support_code = self.support, auto_downcast = 1, verbose = self.verbose,
               libraries = omp_libs)

class CorrStridedStreamInOut(object):
    def __init__(self, in1, in2, outstream, outstore, ncorr, nistride,
                 nostride_stream, nostride_store, verbose = 0):        
        if (in1.dtype != _np.complex64 or in2.dtype != _np.complex64
            or outstream.dtype != _np.complex64 or outstore.dtype != _np.complex64):
            raise ValueError("Only complex64 arrays permitted")
        if (nistride < ncorr) or (nostride_stream < ncorr) or (nostride_store < ncorr):
            raise ValueError("I/O strides must be >= correlation stride")
        if len(in1) != len(in2):
            raise ValueError("Input arrays must have the same length")
        if (len(outstream) < len(in1)) or (len(outstore) < len(in1)):
            raise ValueError("Input arrays cannot be shorter than output arrays")
        # At the moment, we only process blocks whose length is divisble
        # by 'max_chunk'
        if ( (max_chunk % nistride) != 0):
            raise ValueError("'max_chunk' ({0}) must be divisble by input stride".format(max_chunk))
        self.verbose = verbose
        self.in1 = in1
        self.in2 = in2
        self.outstream = outstream
        self.outstore = outstore
        self.ncorr = ncorr
        self.nistride = nistride
        self.nostride_stream = nostride_stream
        self.nostride_store = nostride_store
        self.nbatch = max_chunk/self.nistride
        self.inlen = len(self.in1)
        self.support = omp_support + corr_strided_streaminout_support
        self.code = corr_strided_streaminout_code
        # Note, all of the following assumes that the __init__ of the subclass is
        # called *before* this init
        #
        # First, support code. The factor of two is because our arrays will be seen
        # by the C-correlation function as real arrays rather than complex.
        self.support = self.support.replace('NLEN', str(self.ncorr * 2))
        self.support = self.support.replace('NITER', str(self.nbatch))
        self.support = self.support.replace('ISTRIDE', str(self.nistride * 2))
        self.support = self.support.replace('OSTRIDE_STREAM', str(self.nostride_stream * 2))
        self.support = self.support.replace('OSTRIDE_STORE', str(self.nostride_store * 2))
        # Next, the function code. 
        self.nblocks = self.inlen/(self.nbatch * self.nistride)
        self.code = self.code.replace('NBLOCKS', str(self.nblocks))
        self.code = self.code.replace('ISIZE', str(self.nistride * 2 * self.nbatch))
        self.code = self.code.replace('OSIZE_STREAM', str(self.nostride_stream * 2 * self.nbatch))
        self.code = self.code.replace('OSIZE_STORE', str(self.nostride_store * 2 * self.nbatch))

    def execute(self):
        in1 = _np.array(self.in1, copy = False).view(dtype = float32)
        in2 = _np.array(self.in2, copy = False).view(dtype = float32)
        outstream = _np.array(self.outstream, copy = False).view(dtype = float32)
        outstore = _np.array(self.outstore, copy = False).view(dtype = float32)
        inline(self.code, ['in1', 'in2', 'outstream', 'outstore'],
               extra_compile_args = ['-march=native -O3 -w'] + omp_flags,
               support_code = self.support, auto_downcast = 1, verbose = self.verbose,
               libraries = omp_libs)

class StridedNostream(BaseCorrProblem):
    def __init__(self, size):
        super(StridedNostream, self).__init__(size = size, pad = PAD_LEN)
        self.instance = CorrStridedNoStreaming(self.i1.data, self.i2.data, self.o2.data,
                                               self.nsqrt/2, self.nsqrt, self.nsqrt + PAD_LEN, 
                                               verbose = 2)
        self.execute = self.instance.execute
        # Force compilation as the setup step
        self._setup = self.instance.execute

class StridedStreamIn(BaseCorrProblem):
    def __init__(self, size):
        super(StridedStreamIn, self).__init__(size = size, pad = PAD_LEN)
        self.instance = CorrStridedStreamIn(self.i1.data, self.i2.data, self.o2.data,
                                            self.nsqrt/2, self.nsqrt, self.nsqrt + PAD_LEN, 
                                            verbose = 2)
        self.execute = self.instance.execute
        # Force compilation as the setup step
        self._setup = self.instance.execute

class StridedStreamInOut(BaseCorrProblem):
    def __init__(self, size):
        super(StridedStreamInOut, self).__init__(size = size, pad = PAD_LEN)
        self.instance = CorrStridedStreamInOut(self.i1.data, self.i2.data, self.o2.data, self.o1.data,
                                               self.nsqrt/2, self.nsqrt, self.nsqrt, 
                                               self.nsqrt + PAD_LEN, verbose = 2)
        self.execute = self.instance.execute
        # Force compilation as the setup step
        self._setup = self.instance.execute

def transpose_numpy(vector, nrow):
    ncol = len(vector)/nrow
    return Array(vector.data.copy().reshape(nrow, ncol).transpose().reshape(len(vector)).copy())

def remove_padding(vector, stride, pad):
    if (len(vector) % (stride+pad)) != 0:
        raise ValueError("Length of vector must be divisible by stride+pad")
    nbatch = len(vector)/(stride+pad)
    nopadding = zeros(nbatch*stride, dtype = vector.dtype)
    for i in range(nbatch):
        nopadding.data[i*stride:(i+1)*stride] = vector.data[i*(stride+pad):i*(stride+pad)+stride]
    return nopadding

_class_dict = { 'contig' : ContigNostream,
                'contig_streamin' : ContigStreamIn,
                'contig_streaminout' : ContigStreamInOut,
                'strided' : StridedNostream,
                'strided_streamin' : StridedStreamIn,
                'strided_streaminout' : StridedStreamInOut
                }

valid_methods = _class_dict.keys()

def parse_problem(probstring, method='numpy'):
    """
    This function takes a string of the form <number>, and
    another argument indicating which class type to return. 


    It returns the class and size, so that the call:
        MyClass, n  = parse_problem(probstring, method)
    should usually be followed by:
        MyProblem = MyClass(n)
    """
    prob_class = _class_dict[method]
    size = int(probstring)

    return prob_class, size
