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
import pycbc
from pycbc.types import zeros, complex64, complex128, float32, float64, Array
from pycbc import scheme as _scheme
from scipy.weave import inline
import numpy as _np
import ctypes
from pycbc import libutils

"""
This module contains various long-strings of code intended to be used by other
modules when they implement the thresholding and clustering steps of matched
filtering. Most of these will refer to various constants that must be string-
replaced before the code can be compiled by weave.

At present, these codes also assume that all inputs are SIMD aligned and that
the lengths of various arrays are multiples of the SIMD length, though in
actual applications that will be relaxed.

All array arguments are single-precision real arrays, assuming that the complex
arrays are in interleaved storage formats.  Constants that are substituted before
compilation should accordingly be for the relevant lengths as real arrays.

The basic functions are vectorized by not parallelized; they are intended to
be called from other functions using OpenMP to parallelize.
"""

# What we need to support OpenMP, at least in gcc...

omp_support = """
#include <omp.h>
"""

omp_libs = ['gomp']
omp_flags = ['-fopenmp']

### The following is used by all of our codes, so we just define it once here

tc_common_support = """
#include <x86intrin.h>
#include <stdint.h>
#include <error.h>
#include <complex>

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

#if _HAVE_AVX
#define ALGN 32
#define ALGN_FLT 8
#define ALGN_DBL 4
#else
#define ALGN 16
#define ALGN_FLT 4
#define ALGN_DBL 2
#endif

"""

# The following maximizes over an interval that can be no longer than
# the clustering window.  We do *not* assume alignment, because the window length
# itself may not be a multiple of the alignment.

thresh_cluster_support = tc_common_support + """
void max_simd(float * __restrict inarr, float * __restrict mval,
              float * __restrict norm, uint32_t *mloc,
              uint32_t nstart, uint32_t howmany){

  uint32_t i, curr_mloc;
  float re, im, curr_norm, curr, curr_mval[2], *arrptr;

  // Set everything up.  Doesn't depend on any SIMD.
  curr_norm = 0.0;
  curr_mval[0] = 0.0;
  curr_mval[1] = 0.0;
  curr_mloc = 0;
  arrptr = inarr;

#if _HAVE_AVX

  __m256 norm_lo, cval_lo, arr_lo, reg0_lo, reg1_lo;
  __m256d mloc_lo, count_lo, incr_lo;
  __m256 norm_hi, cval_hi, arr_hi, reg0_hi, reg1_hi;
  __m256d mloc_hi, count_hi, incr_hi;
  float output_vals[2*ALGN_FLT] __attribute__ ((aligned(ALGN)));
  float output_norms[2*ALGN_FLT] __attribute__ ((aligned(ALGN)));
  double output_locs[2*ALGN_DBL] __attribute__ ((aligned(ALGN)));
  double curr_mloc_dbl;
  uint32_t peel, misalgn, j;

  misalgn = (uint32_t)  (((uintptr_t) inarr) % ALGN);
  if (misalgn % 2*sizeof(float)) {
    error(EXIT_FAILURE, 0, "Array given to max_simd must be aligned on a least a complex float boundary\\n");
  }
  peel = ( misalgn ? ((ALGN - misalgn) / (sizeof(float))) : 0 );
  peel = (peel > howmany ? howmany : peel);

  // Below is the only place i gets initialized! It should never be initialized
  // in the 'for' loops.
  i = 0;

  // Peel off any unaligned beginning for the array.
  for ( ; i < peel; i += 2){
    re = *arrptr;
    im = *(arrptr+1);
    curr = re*re + im*im;
    if (curr > curr_norm){
        curr_mval[0] = re;
        curr_mval[1] = im;
        curr_mloc = i;
        curr_norm = curr;
    }
    arrptr += 2;
  }

  // Now a loop, unrolled once (which is all the AVX registers we can use)

  // AVX---as opposed to AVX2---has very little support for packed
  // integer types.  For instance, one cannot add two packed int
  // SIMD vectors.  So we compromise by storing the array indices
  // as doubles, which should be more than enough to exactly represent
  // a 32 bit integer.

  _mm256_zeroupper();

  // Note that the "set_p{s,d}" functions take their arguments from
  // most-significant value to least.

  incr_lo = _mm256_set_pd(6.0, 4.0, 2.0, 0.0);
  count_lo = _mm256_set1_pd( (double) i);
  count_lo = _mm256_add_pd(count_lo, incr_lo);
  incr_lo = _mm256_set_pd(1.0*ALGN_FLT, 1.0*ALGN_FLT, 1.0*ALGN_FLT, 1.0*ALGN_FLT);
  count_hi = _mm256_add_pd(count_lo, incr_lo);
  incr_lo = _mm256_set_pd(2.0*ALGN_FLT, 2.0*ALGN_FLT, 2.0*ALGN_FLT, 2.0*ALGN_FLT);
  incr_hi = _mm256_set_pd(2.0*ALGN_FLT, 2.0*ALGN_FLT, 2.0*ALGN_FLT, 2.0*ALGN_FLT);

  // Now count_lo and count_hi have the current indices into the array

  // We don't need to initialize to what we found in the peel-off loop,
  // since we'll store the results of the high and low SIMD loops into
  // an array that we then loop over comparing with the peel-off values.
  mloc_lo = _mm256_setzero_pd();
  norm_lo = _mm256_setzero_ps();
  cval_lo = _mm256_setzero_ps();

  mloc_hi = _mm256_setzero_pd();
  norm_hi = _mm256_setzero_ps();
  cval_hi = _mm256_setzero_ps();

  for (; i <= howmany - 2*ALGN_FLT; i += 2*ALGN_FLT){
      // Load everything into a register
      arr_lo =  _mm256_load_ps(arrptr);
      arr_hi = _mm256_load_ps(arrptr + ALGN_FLT);

      reg0_lo = _mm256_mul_ps(arr_lo, arr_lo);               // 4 x [re*re, im*im]
      reg1_lo = _mm256_shuffle_ps(reg0_lo, reg0_lo, 0xB1);   // 4 x [im*im, re*re]
      reg0_lo = _mm256_add_ps(reg0_lo, reg1_lo);             // 4 x [re^2 +im^2, re^2 +im^2]
      reg1_lo = _mm256_cmp_ps(reg0_lo, norm_lo, _CMP_GT_OQ); // Now a mask for where > curr_norm

      // Now use the mask to selectively update complex value, norm, and location
      mloc_lo = _mm256_blendv_pd(mloc_lo, count_lo, _mm256_castps_pd(reg1_lo) );
      norm_lo = _mm256_blendv_ps(norm_lo, reg0_lo, reg1_lo);
      cval_lo = _mm256_blendv_ps(cval_lo, arr_lo, reg1_lo);

      reg0_hi = _mm256_mul_ps(arr_hi, arr_hi);               // 4 x [re*re, im*im]
      reg1_hi = _mm256_shuffle_ps(reg0_hi, reg0_hi, 0xB1);   // 4 x [im*im, re*re]
      reg0_hi = _mm256_add_ps(reg0_hi, reg1_hi);             // 4 x [re^2 +im^2, re^2 +im^2]
      reg1_hi = _mm256_cmp_ps(reg0_hi, norm_hi, _CMP_GT_OQ); // Now a mask for where > curr_norm

      // Now use the mask to selectively update complex value, norm, and location
      mloc_hi = _mm256_blendv_pd(mloc_hi, count_hi, _mm256_castps_pd(reg1_hi) );
      norm_hi = _mm256_blendv_ps(norm_hi, reg0_hi, reg1_hi);
      cval_hi = _mm256_blendv_ps(cval_hi, arr_hi, reg1_hi);

      count_lo = _mm256_add_pd(count_lo, incr_lo);
      count_hi = _mm256_add_pd(count_hi, incr_hi);
      arrptr += 2*ALGN_FLT;
  }

  // Finally, one last SIMD loop that is not unrolled, just in case we can.
  // We don't reset increments because we won't use them after this loop, and
  // this loop executes at most once.

  for (; i <= howmany - ALGN_FLT; i += ALGN_FLT){
      // Load everything into a register
      arr_lo = _mm256_load_ps(arrptr);

      reg0_lo = _mm256_mul_ps(arr_lo, arr_lo);               // 4 x [re*re, im*im]
      reg1_lo = _mm256_shuffle_ps(reg0_lo, reg0_lo, 0xB1);   // 4 x [im*im, re*re]
      reg0_lo = _mm256_add_ps(reg0_lo, reg1_lo);             // 4 x [re^2 +im^2, re^2 +im^2]
      reg1_lo = _mm256_cmp_ps(reg0_lo, norm_lo, _CMP_GT_OQ); // Now a mask for where > curr_norm

      // Now use the mask to selectively update complex value, norm, and location
      mloc_lo = _mm256_blendv_pd(mloc_lo, count_lo, _mm256_castps_pd(reg1_lo) );
      norm_lo = _mm256_blendv_ps(norm_lo, reg0_lo, reg1_lo);
      cval_lo = _mm256_blendv_ps(cval_lo, arr_lo, reg1_lo);

      arrptr += ALGN_FLT;
  }

  // Now write out the results to our temporary tables:
  _mm256_store_ps(output_vals, cval_lo);
  _mm256_store_ps(output_vals + ALGN_FLT, cval_hi);
  _mm256_store_ps(output_norms, norm_lo);
  _mm256_store_ps(output_norms + ALGN_FLT, norm_hi);
  _mm256_store_pd(output_locs, mloc_lo);
  _mm256_store_pd(output_locs + ALGN_DBL, mloc_hi);

  _mm256_zeroupper();

  // Now loop over our output arrays
  // When we start, curr_norm, curr_mloc, and
  // curr_mval all have the values they had at
  // the end of the *first* peeling loop

  curr_mloc_dbl = (double) curr_mloc;

  for (j = 0; j < 2*ALGN_FLT; j += 2){
    if (output_norms[j] > curr_norm) {
      curr_norm = output_norms[j];
      curr_mloc_dbl = output_locs[j/2];
      curr_mval[0] = output_vals[j];
      curr_mval[1] = output_vals[j+1];
    }
  }

  curr_mloc = (uint32_t) curr_mloc_dbl;

#elif _HAVE_SSE4_1

  __m128 norm_lo, cval_lo, arr_lo, reg0_lo, reg1_lo;
  __m128d mloc_lo, count_lo, incr_lo;
  __m128 norm_hi, cval_hi, arr_hi, reg0_hi, reg1_hi;
  __m128d mloc_hi, count_hi, incr_hi;
  float output_vals[2*ALGN_FLT] __attribute__ ((aligned(ALGN)));
  float output_norms[2*ALGN_FLT] __attribute__ ((aligned(ALGN)));
  double output_locs[2*ALGN_DBL] __attribute__ ((aligned(ALGN)));
  double curr_mloc_dbl;
  uint32_t peel, misalgn, j;

  misalgn = (uint32_t)  (((uintptr_t) inarr) % ALGN);
  if (misalgn % 2*sizeof(float)) {
    error(EXIT_FAILURE, 0, "Array given to max_simd must be aligned on a least a complex float boundary");
  }
  peel = ( misalgn ? ((ALGN - misalgn) / (sizeof(float))) : 0 );
  peel = (peel > howmany ? howmany : peel);

  // Below is the only place i gets initialized! It should never be initialized
  // in the for loops.
  i = 0;

  // Peel off any unaligned beginning for the array.
  for ( ; i < peel; i += 2){
    re = *arrptr;
    im = *(arrptr+1);
    curr = re*re + im*im;
    if (curr > curr_norm){
        curr_mval[0] = re;
        curr_mval[1] = im;
        curr_mloc = i;
        curr_norm = curr;
    }
    arrptr += 2;
  }

  // Now a loop, unrolled once (which is all the SSE registers we can use)

  // Note that the "set_p{s,d}" functions take their arguments from
  // most-significant value to least.

  incr_lo = _mm_set_pd(2.0, 0.0);
  count_lo = _mm_set1_pd( (double) i);
  count_lo = _mm_add_pd(count_lo, incr_lo);
  incr_lo = _mm_set_pd(1.0*ALGN_FLT, 1.0*ALGN_FLT);
  count_hi = _mm_add_pd(count_lo, incr_lo);
  incr_lo = _mm_set_pd(2.0*ALGN_FLT, 2.0*ALGN_FLT);
  incr_hi = _mm_set_pd(2.0*ALGN_FLT, 2.0*ALGN_FLT);

  // Now count_lo and count_hi have the current indices into the array

  // We don't need to initialize to what we found in the peel-off loop,
  // since we'll store the results of the high and low SIMD loops into
  // an array that we then loop over comparing with the peel-off values.
  mloc_lo = _mm_setzero_pd();
  norm_lo = _mm_setzero_ps();
  cval_lo = _mm_setzero_ps();

  mloc_hi = _mm_setzero_pd();
  norm_hi = _mm_setzero_ps();
  cval_hi = _mm_setzero_ps();

  for (; i <= howmany - 2*ALGN_FLT; i += 2*ALGN_FLT){
      // Load everything into a register
      arr_lo =  _mm_load_ps(arrptr);
      arr_hi = _mm_load_ps(arrptr + ALGN_FLT);

      reg0_lo = _mm_mul_ps(arr_lo, arr_lo);               // 2 x [re*re, im*im]
      reg1_lo = _mm_shuffle_ps(reg0_lo, reg0_lo, 0xB1);   // 2 x [im*im, re*re]
      reg0_lo = _mm_add_ps(reg0_lo, reg1_lo);             // 2 x [re^2 +im^2, re^2 +im^2]
      reg1_lo = _mm_cmpgt_ps(reg0_lo, norm_lo);           // Now a mask for where > curr_norm

      // Now use the mask to selectively update complex value, norm, and location
      mloc_lo = _mm_blendv_pd(mloc_lo, count_lo, _mm_castps_pd(reg1_lo) );
      norm_lo = _mm_blendv_ps(norm_lo, reg0_lo, reg1_lo);
      cval_lo = _mm_blendv_ps(cval_lo, arr_lo, reg1_lo);

      reg0_hi = _mm_mul_ps(arr_hi, arr_hi);               // 2 x [re*re, im*im]
      reg1_hi = _mm_shuffle_ps(reg0_hi, reg0_hi, 0xB1);   // 2 x [im*im, re*re]
      reg0_hi = _mm_add_ps(reg0_hi, reg1_hi);             // 2 x [re^2 +im^2, re^2 +im^2]
      reg1_hi = _mm_cmpgt_ps(reg0_hi, norm_hi);           // Now a mask for where > curr_norm

      // Now use the mask to selectively update complex value, norm, and location
      mloc_hi = _mm_blendv_pd(mloc_hi, count_hi, _mm_castps_pd(reg1_hi) );
      norm_hi = _mm_blendv_ps(norm_hi, reg0_hi, reg1_hi);
      cval_hi = _mm_blendv_ps(cval_hi, arr_hi, reg1_hi);

      count_lo = _mm_add_pd(count_lo, incr_lo);
      count_hi = _mm_add_pd(count_hi, incr_hi);
      arrptr += 2*ALGN_FLT;
  }

  // Finally, one last SIMD loop that is not unrolled, just in case we can.
  // We don't reset increments because we won't use them after this loop, and
  // this loop executes at most once.

  for (; i <= howmany - ALGN_FLT; i += ALGN_FLT){
      // Load everything into a register
      arr_lo =  _mm_load_ps(arrptr);

      reg0_lo = _mm_mul_ps(arr_lo, arr_lo);               // 2 x [re*re, im*im]
      reg1_lo = _mm_shuffle_ps(reg0_lo, reg0_lo, 0xB1);   // 2 x [im*im, re*re]
      reg0_lo = _mm_add_ps(reg0_lo, reg1_lo);             // 2 x [re^2 +im^2, re^2 +im^2]
      reg1_lo = _mm_cmpgt_ps(reg0_lo, norm_lo);           // Now a mask for where > curr_norm

      // Now use the mask to selectively update complex value, norm, and location
      mloc_lo = _mm_blendv_pd(mloc_lo, count_lo, _mm_castps_pd(reg1_lo) );
      norm_lo = _mm_blendv_ps(norm_lo, reg0_lo, reg1_lo);
      cval_lo = _mm_blendv_ps(cval_lo, arr_lo, reg1_lo);

      arrptr += ALGN_FLT;
  }

  // Now write out the results to our temporary tables:
  _mm_store_ps(output_vals, cval_lo);
  _mm_store_ps(output_vals + ALGN_FLT, cval_hi);
  _mm_store_ps(output_norms, norm_lo);
  _mm_store_ps(output_norms + ALGN_FLT, norm_hi);
  _mm_store_pd(output_locs, mloc_lo);
  _mm_store_pd(output_locs + ALGN_DBL, mloc_hi);

  // Now loop over our output arrays
  // When we start, curr_norm, curr_mloc, and
  // curr_mval all have the values they had at
  // the end of the *first* peeling loop

  curr_mloc_dbl = (double) curr_mloc;

  for (j = 0; j < 2*ALGN_FLT; j += 2){
    if (output_norms[j] > curr_norm) {
      curr_norm = output_norms[j];
      curr_mloc_dbl = output_locs[j/2];
      curr_mval[0] = output_vals[j];
      curr_mval[1] = output_vals[j+1];
    }
  }

  curr_mloc = (uint32_t) curr_mloc_dbl;

#else
 // If we have no SSE, all we have to do is initialize
 // our loop counter, and the last "cleanup" loop
 // will in fact do all the work.

 i = 0;

#endif

  for ( ; i < howmany; i += 2){
    re = *arrptr;
    im = *(arrptr+1);
    curr = re*re + im*im;
    if (curr > curr_norm){
        curr_mval[0] = re;
        curr_mval[1] = im;
        curr_mloc = i;
        curr_norm = curr;
    }
    arrptr += 2;
  }

  // Store our answers and return
  *mval = curr_mval[0];
  *(mval+1) = curr_mval[1];
  *norm = curr_norm;

  // Note that curr_mloc is a real array index, but we
  // must return the index into the complex array.
  *mloc = (curr_mloc/2) + nstart;

  return;

}

void windowed_max(std::complex<float> * __restrict inarr, const uint32_t arrlen,
                  std::complex<float> * __restrict cvals, float * __restrict norms,
                  uint32_t * __restrict locs, const uint32_t winsize,
                  const uint32_t startoffset){


  /*

   This function fills in the arrays cvals, norms, and locs, with the max (as
   complex value, norm, and location, resp.) of the array inarr.  The length of
   inarr is arrlen, and the function assumes that it computes the max over successive
   windows of length winsize, starting at the beginning of the array and continuing
   to the end.  If winsize does not evenly divide arrlen, then the last partial window
   will be maximized over as well.  If winsize is greater than arrlen, then just one
   maximization will be performed over the (partial) array inarr.

   Thus, in all cases, the length of cvals, norms, and locs should be:
      nwindows = ( (arrlen % winsize) ? (arrlen/winsize) + 1 : (arrlen/winsize) )

   Note that all input sizes are for *complex* arrays; the function this function calls
   often requires *real* arrays, and lengths will be converted where appropriate.

  */

  uint32_t i, nwindows;

  nwindows = ( (arrlen % winsize) ? (arrlen/winsize) + 1 : (arrlen/winsize) );

  // Everything but the last window, which may not be full length

  for (i = 0; i < nwindows-1; i++){
    // The factor of 2 multiplying lengths[i] is because max_simd needs its length as a real
    // length, not complex.  But startpts and startoffset are complex values.
    max_simd((float *) &inarr[i*winsize], (float *) &cvals[i],
             &norms[i], &locs[i], startoffset + i*winsize, 2*winsize);
  }
  // Now the last window (which will be the only window if arrlen <= winzise)
  max_simd((float *) &inarr[i*winsize], (float *) &cvals[i],
             &norms[i], &locs[i], startoffset + i*winsize, 2*(arrlen - i*winsize));

  return; 
}

int parallel_thresh_cluster(std::complex<float> * __restrict inarr, const uint32_t arrlen,
                            std::complex<float> * __restrict values, uint32_t * __restrict locs, 
                            const float thresh, const uint32_t winsize, const uint32_t segsize){

  uint32_t i, nsegs, nwins_ps, last_arrlen, last_nwins_ps, outlen;
  uint32_t *seglens, *mlocs, curr_loc;
  float *norms, thr_sqr, curr_norm;
  std::complex<float> *cvals, curr_cval;
  int cnt;

  thr_sqr = thresh*thresh;
  
  nsegs = ( (arrlen % segsize) ? (arrlen/segsize) + 1 : (arrlen/segsize) );
  nwins_ps = ( (segsize % winsize) ? (segsize/winsize) + 1 : (segsize/winsize) );
  // Our logic will be to treat the last segment differently always.  However if 
  // segsize evenly divides arrlen, then the last segment will be no different.
  // The following ternary operator captures that logic:
  last_arrlen = ( (arrlen % segsize) ? (arrlen - (nsegs-1) * segsize) : (segsize) );
  last_nwins_ps = ( (last_arrlen % winsize) ? (last_arrlen/winsize) + 1 : (last_arrlen/winsize) );
  // Then the total length of the working arrays we must dynamically allocate is:
  outlen = (nsegs-1) * nwins_ps + last_nwins_ps;

  // Now dynamic allocation.  No reason really to align this memory; it will be parceled
  // out to different cores anyway.

  cvals = (std::complex<float> *) malloc(outlen * sizeof(std::complex<float>) );
  norms = (float *) malloc(outlen * sizeof(float) );
  mlocs = (uint32_t *) malloc(outlen * sizeof(uint32_t) );

  // The next array allows us to dynamically communicate possibly changed sizes to the
  // many parallel calls to windowed_max:

  seglens = (uint32_t *) malloc(nsegs * sizeof(uint32_t) );

  // check to see if anything failed
  if ( (cvals == NULL) || (norms == NULL) || (mlocs == NULL) || (seglens == NULL) ){
    error(EXIT_FAILURE, ENOMEM, "Could not allocate temporary memory needed by parallel_thresh_cluster");
  }

  for (i = 0; i < (nsegs-1); i++){
    seglens[i] = segsize;
  }
  seglens[i] = last_arrlen;

  // Now the real work, in an OpenMP parallel for loop:
#pragma omp parallel for schedule(dynamic,1)
  for (i = 0; i < nsegs; i++){
    windowed_max(&inarr[i*segsize], seglens[i], &cvals[i*nwins_ps],
                 &norms[i*nwins_ps], &mlocs[i*nwins_ps], winsize, i*segsize);
  }
  
  // We should now have the requisite max in cvals, norms, and mlocs.
  // So one last loop...
  cnt = 0;
  curr_norm = 0.0;
  curr_loc = 0;
  for (i = 0; i < outlen; i++){
    if (norms[i] > thr_sqr){
      if (!cnt){
        // We only do this the first time we find a point above threshold.
        cnt = 1;
        curr_norm = norms[i];
        curr_loc = mlocs[i];
        curr_cval = cvals[i];
      }
      if ( (mlocs[i] - curr_loc) > winsize){
        // The last one survived, so write
        // it out. 
        values[cnt-1] = curr_cval;
        locs[cnt-1] = curr_loc;
        curr_cval = cvals[i];
        curr_norm = norms[i];
        curr_loc = mlocs[i];
        cnt += 1;        
      } else if (norms[i] > curr_norm) {
        curr_cval = cvals[i];
        curr_norm = norms[i];
        curr_loc = mlocs[i];
      }
    }
  }

  // Note that in the above logic, we have only written
  // values out if we found another point above threshold
  // after the current one and further away. That's also
  // the only time we increment cnt.  So if we found
  // *anything*, we have one more point to write; if we
  // found more than one point, the value of cnt is also
  // one too low.

  if (cnt > 0){
    if (cnt > 1){
      cnt += 1;
    }
    values[cnt-1] = curr_cval;
    locs[cnt-1] = curr_loc;
  }

  free(cvals);
  free(norms);
  free(mlocs);
  free(seglens);

  return cnt;
}

"""

### Now some actual code that just implements the different
### correlations in a parallelized fashion.

# First, a simple thing, that just does the max...

max_only_code = """
max_simd(inarr, mval, norm, (uint32_t *) mloc, (uint32_t) nstart[0], (uint32_t) howmany[0]);
"""

class MaxOnlyObject(object):
    def __init__(self, inarray, verbose=0):
        self.inarr = _np.array(inarray.data, copy=False).view(dtype = float32)
        self.howmany = _np.zeros(1, dtype = _np.uint32)
        self.howmany[0] = len(self.inarr)
        self.nstart = _np.zeros(1, dtype = _np.uint32)
        self.nstart[0] = 0
        self.cmplx_mval = zeros(1, dtype = complex64)
        self.mval = _np.array(self.cmplx_mval.data, copy = False).view(dtype = float32)
        self.norm = _np.zeros(1, dtype = float32)
        self.mloc = _np.zeros(1, dtype = _np.uint32)
        self.code = max_only_code
        self.support = thresh_cluster_support
        self.verbose = verbose

    def execute(self):
        inarr = self.inarr
        mval = self.mval
        norm = self.norm
        mloc = self.mloc
        nstart = self.nstart
        howmany = self.howmany
        inline(self.code, ['inarr', 'mval', 'norm', 'mloc', 'nstart', 'howmany'],
               extra_compile_args = ['-march=native -O3 -w'],
               #extra_compile_args = ['-mno-avx -mno-sse2 -mno-sse3 -mno-ssse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-sse4a -O2 -w'],
               #extra_compile_args = ['-msse4.1 -O3 -w'],
               support_code = self.support, auto_downcast = 1, verbose = self.verbose)

class MaxProblem(_mb.MultiBenchProblem):
    def __init__(self, size):
        self.arr = zeros(size, dtype=complex64)
        self.maxobj = MaxOnlyObject(self.arr)
        self.execute = self.maxobj.execute
        
    def _setup(self):
        self.execute()

global_winsize = 4096

windowed_max_code = """
windowed_max(inarr, (uint32_t) arrlen[0], cvals, norms, (uint32_t *) locs, (uint32_t ) winsize[0],
             (uint32_t) startoffset[0]);
"""

class WindowedMaxObject(object):
    def __init__(self, inarray, winsize, verbose=0):
        self.inarr = _np.array(inarray.data, copy=False)
        self.arrlen = _np.zeros(1, dtype = _np.uint32)
        self.arrlen[0] = len(self.inarr)
        self.len_win = winsize
        nwindows = int( len(self.inarr) / winsize)
        if (nwindows * winsize < len(self.inarr)):
            nwindows = nwindows + 1
        self.nwindows = nwindows
        self.cvals = _np.zeros(self.nwindows, dtype = complex64)
        self.norms = _np.zeros(self.nwindows, dtype = float32)
        self.locs = _np.zeros(self.nwindows, dtype = _np.uint32)
        self.winsize = _np.zeros(1, dtype = _np.uint32)
        self.winsize[0] = self.len_win
        self.startoffset = _np.zeros(1, dtype = _np.uint32)
        self.code = windowed_max_code
        self.support = thresh_cluster_support
        self.verbose = verbose

    def execute(self):
        inarr = self.inarr
        arrlen = self.arrlen
        cvals = self.cvals
        norms = self.norms
        locs = self.locs
        winsize = self.winsize
        startoffset = self.startoffset
        inline(self.code, ['inarr', 'arrlen', 'cvals', 'norms', 'locs', 'winsize', 'startoffset'],
               extra_compile_args = ['-march=native -O3 -w'],
               #extra_compile_args = ['-mno-avx -mno-sse2 -mno-sse3 -mno-ssse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-sse4a -O2 -w'],
               #extra_compile_args = ['-msse4.1 -O3 -w'],
               support_code = self.support, auto_downcast = 1, verbose = self.verbose)

class WindowedMaxProblem(_mb.MultiBenchProblem):
    def __init__(self, size):
        self.arr = zeros(size, dtype=complex64)
        self.winmaxobj = WindowedMaxObject(self.arr, global_winsize)
        self.execute = self.winmaxobj.execute
        
    def _setup(self):
        self.execute()

def check_pow_two(n):
    return ( (n != 0) and ( (n & (n-1)) == 0) )

#int parallel_thresh_cluster(std::complex<float> * __restrict inarr, const uint32_t arrlen,
#                            std::complex<float> * __restrict values, uint32_t * restrict locs, 
#                            const float thresh, const uint32_t winsize, const uint32_t segsize){


thresh_cluster_code = """
return_val = parallel_thresh_cluster(inarr, (uint32_t) arrlen[0], values, locs,
                                     thresh[0], (uint32_t) winsize[0], (uint32_t) segsize[0]);
"""

class ThreshClusterObject(object):
    def __init__(self, inarray, threshold, winsize, segsize, verbose=0):
        self.inarr = _np.array(inarray.data, copy=False)
        self.arrlen = _np.zeros(1, dtype = _np.uint32)
        self.arrlen[0] = len(self.inarr)
        nwindows = int( len(self.inarr) / winsize)
        if (nwindows * winsize < len(self.inarr)):
            nwindows = nwindows + 1
        self.nwindows = nwindows
        self.values = _np.zeros(self.nwindows, dtype = complex64)
        self.locs = _np.zeros(self.nwindows, dtype = _np.uint32)
        self.thresh = _np.zeros(1, dtype = float32)
        self.thresh[0] = threshold
        self.winsize = _np.zeros(1, dtype = _np.uint32)
        self.winsize[0] = winsize
        self.segsize = _np.zeros(1, dtype = _np.uint32)
        self.segsize[0] = segsize
        self.code = thresh_cluster_code
        self.support = omp_support + thresh_cluster_support
        self.verbose = verbose

    def execute(self):
        inarr = self.inarr
        arrlen = self.arrlen
        values = self.values
        locs = self.locs
        thresh = self.thresh
        winsize = self.winsize
        segsize = self.segsize
        retval = inline(self.code, ['inarr', 'arrlen', 'values', 'locs', 'thresh', 'winsize', 'segsize'],
                        extra_compile_args = ['-march=native -O3 -w'] + omp_flags,
                        #extra_compile_args = ['-mno-avx -mno-sse2 -mno-sse3 -mno-ssse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-sse4a -O2 -w'],
                        #extra_compile_args = ['-msse4.1 -O3 -w'],
                        support_code = self.support, libraries = omp_libs,
                        auto_downcast = 1, verbose = self.verbose)
        self.cvals = self.values[0:retval]
        self.lvals = self.locs[0:retval]

global_segsize = 32768

class ThreshClusterProblem(_mb.MultiBenchProblem):
    def __init__(self, size):
        self.arr = zeros(size, dtype=complex64)
        self.arr[-1] = 0.8+0.8j
        self.thresh = 1.0
        self.winsize = global_winsize
        self.segsize = global_segsize
        self.tcobj = ThreshClusterObject(self.arr, self.thresh, self.winsize, self.segsize)
        self.execute = self.tcobj.execute
        
    def _setup(self):
        self.execute()



_class_dict = { 'max' : MaxProblem,
                'win_max' : WindowedMaxProblem,
                'tc' : ThreshClusterProblem
              }

valid_methods = _class_dict.keys()

def parse_problem(probstring, method='max'):
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
