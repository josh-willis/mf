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
import pycbc.fft
import pycbc.fft.fftw as _fftw
from pycbc import scheme as _scheme
from pycbc import libutils
from scipy.weave import inline
import ctypes
import numpy as _np
from math import sqrt
import sys
from .corr import corr_contig_nostream_support
from .corr import omp_support, omp_flags, omp_libs

many_fft_support = omp_support + """
#include <complex>
#include <fftw3.h>
"""

# The code to do an FFT by hand.  Called using weave. Requires:
#     NJOBS1, NJOBS2, NCHUNK1, and NCHUNK2 to be substituted
#           before compilation
# Input arrays: vin, vout, vscratch
# Input plan arrays: plan1, plan2
# Input plans: tplan1, tplan2
#

many_corr_code = """
int j;

// First, correlate the vectors, in parallel
#pragma omp parallel for schedule(static, 1)
for (j = 0; j < NCPUS; j++){
   ccorrf_contig_nostream((float *) st[j], (float *) ht[j], (float *) qt[j]);
}
"""
just_fft_code = """
int j;

// Then, perform the FFTs, serially (but each FFT is
// parallelized)
for (j = 0; j < NCPUS; j++){
   fftwf_execute_dft((fftwf_plan) plan[0], (fftwf_complex *) qt[j],
                     (fftwf_complex *) snr);
}

"""

many_fft_code = """
int j;

// First, correlate the vectors, in parallel
#pragma omp parallel for schedule(static, 1)
for (j = 0; j < NCPUS; j++){
   ccorrf_contig_nostream((float *) st[j], (float *) ht[j], (float *) qt[j]);
}

// Then, perform the FFTs, serially (but each FFT is
// parallelized)
for (j = 0; j < NCPUS; j++){
   fftwf_execute_dft((fftwf_plan) plan[0], (fftwf_complex *) qt[j],
                     (fftwf_complex *) snr);
}
"""

many_fft_libs = ['fftw3f', 'fftw3f_omp', 'gomp', 'm']
# The following could return system libraries, but oh well,
# shouldn't hurt anything
many_fft_libdirs = libutils.pkg_config_libdirs(['fftw3f'])

rpath_list = []
for libdir in many_fft_libdirs:
    rpath = "-Wl,-rpath="+libdir
    rpath_list.append(rpath)

libfftw3f = _fftw.float_lib

def fftw_plan(size, nthreads = 1):
    if not _fftw._fftw_threaded_set:
        _fftw.set_threads_backend()
    if nthreads != _fftw._fftw_current_nthreads:
        _fftw._fftw_plan_with_nthreads(nthreads)
    # Convert a measure-level to flags
    flags = _fftw.get_flag(_fftw.get_measure_level(), aligned=True)
    fplan = libfftw3f.fftwf_plan_dft_1d
    fplan.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                      ctypes.c_int, ctypes.c_int]
    fplan.restype = ctypes.c_void_p
    inv = zeros(size, dtype = complex64)
    outv = zeros(size, dtype = complex64)
    res = fplan(size, inv.ptr, outv.ptr, _fftw.FFTW_BACKWARD, flags)
    del inv
    del outv
    return res

def check_pow_two(n):
    return ( (n != 0) and ( (n & (n-1)) == 0) )

class ManyFFTProblem(_mb.MultiBenchProblem):
    def __init__(self, size):
        # We'll do some arithmetic with these, so sanity check first:
        if not check_pow_two(size):
            raise ValueError("Only power-of-two sizes supported")

        self.ncpus = _scheme.mgr.state.num_threads
        self.size = size

        self.st = _np.zeros(self.ncpus, dtype = _np.uintp)
        self.ht = _np.zeros(self.ncpus, dtype = _np.uintp)
        self.qt = _np.zeros(self.ncpus, dtype = _np.uintp)
        self.snr = zeros(size, dtype = complex64)
        self.st_list = []
        self.ht_list = []
        self.qt_list = []
        for i in range(self.ncpus):
            st = zeros(size, dtype = complex64)
            self.st_list.append(st)
            self.st[i] = st.ptr
            ht = zeros(size, dtype = complex64)
            self.ht_list.append(ht)
            self.ht[i] = ht.ptr
            qt = zeros(size, dtype = complex64)
            self.qt_list.append(qt)
            self.qt[i] = qt.ptr
        self.plan = _np.zeros(1, dtype = _np.uintp)


        self.support = corr_contig_nostream_support + many_fft_support
        # We in fact only correlate the first half, but the correlation
        # code needs the length as a *real* array so the full length as
        # a *complex* array is the right thing to use here.
        self.support = self.support.replace('NLEN', str(self.size))
        self.code = many_fft_code
        self.code = self.code.replace('NCPUS', str(self.ncpus))
        self.corr_code = many_corr_code
        self.corr_code = self.corr_code.replace('NCPUS', str(self.ncpus))
        self.fft_code = just_fft_code
        self.fft_code = self.fft_code.replace('NCPUS', str(self.ncpus))

    def _setup(self):
        # Our transposes are executed using all available threads, and always in-place
        #fftplan = fftw_plan(self.size, nthreads = self.ncpus)
        fftplan = _fftw.plan(self.size, complex64, complex64, _fftw.FFTW_BACKWARD,
                             _fftw.get_measure_level(), True, self.ncpus, False)
        self.plan[0] = fftplan
        # Force compilation as part of setup
        self.execute()

    def execute(self):
        st = _np.array(self.st, copy = False)
        ht = _np.array(self.ht, copy = False)
        qt = _np.array(self.qt, copy = False)
        snr = _np.array(self.snr.data, copy = False)
        plan = _np.array(self.plan, copy = False)
        inline(self.code, ['st', 'ht', 'qt', 'snr', 'plan'],
               extra_compile_args=['-fopenmp -march=native -O3 -w'],
               libraries = many_fft_libs, library_dirs = many_fft_libdirs,
               support_code = self.support, extra_link_args = rpath_list,
               verbose = 2, auto_downcast = 1)
#        inline(self.corr_code, ['st', 'ht', 'qt'],
#               extra_compile_args=['-fopenmp -march=native -O3 -w'],
#               libraries = ['gomp'], 
#               support_code = self.support,
#               verbose = 2, auto_downcast = 1)
#        inline(self.fft_code, ['qt', 'snr', 'plan'],
#               extra_compile_args=['-fopenmp -march=native -O3 -w'],
#               libraries = many_fft_libs, library_dirs = many_fft_libdirs,
#               support_code = self.support, extra_link_args = rpath_list,
#               verbose = 2, auto_downcast = 1)
               


_class_dict = { 'many_fft' : ManyFFTProblem
              }

valid_methods = _class_dict.keys()

def parse_problem(probstring, method):
    """
    This function takes a string of the form <number>


    It returns the class and size, so that the call:
        MyClass, n = parse_trans_problem(probstring)
    should usually be followed by:
        MyProblem = MyClass(n)
    """
    prob_class = _class_dict[method]
    size = int(probstring)

    return prob_class, size
