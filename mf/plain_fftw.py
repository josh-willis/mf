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
from pycbc import scheme as _scheme
import pycbc.fft.fftw as _fftw
from scipy.weave import inline
import numpy as _np
import ctypes
from pycbc import libutils
from math import sqrt
import sys

from .corr import CorrContiguousNoStreaming, CorrContiguousStreamIn

# Several of the OpenMP based approaches use this
max_chunk = 8192

libfftw3f = _fftw.float_lib

fexecute = libfftw3f.fftwf_execute_dft
fexecute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

def check_pow_two(n):
    return ( (n != 0) and ( (n & (n-1)) == 0) )

class BasePlainFFTWProblem(_mb.MultiBenchProblem):
    def __init__(self, size):
        # We'll do some arithmetic with these, so sanity check first:
        if not check_pow_two(size):
            raise ValueError("Only power-of-two sizes supported")

        self.ncpus = _scheme.mgr.state.num_threads
        self.size = size
        self.stilde = zeros(self.size, dtype = complex64)
        self.htilde = zeros(self.size, dtype = complex64)
        self.qtilde = zeros(self.size, dtype = complex64)
        self.snr = zeros(self.size, dtype = complex64)
        self.iptr = self.qtilde.ptr
        self.optr = self.snr.ptr
        self.in1 = self.stilde.data
        self.in2 = self.htilde.data
        self.out = self.qtilde.data

    def _setup(self):
        self.fftplan = _fftw.plan(len(self.snr), self.qtilde.dtype, self.snr.dtype,
                                  _fftw.FFTW_BACKWARD, _fftw.get_measure_level(),
                                  True, self.ncpus, False)
        self.fftfunc = _fftw.execute_function[str(self.qtilde.dtype)][str(self.snr.dtype)]
        self.fftfunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.execute()

class FFTOnly(BasePlainFFTWProblem):
    def __init__(self, size):
        super(FFTOnly, self).__init__(size)

    def execute(self):
        self.fftfunc(self.fftplan, self.iptr, self.optr)

class NoStreamFFT(BasePlainFFTWProblem):
    def __init__(self, size):
        super(NoStreamFFT, self).__init__(size)
        self.ncorr = self.size/2
        self.corrobj = CorrContiguousNoStreaming(self.in1, self.in2, self.out, self.ncorr)

    def execute(self):
        self.corrobj.execute()
        self.fftfunc(self.fftplan, self.iptr, self.optr)

class CorrNoStream(BasePlainFFTWProblem):
    def __init__(self, size):
        super(CorrNoStream, self).__init__(size)
        self.ncorr = self.size/2
        self.corrobj = CorrContiguousNoStreaming(self.in1, self.in2, self.out, self.ncorr)

    def execute(self):
        self.corrobj.execute()

class StreamInFFT(BasePlainFFTWProblem):
    def __init__(self, size):
        super(StreamInFFT, self).__init__(size)
        self.ncorr = self.size/2
        self.corrobj = CorrContiguousStreamIn(self.in1, self.in2, self.out, self.ncorr)

    def execute(self):
        self.corrobj.execute()
        self.fftfunc(self.fftplan, self.iptr, self.optr)

class CorrStreamIn(BasePlainFFTWProblem):
    def __init__(self, size):
        super(CorrStreamIn, self).__init__(size)
        self.ncorr = self.size/2
        self.corrobj = CorrContiguousStreamIn(self.in1, self.in2, self.out, self.ncorr)

    def execute(self):
        self.corrobj.execute()


_class_dict = { 'fft' : FFTOnly,
                'nostream' : NoStreamFFT,
                'streamin' : StreamInFFT,
                'corr_ns' : CorrNoStream,
                'corr_si' : CorrStreamIn
               }

valid_methods = _class_dict.keys()

def parse_problem(probstring, method):
    """
    This function takes a string of the form <number>


    It returns the class and size, so that the call:
        MyClass, n = parse_problem(probstring)
    should usually be followed by:
        MyProblem = MyClass(n)
    """
    prob_class = _class_dict[method]
    size = int(probstring)

    return prob_class, size
