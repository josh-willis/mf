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
from pycbc.types import zeros, complex64, float32, Array
from pycbc import scheme as _scheme
from pycbc.filter.matchedfilter import Correlator
from pycbc.events import ThresholdCluster
from pycbc.filter.matchedfilter_cpu import correlate_parallel, correlate_inline
from pycbc.events.threshold_cpu import threshold_inline
import numpy as _np
from numpy.random import random
import ctypes
from pycbc import libutils
import pycbc.fft
from pycbc.fft import IFFT, ifft
import pycbc.fft.fftw as _fftw


"""
This module benchmarks old and new versions of various pycbc matched
filtering components.
"""

# Some things to set up our FFT
libfftw3f = _fftw.float_lib

fexecute = libfftw3f.fftwf_execute_dft
fexecute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

class IFFT(object):
    def __init__(self, invec, outvec):
        self.obj = pycbc.fft.IFFT(invec, outvec)
        self.execute = self.obj.execute

    def _setup(self):
        self.execute()

class BaseProblem(_mb.MultiBenchProblem):
    def __init__(self, size):
        # We'll do some arithmetic with these, so sanity check first:
        if (size < 1):
            raise ValueError("size must be >= 1")
        self.size = size
        # Assume 256 sec segment
        self.srate = size/256

        self.ht = zeros(self.size, dtype=complex64)
        self.st = zeros(self.size, dtype=complex64)
        self.qt = zeros(self.size, dtype=complex64)
        self.snr = zeros(self.size, dtype=complex64)
        self.ht.data[:size/2] = random(size/2) + 1j * random(size/2)
        self.st.data[:size/2] = random(size/2) + 1j * random(size/2)
        # The following represents a 30 Hz, 256 segment configuration
        self.analyze = slice(112 * self.srate, (256 - 16) * self.srate)

class CorrOld(BaseProblem):
    def _setup(self):
        self.execute()
    def execute(self):
        correlate_inline(self.ht[:self.size/2], self.st[:self.size/2], self.qt[:self.size/2])

class CorrNew(BaseProblem):
    def __init__(self, size):
        super(CorrNew, self).__init__(size)
        self.ht_np = _np.array(self.ht.data[:self.size/2], copy = False)
        self.st_np = _np.array(self.st.data[:self.size/2], copy = False)
        self.qt_np = _np.array(self.qt.data[:self.size/2], copy = False)
        self.correlator = Correlator(self.ht_np, self.st_np, self.qt_np)
        self.execute = self.correlator.correlate

    def _setup(self):
        self.execute()

class FFTOld(BaseProblem):
    def __init__(self, size):
        super(FFTOld, self).__init__(size)

    def _setup(self):
        self.execute()

    def execute(self):
        ifft(self.qt, self.snr)

class FFTNew(BaseProblem):
    def __init__(self, size):
        super(FFTNew, self).__init__(size)
        self.fftobj = IFFT(self.qt, self.snr)
        self.execute = self.fftobj.execute
        self._setup = self.fftobj._setup

class OldCorrFFT(BaseProblem):
    def __init__(self, size):
        super(OldCorrFFT, self).__init__(size)

    def _setup(self):
        self.execute()

    def execute(self):
        correlate_inline(self.ht[:self.size/2], self.st[:self.size/2], self.qt[:self.size/2])
        ifft(self.qt, self.snr)

class NewCorrFFT(BaseProblem):
    def __init__(self, size):
        super(NewCorrFFT, self).__init__(size)
        self.ht_np = _np.array(self.ht.data[:self.size/2], copy = False)
        self.st_np = _np.array(self.st.data[:self.size/2], copy = False)
        self.qt_np = _np.array(self.qt.data[:self.size/2], copy = False)
        self.correlator = Correlator(self.ht_np, self.st_np, self.qt_np)
        self.fftobj = IFFT(self.qt, self.snr)

    def _setup(self):
        self.fftobj._setup()
        self.execute()

    def execute(self):
        self.correlator.correlate()
        self.fftobj.execute()

global_winsize = 4096

def get_one_thresh(inarr):
    tmparr = abs(inarr)
    mval, mloc = tmparr.max_loc()
    tmparr[mloc] = 0
    m2 = max(tmparr)
    del tmparr
    return 0.5*(m2+mval)

class OldThresh(BaseProblem):
    def __init__(self, size):
        super(OldThresh, self).__init__(size)
        self.fftobj = IFFT(self.qt, self.snr)

    def _setup(self):
        self.fftobj._setup()
        correlate_parallel(self.ht[:self.size/2], self.st[:self.size/2], self.qt[:self.size/2])
        self.fftobj.execute()
        self.threshold = get_one_thresh(self.snr)
        self.execute()

    def execute(self):
        threshold_inline(self.snr[self.analyze], self.threshold)

class NewThresh(BaseProblem):
    def __init__(self, size):
        super(NewThresh, self).__init__(size)
        self.ht_np = _np.array(self.ht.data[:self.size/2], copy = False)
        self.st_np = _np.array(self.st.data[:self.size/2], copy = False)
        self.qt_np = _np.array(self.qt.data[:self.size/2], copy = False)
        self.snr_np = _np.array(self.snr.data[self.analyze], copy = False)
        self.correlator = Correlator(self.ht_np, self.st_np, self.qt_np)
        self.fftobj = IFFT(self.qt, self.snr)
        self.threshold_and_clusterer = ThresholdCluster(self.snr_np, global_winsize)

    def _setup(self):
        self.fftobj._setup()
        self.correlator.correlate()
        self.fftobj.execute()
        self.threshold = get_one_thresh(self.snr)
        self.execute()

    def execute(self):
        self.threshold_and_clusterer.threshold_and_cluster(self.threshold)

class NewNewThresh(BaseProblem):
    def __init__(self, size):
        super(NewNewThresh, self).__init__(size)
        self.ht_np = _np.array(self.ht.data[:self.size/2], copy = False)
        self.st_np = _np.array(self.st.data[:self.size/2], copy = False)
        self.qt_np = _np.array(self.qt.data[:self.size/2], copy = False)
        self.snr_np = _np.array(self.snr.data[self.analyze], copy = False)
        self.correlator = Correlator(self.ht_np, self.st_np, self.qt_np)
        self.fftobj = IFFT(self.qt, self.snr)
        self.threshold_and_clusterer = ThresholdCluster(self.snr_np)

    def _setup(self):
        self.fftobj._setup()
        self.correlator.correlate()
        self.fftobj.execute()
        self.threshold = get_one_thresh(self.snr)
        self.execute()

    def execute(self):
        self.threshold_and_clusterer.threshold_and_cluster(self.threshold, global_winsize)

class OldFFTThresh(BaseProblem):
    def __init__(self, size):
        super(OldFFTThresh, self).__init__(size)

    def _setup(self):
        correlate_parallel(self.ht[:self.size/2], self.st[:self.size/2], self.qt[:self.size/2])
        ifft(self.qt, self.snr)
        self.threshold = get_one_thresh(self.snr)
        self.execute()

    def execute(self):
        ifft(self.qt, self.snr)
        threshold_inline(self.snr[self.analyze], self.threshold)

class NewFFTThresh(BaseProblem):
    def __init__(self, size):
        super(NewFFTThresh, self).__init__(size)
        self.ht_np = _np.array(self.ht.data[:self.size/2], copy = False)
        self.st_np = _np.array(self.st.data[:self.size/2], copy = False)
        self.qt_np = _np.array(self.qt.data[:self.size/2], copy = False)
        self.snr_np = _np.array(self.snr.data[self.analyze], copy = False)
        self.correlator = Correlator(self.ht_np, self.st_np, self.qt_np)
        self.fftobj = IFFT(self.qt, self.snr)
        self.threshold_and_clusterer = ThresholdCluster(self.snr_np, global_winsize)

    def _setup(self):
        self.fftobj._setup()
        self.correlator.correlate()
        self.fftobj.execute()
        self.threshold = get_one_thresh(self.snr)
        self.execute()

    def execute(self):
        self.fftobj.execute()
        self.threshold_and_clusterer.threshold_and_cluster(self.threshold)

class OldAll(BaseProblem):
    def __init__(self, size):
        super(OldAll, self).__init__(size)

    def _setup(self):
        correlate_inline(self.ht[:self.size/2], self.st[:self.size/2], self.qt[:self.size/2])
        ifft(self.qt, self.snr)
        self.threshold = get_one_thresh(self.snr)
        self.execute()

    def execute(self):
        correlate_inline(self.ht[:self.size/2], self.st[:self.size/2], self.qt[:self.size/2]) 
        ifft(self.qt, self.snr)
        threshold_inline(self.snr[self.analyze], self.threshold)

class NewAll(BaseProblem):
    def __init__(self, size):
        super(NewAll, self).__init__(size)
        self.ht_np = _np.array(self.ht.data[:self.size/2], copy = False)
        self.st_np = _np.array(self.st.data[:self.size/2], copy = False)
        self.qt_np = _np.array(self.qt.data[:self.size/2], copy = False)
        self.snr_np = _np.array(self.snr.data[self.analyze], copy = False)
        self.correlator = Correlator(self.ht_np, self.st_np, self.qt_np)
        self.fftobj = IFFT(self.qt, self.snr)
        self.threshold_and_clusterer = ThresholdCluster(self.snr_np, global_winsize)

    def _setup(self):
        self.fftobj._setup()
        self.correlator.correlate()
        self.fftobj.execute()
        self.threshold = get_one_thresh(self.snr)
        self.execute()

    def execute(self):
        self.correlator.correlate()
        self.fftobj.execute()
        self.threshold_and_clusterer.threshold_and_cluster(self.threshold)

class BaseManyProblem(_mb.MultiBenchProblem):
    def __init__(self, size):
        # We'll do some arithmetic with these, so sanity check first:
        if (size < 1):
            raise ValueError("size must be >= 1")
        self.size = size
        # Assume 256 sec segment, 15 segments
        self.srate = size/256
        self.nsegs = 15

        self.ht = zeros(self.size, dtype=complex64)
        self.qt = zeros(self.size, dtype=complex64)
        self.snr = zeros(self.size, dtype=complex64)
        self.ht.data[:size/2] = random(size/2) + 1j * random(size/2)
        self.stlist = []
        for i in range(0, self.nsegs):
            st = zeros(self.size, dtype=complex64)
            st.data[:size/2] = random(size/2) + 1j * random(size/2)
            self.stlist.append(st)
        # The following represents a 30 Hz, 256 segment configuration
        self.analyze = slice(112 * self.srate, (256 - 16) * self.srate)

class OldAllMany(BaseManyProblem):
    def __init__(self, size):
        super(OldAllMany, self).__init__(size)

    def _setup(self):
        correlate_inline(self.ht[:self.size/2], self.stlist[0][:self.size/2], self.qt[:self.size/2])
        ifft(self.qt, self.snr)
        self.threshold = get_one_thresh(self.snr)
        self.execute()

    def execute(self):
        for i in range(0, self.nsegs):
            correlate_inline(self.ht[:self.size/2], self.stlist[i][:self.size/2], self.qt[:self.size/2])
            ifft(self.qt, self.snr)
            threshold_inline(self.snr[self.analyze], self.threshold)

class NewAllMany(BaseManyProblem):
    def __init__(self, size):
        super(NewAllMany, self).__init__(size)
        self.ht_np = _np.array(self.ht.data[:self.size/2], copy = False)
        self.st_np_list = []
        for i in range(0, self.nsegs):
            self.st_np_list.append( _np.array( self.stlist[i].data[:self.size/2], copy = False ))
        self.qt_np = _np.array(self.qt.data[:self.size/2], copy = False)
        self.snr_np = _np.array(self.snr.data[self.analyze], copy = False)
        self.correlators = []
        for i in range(0, self.nsegs):
            self.correlators.append(Correlator(self.ht_np, self.st_np_list[i], self.qt_np))
        self.fftobj = IFFT(self.qt, self.snr)
        self.threshold_and_clusterer = ThresholdCluster(self.snr_np, global_winsize)

    def _setup(self):
        self.fftobj._setup()
        self.correlators[0].correlate()
        self.fftobj.execute()
        self.threshold = get_one_thresh(self.snr)
        self.execute()

    def execute(self):
        for i in range(0, self.nsegs):
            self.correlators[i].correlate()
            self.fftobj.execute()
            self.threshold_and_clusterer.threshold_and_cluster(self.threshold)


_class_dict = { 'corr_old' : CorrOld,
                'corr_new' : CorrNew,
                'fft_old' : FFTOld,
                'fft_new' : FFTNew,
                'corr_fft_old' : OldCorrFFT,
                'corr_fft_new' : NewCorrFFT,
                'thresh_old' : OldThresh,
                'thresh_new' : NewThresh,
                'thresh_new_new' : NewNewThresh,
                'fft_thresh_old' : OldFFTThresh,
                'fft_thresh_new' : NewFFTThresh,
                'all_old' : OldAll,
                'all_new' : NewAll,
                'all_old_many' : OldAllMany,
                'all_new_many' : NewAllMany
                }

valid_methods = _class_dict.keys()

def parse_problem(probstring, method='corr_old'):
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
