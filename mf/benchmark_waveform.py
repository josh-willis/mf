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
import numpy as _np
from pycbc.types import zeros, complex64
from pycbc.waveform import get_waveform_end_frequency as _gwef
from pycbc.waveform import get_waveform_filter as _gwf
from pycbc import DYN_RANGE_FAC as _DRF


"""
This module benchmarks old and new versions of various pycbc waveforms.
"""

class BaseWaveformProblem(_mb.MultiBenchProblem):
    def __init__(self, srate, seglen):
        self.seglen = float(seglen)
        self.srate = float(srate)
        self.size = int(self.seglen * self.srate)
        self.flen = self.size/2+1
        self.wf_mem = zeros(self.flen, dtype=complex64)
        self.dist = 1.0/_DRF
        self.df = self.srate/self.size
        self.dt = 1.0/self.srate
        # For the moment, we will hardcode this, but we should do it
        # in a more general way sometime
        self.f_lower = 30.0
        self.end_freq = _gwef(approximant=self.approximant, **self.params)
        
    def _setup(self):
        # Once through to handle any one-time setup
        self.execute()

    def execute(self):
        # The following all occur inside of __getitem__ within
        # pycbc.waveform.FilterBank
        poke = self.wf_mem.data
        self.wf_mem.clear()
        _gwf(self.wf_mem, approximant=self.approximant, f_lower=self.f_lower,
             delta_f = self.df, delta_t = self.dt, distance=self.dist, **self.params)

class SPAProblem(BaseWaveformProblem):
    def __init__(self, srate, seglen):
        self.approximant = 'SPAtmplt'
        self.params = { 'mass1' : 1.40,
                        'mass2' : 1.40,
                        'spin1z' : 0.0,
                        'spin2z' : 0.0 }
        super(SPAProblem, self).__init__(srate, seglen)

class SEOBNR_ROM_DS_Problem(BaseWaveformProblem):
    def __init__(self, srate, seglen):
        self.approximant = 'SEOBNRv2_ROM_DoubleSpin'
        self.params = { 'mass1' : 3.0,
                        'mass2' : 3.5,
                        'spin1z' : 0.15,
                        'spin2z' : 0.1 }
        super(SEOBNR_ROM_DS_Problem, self).__init__(srate, seglen)


_class_dict = { 'spa' : SPAProblem,
                'seobnrv2_rom' : SEOBNR_ROM_DS_Problem
                }

valid_waveforms = _class_dict.keys()

def parse_problem(waveformstr='spa'):
    """
    This function takes a string of the form <waveform>.
    
    It returns the class, so that the call:
        MyProblem  = parse_problem(wf)
    should usually be followed by:
        MyProblem = MyClass(srate, seglen)
    """
    return _class_dict[method]
