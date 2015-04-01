#!/usr/bin/env python

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

from pycbc.types import Array, zeros, complex64
from pycbc.filter.matchedfilter_cpu import correlate_numpy, correlate_parallel
from numpy.random import random
from pycbc import scheme
import argparse

parser = argparse.ArgumentParser(description = "Test complex correlation")

scheme.insert_processing_option_group(parser)

# And parse
opt = parser.parse_args()

# Check that the values returned for the options make sense
scheme.verify_processing_options(opt, parser)

# Do what we can with command line options
ctx = scheme.from_cli(opt)

with ctx:
    in1 = zeros(1024 * 1024, dtype = complex64)
    in2 = zeros(1024 * 1024, dtype = complex64)
    in1.data[:1024*512] = random(1024*512) + 1j * random(1024*512)
    in2.data[:1024*512] = random(1024*512) + 1j * random(1024*512)

    out_np = zeros(1024 * 1024, dtype = complex64)
    out_parallel = zeros(1024 * 1024, dtype = complex64)

    print "Running with aligned input"
    correlate_numpy(in1, in2, out_np)
    correlate_parallel(in1, in2, out_parallel)
    print "Results: {0}".format(out_np == out_parallel)

    a = zeros(1024 * 1024 + 1, dtype = complex64)
    b = zeros(1024 * 1024 + 1, dtype = complex64)
    c = zeros(1024 * 1024 + 1, dtype = complex64)
    d = zeros(1024 * 1024 + 1, dtype = complex64)
    in1 = Array(a[1:], copy = False)
    in2 = Array(b[1:], copy = False)
    out_np = Array(c[1:], copy = False)
    out_parallel = Array(d[1:], copy = False)
    in1.data[:1024*512] = random(1024*512) + 1j * random(1024*512)
    in2.data[:1024*512] = random(1024*512) + 1j * random(1024*512)

    print "Running with mis-aligned input"
    correlate_numpy(in1, in2, out_np)
    correlate_parallel(in1, in2, out_parallel)
    print "Results: {0}".format(out_np == out_parallel)

