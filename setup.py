#!/usr/bin/env /usr/bin/python
#
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

from distutils.core import setup
from distutils.command.install import install as _install
from distutils.file_util import write_file
import os

required_list = ['multibench', 'cProfile', 'pycbc']

class install(_install):
    def run(self):
        # Check for some of the required python packages
        for ppkg in required_list:
            try:
                __import__(ppkg)
            except:
                raise RuntimeError("Failed to locate required package: {0}".format(ppkg))

        etcdirectory = os.path.join(self.install_data, 'etc')
        if not os.path.exists(etcdirectory):
            os.makedirs(etcdirectory)

        filename = os.path.join(etcdirectory, 'mf-timing-user-env.sh')
        self.execute(write_file,
                     (filename, [self.extra_dirs]),
                     "creating {0}".format(filename))

        env_file = open(filename, 'w')
        print >> env_file, "# Source this file to access corr_timing"
        print >> env_file, "PATH=" + self.install_scripts + ":$PATH"
        print >> env_file, "PYTHONPATH=" + self.install_libbase + ":$PYTHONPATH"
        print >> env_file, "export PYTHONPATH"
        print >> env_file, "export PATH"
        env_file.close()
        # We need to call the parent, and must do it this way because
        # apparently distutils doesn't derive its classes from 'object'
        _install.run(self)


setup (
    name = 'mf',
    version = '0.1',
    description = 'Tool to benchmark various PyCBC approaches to matched filtering',
    author = 'Joshua L. Willis',
    author_email = 'joshua.willis@ligo.org',
    requires = required_list,
    cmdclass = { "install" : install },
    packages = [
        'mf'
        ],
    scripts = ['bin/corr_timing',
               'bin/test_corr.py',
               'bin/plain_fftw_timing',
               'bin/many_fftw_timing',
               'bin/execute_fixed_plain',
               'bin/thresh_timing',
               'bin/pycbc_timing',
               'bin/pycbc_dummy',
               'bin/print_fftw_libs'
        ]
    )
