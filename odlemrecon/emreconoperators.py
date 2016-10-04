# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Operators for EMRecon - ODL bindings."""


import tempfile
import odl
import numpy as np
import os


__all__ = ('EMReconForwardProjector', 'EMReconBackProjector')


class EMReconForwardProjector(odl.Operator):
    def __init__(self, settings_file_name, domain, range,
                 volume_file_name=None, sinogram_file_name=None):
        self.settings_file_name = settings_file_name
        odl.Operator.__init__(self, domain, range, linear=True)

    def _call(self, volume):
        with tempfile.NamedTemporaryFile(mode='w+') as volume_file, \
             tempfile.NamedTemporaryFile(mode='r') as sinogram_file:

            # Copy volume to disk
            fortranvolume = np.asfortranarray(volume, dtype='float32')
            fortranvolume = fortranvolume.swapaxes(0, 2)
            volume_file.write(fortranvolume.tobytes())
            volume_file_name = volume_file.name
            volume_file.flush()

            # Get sinogram "path"
            sinogram_file_name = sinogram_file.name

            command = 'echo "4" | EMrecon_siemens_pet_tools {} {} {}'.format(
                self.settings_file_name, volume_file_name, sinogram_file_name)
            os.system(command)

            sinogram = np.fromfile(sinogram_file_name, dtype='float32')
            sinogram = sinogram.reshape(self.range.shape, order='F')

        return sinogram

    @property
    def adjoint(self):
        return EMReconBackProjector(self.settings_file_name,
                                    self.range, self.domain)


class EMReconBackProjector(odl.Operator):
    def __init__(self, settings_file_name, domain, range):
        self.settings_file_name = settings_file_name
        odl.Operator.__init__(self, domain, range, linear=True)

    def _call(self, sinogram):
        with tempfile.NamedTemporaryFile(mode='w+') as sinogram_file, \
             tempfile.NamedTemporaryFile(mode='r') as backproj_file:
            fortransinogram = np.asfortranarray(sinogram, dtype='float32')
            fortransinogram = fortransinogram.swapaxes(0, 2)
            sinogram_file.write(fortransinogram.tobytes())
            sinogram_file_name = sinogram_file.name

            command = 'echo "5" | EMrecon_siemens_pet_tools {} {} {}'.format(
                self.settings_file_name, sinogram_file_name,
                backproj_file.name)
            os.system(command)

            backproj = np.fromfile(backproj_file.name, dtype='float32')
            backproj = backproj.reshape(self.range.shape, order='F')

        return backproj

    @property
    def adjoint(self):
        return EMReconForwardProjector(self.settings_file_name,
                                       self.range, self.domain)

if __name__ == '__main__':
    import odl
    import odlemrecon

    shape = [175, 175, 47]
    ran_shape = [192, 192, 175]

    settings = {'SCANNERTYPE': 3,
                'SIZE_X': shape[0],
                'SIZE_Y': shape[1],
                'SIZE_Z': shape[2],
                'VERBOSE': 0}
    settings_file_name = odlemrecon.make_settings_file(settings)

    space = odl.uniform_discr([0]*3, shape, shape)
    ran = odl.uniform_discr([0]*3, ran_shape, ran_shape)

    op = odlemrecon.EMReconForwardProjector(settings_file_name, space, ran)

    phantom = odl.phantom.shepp_logan(space, modified=True)
    phantom.show('phantom')

    projection = op(phantom)
    projection.show('forward')

    backprojection = op.adjoint(projection)
    backprojection.show('adjoint')
