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

from odlemrecon.util import settings_from_domain, make_settings_file

__all__ = ('EMReconForwardProjector', 'EMReconBackProjector',
           'EMReconAttenuationCorrection', 'EMReconScatteringSimulation',
           'EMReconForwardProjectorList', 'EMReconBackProjectorList')


class EMReconForwardProjector(odl.Operator):
    def __init__(self, domain, range, settings=None, settings_file_name=None):
        if settings_file_name is None and settings is None:
            settings = {}
        elif settings_file_name is not None and settings is not None:
            raise ValueError('need either `settings_file_name` or `settings`')
        elif settings is not None:
            settings.update(settings_from_domain(domain))
            settings_file_name = make_settings_file(settings)

        self.settings_file_name = settings_file_name
        self.volume_file = tempfile.NamedTemporaryFile(mode='w+')
        self.sinogram_file = tempfile.NamedTemporaryFile(mode='r')
        odl.Operator.__init__(self, domain, range, linear=True)

    def _call(self, volume):
        # Copy volume to disk
        fortranvolume = np.asfortranarray(volume, dtype='float32')
        fortranvolume = fortranvolume.swapaxes(0, 2)
        self.volume_file.seek(0)
        self.volume_file.write(fortranvolume.tobytes())
        self.volume_file.flush()

        command = 'echo "4" | EMrecon_siemens_pet_tools {} {} {} > /dev/null'.format(
            self.settings_file_name,
            self.volume_file.name,
            self.sinogram_file.name)
        os.system(command)

        sinogram = np.fromfile(self.sinogram_file.name, dtype='float32')
        sinogram = sinogram.reshape(self.range.shape, order='F')

        return sinogram

    @property
    def adjoint(self):
        return EMReconBackProjector(
            self.range, self.domain,
            settings_file_name=self.settings_file_name)


class EMReconBackProjector(odl.Operator):
    def __init__(self, domain, range, settings=None, settings_file_name=None):
        if settings_file_name is None and settings is None:
            settings = {}
        elif settings_file_name is not None and settings is not None:
            raise ValueError('need either `settings_file_name` or `settings`')
        elif settings is not None:
            settings.update(settings_from_domain(domain))
            settings_file_name = make_settings_file(settings)

        self.settings_file_name = settings_file_name
        self.sinogram_file = tempfile.NamedTemporaryFile(mode='w+')
        self.backproj_file = tempfile.NamedTemporaryFile(mode='r')
        odl.Operator.__init__(self, domain, range, linear=True)

    def _call(self, sinogram):
        fortransinogram = np.asfortranarray(sinogram, dtype='float32')
        fortransinogram = fortransinogram.swapaxes(0, 2)
        self.sinogram_file.seek(0)
        self.sinogram_file.write(fortransinogram.tobytes())
        self.sinogram_file.flush()

        command = 'echo "5" | EMrecon_siemens_pet_tools {} {} {} > /dev/null'.format(
            self.settings_file_name,
            self.sinogram_file.name,
            self.backproj_file.name)
        os.system(command)

        backproj = np.fromfile(self.backproj_file.name, dtype='float32')
        backproj = backproj.reshape(self.range.shape, order='F')

        # Scale the adjoint properly
        backproj /= self.range.cell_volume

        return backproj

    @property
    def adjoint(self):
        return EMReconForwardProjector(
            self.range, self.domain,
            settings_file_name=self.settings_file_name)


class EMReconForwardProjectorList(odl.Operator):
    def __init__(self, domain, range, geometry, settings):
        settings.update(settings_from_domain(domain))
        settings_file_name = make_settings_file(settings)

        self.settings = settings
        self.settings_file_name = settings_file_name
        self.geometry = geometry
        self.volume_file = tempfile.NamedTemporaryFile(mode='w+')
        self.sinogram_file = tempfile.NamedTemporaryFile(mode='r')

        # Create reference sinogram file
        npsinogram = np.asarray(range.zero(), dtype='float32')
        sinogram_with_geom = np.hstack((self.geometry, npsinogram[:, None]))
        fortransinogram = np.asfortranarray(sinogram_with_geom,
                                            dtype='float32')
        self.reference_sinogram_file = tempfile.NamedTemporaryFile(mode='w+')
        self.reference_sinogram_file.write(fortransinogram.tobytes())
        self.reference_sinogram_file.flush()

        odl.Operator.__init__(self, domain, range, linear=True)

    def _call(self, volume):
        # Copy volume to disk
        fortranvolume = np.asfortranarray(volume, dtype='float32')
        fortranvolume = fortranvolume.swapaxes(0, 2)
        self.volume_file.seek(0)
        self.volume_file.write(fortranvolume.tobytes())
        self.volume_file.flush()

        command = 'echo "3" | EMrecon_artificial_tools {} {} {} {} > /dev/null'.format(
            self.settings_file_name,
            self.volume_file.name,
            self.reference_sinogram_file.name,
            self.sinogram_file.name)
        os.system(command)

        sinogram = np.fromfile(self.sinogram_file.name, dtype='float32')
        sinogram = sinogram.reshape([self.range.size, 7], order='C')

        return sinogram[:, -1]

    @property
    def adjoint(self):
        return EMReconBackProjectorList(
            self.range, self.domain,
            geometry=self.geometry,
            settings=self.settings)


class EMReconBackProjectorList(odl.Operator):
    def __init__(self, domain, range, geometry, settings):
        settings.update(settings_from_domain(range))
        settings_file_name = make_settings_file(settings)

        self.geometry = geometry
        self.settings_file_name = settings_file_name
        self.settings = settings
        self.sinogram_file = tempfile.NamedTemporaryFile(mode='w+')
        self.backproj_file = tempfile.NamedTemporaryFile(mode='r')
        odl.Operator.__init__(self, domain, range, linear=True)

    def _call(self, sinogram):
        npsinogram = np.asarray(sinogram, dtype='float32')
        sinogram_with_geom = np.hstack((self.geometry, npsinogram[:, None]))
        fortransinogram = np.asfortranarray(sinogram_with_geom,
                                            dtype='float32')
        # fortransinogram = fortransinogram.swapaxes(0, 2)
        self.sinogram_file.seek(0)
        self.sinogram_file.write(fortransinogram.tobytes())
        self.sinogram_file.flush()

        command = 'echo "4" | EMrecon_artificial_tools {} {} {} > /dev/null'.format(
            self.settings_file_name,
            self.sinogram_file.name,
            self.backproj_file.name)
        os.system(command)

        backproj = np.fromfile(self.backproj_file.name, dtype='float32')
        backproj = backproj.reshape(self.range.shape, order='F')

        # Scale the adjoint properly
        backproj /= self.range.cell_volume

        return backproj

    @property
    def adjoint(self):
        return EMReconForwardProjectorList(
            self.range, self.domain,
            geometry=self.geometry,
            settings=self.settings)


class EMReconAttenuationCorrection(odl.Operator):
    """Attenuation as data-to-data mapping.

    Requires ``settings`` to contain a ``'UMAPFILENAME'`` entry.
    """
    def __init__(self, sinogram_space, settings=None, settings_file_name=None):
        if settings_file_name is None and settings is None:
            settings = {}
        elif settings_file_name is not None and settings is not None:
            raise ValueError('need either `settings_file_name` or `settings`')
        elif settings is not None:
            self.umapfile = settings['UMAPFILENAME']
            settings_file_name = make_settings_file(settings)

        self.settings_file_name = settings_file_name
        self.sinogram_in = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        self.sinogram_out = tempfile.NamedTemporaryFile(mode='r', delete=False)
        odl.Operator.__init__(self, domain=sinogram_space,
                              range=sinogram_space, linear=True)

    def _call(self, volume):
        # Copy volume to disk
        fortranvolume = np.asfortranarray(volume, dtype='float32')
        fortranvolume = fortranvolume.swapaxes(0, 2)
        self.sinogram_in.seek(0)
        self.sinogram_in.write(fortranvolume.tobytes())
        self.sinogram_in.flush()

        command = 'echo "3" | EMrecon_siemens_pet_tools {} {} {} {} > /dev/null'.format(
            self.settings_file_name,
            self.umapfile,
            self.sinogram_in.name,
            self.sinogram_out.name)
        os.system(command)

        sinogram = np.fromfile(self.sinogram_out.name, dtype='float32')
        sinogram = sinogram.reshape(self.range.shape, order='F')

        return sinogram


class EMReconScatteringSimulation(odl.Operator):
    def __init__(self, domain, range, sinogram, settings=None,
                 settings_file_name=None):
        if settings_file_name is None and settings is None:
            settings = {}
        elif settings_file_name is not None and settings is not None:
            raise ValueError('need either `settings_file_name` or `settings`')
        elif settings is not None:
            settings.update(settings_from_domain(domain))
            settings_file_name = make_settings_file(settings)
            self.umap_file_name = settings['UMAPFILENAME']

        self.settings_file_name = settings_file_name
        self.volume_file = tempfile.NamedTemporaryFile(mode='w+')
        self.scatter_file = tempfile.NamedTemporaryFile(mode='r')
        self.sinogram_in = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        fortransino = np.asfortranarray(sinogram, dtype='float32')
        fortransino = fortransino.swapaxes(0, 2)
        self.sinogram_in.seek(0)
        self.sinogram_in.write(fortransino.tobytes())
        self.sinogram_in.flush()
        odl.Operator.__init__(self, domain, range, linear=False)

    def _call(self, volume):
        # Copy volume to disk
        fortranvolume = np.asfortranarray(volume, dtype='float32')
        fortranvolume = fortranvolume.swapaxes(0, 2)
        self.volume_file.seek(0)
        self.volume_file.write(fortranvolume.tobytes())
        self.volume_file.flush()

        command = 'echo "7" | EMrecon_siemens_pet_tools {} {} {} {} -1 {} > /dev/null'.format(
            self.settings_file_name,
            self.volume_file.name,
            self.umap_file_name,
            self.sinogram_in.name,
            self.scatter_file.name)
        os.system(command)

        scatter = np.fromfile(self.scatter_file.name, dtype='float32')
        scatter = scatter.reshape(self.range.shape, order='F')

        return scatter


if __name__ == '__main__':
    import odl
    import odlemrecon

    fov = np.array([590.625, 590.625, 158.625])
    shape = [175, 175, 47]

    settings = {'SCANNERTYPE': 3}
    ran_shape = [192, 192, 175]  # Given by scanner

    space = odl.uniform_discr(-fov/2, fov/2, shape)
    ran = odl.uniform_discr([0]*3, ran_shape, ran_shape)

    op = odlemrecon.EMReconForwardProjector(space, ran, settings=settings)

    phantom = odl.phantom.shepp_logan(space, modified=True)
    phantom.show('phantom')

    projection = op(phantom)
    projection.show('projection')

    backprojection = op.adjoint(projection)
    backprojection.show('adjoint')
