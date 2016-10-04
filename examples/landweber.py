import odl
import os
import tempfile
import numpy as np


def make_settings_file(settings):
    settings_str = '\n'.join('{}={}'.format(key, settings[key])
                             for key in settings)

    with tempfile.NamedTemporaryFile(delete=False) as settings_file:
        settings_file.write(settings_str)
        settings_file_name = settings_file.name

    return settings_file_name


class MyOperator(odl.Operator):
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
        return MyOperatorAdjoint(self.settings_file_name,
                                 self.range, self.domain)


class MyOperatorAdjoint(odl.Operator):
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
        return MyOperator(self.settings_file_name, self.range, self.domain)

shape = [175, 175, 47]
ran_shape = [192, 192, 175]

settings = {'SCANNERTYPE':3,
            'SIZE_X':shape[0],
            'SIZE_Y':shape[1],
            'SIZE_Z':shape[2],
            'VERBOSE':0}
settings_file_name = make_settings_file(settings)

space = odl.uniform_discr([0]*3, shape, shape)
ran = odl.uniform_discr([0]*3, ran_shape, ran_shape)

op = MyOperator(settings_file_name, space, ran)

phantom = odl.phantom.shepp_logan(space, modified=True)
phantom.show()

projection = op(phantom)
projection.show()

opnorm = projection.norm() / phantom.norm()
omega = 0.8/opnorm**2

callback = odl.solvers.CallbackShow()

x = space.zero()
odl.solvers.landweber(op, x, projection, omega=omega, niter=100,
                      callback=callback)
