import odl
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt

shape = [175, 175, 47]
ran_shape = [192, 192, 175]

settings = """SCANNERTYPE=3
SIZE_X={shape[0]}
SIZE_Y={shape[1]}
SIZE_Z={shape[2]}
VERBOSE=0
""".format(shape=shape)

settings_file = tempfile.NamedTemporaryFile(delete=False)
settings_file.write(settings)
settings_file_name = settings_file.name
settings_file.close()

class MyOperator(odl.Operator):
    def __init__(self, domain, range):
        odl.Operator.__init__(self, domain, range, linear=True)

    def _call(self, volume):
        volume_file = tempfile.NamedTemporaryFile(delete=False)
        volume_file.write(np.asfortranarray(volume, dtype='float32').tobytes())
        volume_file_name = volume_file.name
        volume_file.close()

        sinogram_file = tempfile.NamedTemporaryFile(delete=False)
        sinogram_file_name = sinogram_file.name
        sinogram_file.close()

        command = 'echo "4" | EMrecon_siemens_pet_tools {} {} {}'.format(
            settings_file_name, volume_file_name, sinogram_file_name)

        print('FORWARD, CALLING COMMAND:')
        print(command)
        os.system(command)

        with open(sinogram_file_name) as sinogram_file:
            sinogram = np.fromfile(sinogram_file, dtype='float32')
            sinogram = sinogram.reshape(self.range.shape[::-1], order='F')
            sinogram = sinogram.swapaxes(0, 2)

        return sinogram

    @property
    def adjoint(self):
        return MyOperatorAdjoint(self.range, self.domain)


class MyOperatorAdjoint(odl.Operator):
    def __init__(self, domain, range):
        odl.Operator.__init__(self, domain, range, linear=True)

    def _call(self, sinogram):
        sinogram_file = tempfile.NamedTemporaryFile(delete=False)
        sinogram_file.write(np.asfortranarray(sinogram).tobytes())
        sinogram_file_name = sinogram_file.name
        sinogram_file.close()

        backproj_file = tempfile.NamedTemporaryFile(delete=False)
        backproj_file_name = backproj_file.name
        backproj_file.close()

        command = 'echo "5" | EMrecon_siemens_pet_tools {} {} {}'.format(
            settings_file_name, sinogram_file_name, backproj_file_name)

        print('BACKWARD, CALLING COMMAND:')
        print(command)
        os.system(command)

        with open(backproj_file_name) as backproj_file:
            backproj = np.fromfile(backproj_file, dtype='float32')
            backproj = backproj.reshape(self.range.shape[::-1], order='F')
            backproj = backproj.swapaxes(0, 2)

        return backproj

    @property
    def adjoint(self):
        return MyOperator(self.range, self.domain)

space = odl.uniform_discr([0]*3, shape[::-1], shape[::-1], dtype='float32')
ran = odl.uniform_discr([0]*3, ran_shape[::-1], ran_shape[::-1], dtype='float32')

op = MyOperator(space, ran)

phantom = odl.phantom.shepp_logan(space, modified=True)
phantom.show()

projection = op(phantom)

opnorm = projection.norm() / phantom.norm()
omega = 0.5/opnorm**2

callback = odl.solvers.CallbackShow()

x = space.zero()
odl.solvers.landweber(op, x, projection, omega=omega, niter=100, callback=callback)
