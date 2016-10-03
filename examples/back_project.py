import odl
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt

shape = [175, 175, 47]

settings = """SCANNERTYPE=3
SIZE_X={shape[0]}
SIZE_Y={shape[1]}
SIZE_Z={shape[2]}
""".format(shape=shape)

settings_file = tempfile.NamedTemporaryFile(delete=False)
settings_file.write(settings)
settings_file_name = settings_file.name
settings_file.close()

for line in open(settings_file_name):
    print(line)

space = odl.uniform_discr([-1]*3, [1]*3, shape[::-1], dtype='float32')
volume = odl.phantom.shepp_logan(space, modified=True).asarray()

volume_file = tempfile.NamedTemporaryFile(delete=False)
volume_file.write(np.asfortranarray(volume).tobytes())
volume_file_name = volume_file.name
volume_file.close()

# volume_file_name = '/media/windows-share/emrecon/example3_forward_project/reconstruction.raw'

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
    sinogram = sinogram.reshape([192, 192, 175], order='F')

plt.figure()
plt.imshow(sinogram[:, :, 24])
plt.figure()
plt.imshow(sinogram[100, :, :])

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
    backproj = backproj.reshape(shape, order='F').swapaxes(0, 2)

plt.figure()
plt.imshow(backproj[23, :, :])
plt.colorbar()

plt.figure()
plt.imshow(volume[23, :, :])
plt.colorbar()

plt.figure()
plt.imshow(backproj[:, :, 80])
plt.colorbar()

plt.figure()
plt.imshow(volume[:, :, 80])
plt.colorbar()