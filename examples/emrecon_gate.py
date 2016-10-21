"""Example of PET tomography problem using list mode data from GATE.

This example requires that the user has access to a .l file output from gate,
the file paths here are examples.

Note that this example is relatively slow given a large number of lines.
"""

import odl
import odlemrecon
import numpy as np
import os

# NOTE: These folders need to be updated to local paths.
folder = '/media/windows-share/emrecon_gate_list_mode_chest'
filen = 'PulmPET_Lesions_20160826_Phantom1_BedPos2.l'

# Load data as a list of float32, then reshape into a n x 7 array.
# The array is given as [px_1, py_1, pz_1, px_2, py_2, pz_2, val]
# where px etc give the points of incidence with the detector, and val is the
# value along the line (usualy 1.0).
data = np.fromfile(os.path.join(folder, filen), dtype='float32')
data = data.reshape([-1, 7])
geometry = data[:, 0:6]
proj_data = data[:, -1]

# Specify the volume geometry
fov = np.array([800., 800., 300.])
shape = np.array([100, 100, 50])
space = odl.uniform_discr(-fov/2, fov/2, shape)

# The range for list mode data is simply a list of real numbers, hence rn.
ran = odl.rn(proj_data.size, dtype='float32')

# SCANNERTYPE 1 means list mode projector, see EMrecon doc
settings = {'SCANNERTYPE': 1}

# Create projector
op = odlemrecon.EMReconForwardProjectorList(space, ran, geometry,
                                            settings=settings)

# Solve the problem using the MLEM method. Note that the regular MLEM method
# does not apply to the case of list mode data, but that an adequate
# approximation is given by ignoring the sensitivities (setting them to 1).
x = op.domain.one()
odl.solvers.mlem(op, x, proj_data, niter=100,
                 callback=odl.solvers.CallbackShow(cmap='hot'),
                 sensitivities=1.0)
