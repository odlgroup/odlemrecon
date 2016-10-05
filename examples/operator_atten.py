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

# %% Set up problem
"""Example of PET operators."""

import odl
import odlemrecon
import numpy as np

fov = np.array([590.625, 590.625, 158.625])
shape = [175, 175, 47]
ran_shape = [192, 192, 175]

#SCANNERTYPE=3
#SENSITIVITYFILENAME=sens16.sens
#DATAFILENAME=prompts.s
#OUTPUTFILENAME=reconstruction
#RANDOMDATAFILENAME=delayed.s
#UMAPFILENAME=mumap.v
#RECONTYPE=3

settings = {'SCANNERTYPE': 3,
            'VERBOSE': 0,
            'UMAPFILENAME': 'data/mumap.v'}

space = odl.uniform_discr(-fov/2, fov/2, shape)
ran = odl.uniform_discr([0]*3, ran_shape, ran_shape)

pet_op = odlemrecon.EMReconForwardProjector(space, ran, settings=settings)

activity_phantom = np.fromfile('data/reconstruction_ac.raw', dtype='float32')
activity_phantom = space.element(activity_phantom.reshape(space.shape,
                                                          order='F'))

# Attenuation volume
attenuation = np.fromfile(settings['UMAPFILENAME'], dtype='float32')
attenuation = space.element(attenuation.reshape(space.shape, order='F'))
attenuation.show('Attenuation map')

atten_proj = pet_op(attenuation)
atten_corr_factor = np.exp(-atten_proj / 10)  # MAGIC!!
atten_corr_pet_op = atten_corr_factor * pet_op

# Sensitivity map, A^*1
#sensitivity_map = atten_corr_pet_op.adjoint(atten_corr_pet_op.range.one())

# Real data
data = np.fromfile('data/prompts.s', dtype='float32')
data = pet_op.range.element(data.reshape(ran_shape, order='F'))

# Scatter
scatter_op = odlemrecon.EMReconScatteringSimulation(
    space, ran, data, settings=settings)
scatter = scatter_op(activity_phantom)

# Final operator - 
final_pet_op = atten_corr_pet_op + odl.ConstantOperator(scatter, domain=space,
                                                        range=ran)

# %% MLEM - not yet suited for nonlinear forward operators

#callback = (odl.solvers.CallbackShow('MLEM iterate') &
#            odl.solvers.CallbackPrintIteration())
#reco = space.one()
#odl.solvers.mlem(final_pet_op, reco, data, niter=20, callback=callback)

# %% Landweber's Method

# Rough operator norm estimate
opnorm = data.norm() / activity_phantom.norm()
omega = 0.5 / opnorm ** 2

callback = (odl.solvers.CallbackShow('Landweber iterate') &
            odl.solvers.CallbackPrintIteration())
reco = space.zero()
odl.solvers.landweber(final_pet_op, reco, data, niter=20, omega=omega,
                      callback=callback)

