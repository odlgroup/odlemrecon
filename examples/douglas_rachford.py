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

"""Example of PET tomography problem using the TV regularization."""

import odl
import odlemrecon
import numpy as np

fov = np.array([590.625, 590.625, 158.625])
shape = [175, 175, 47]
ran_shape = [192, 192, 175]

settings = {'SCANNERTYPE': 3,
            'VERBOSE': 0}

space = odl.uniform_discr(-fov/2, fov/2, shape)
ran = odl.uniform_discr([0]*3, ran_shape, ran_shape)

pet_op = odlemrecon.EMReconForwardProjector(space, ran, settings=settings)

phantom = odl.phantom.shepp_logan(space, modified=True)
phantom.show('phantom')

mask = ran.one()
mask[::2] = 0
pet_op = mask * pet_op
projection = pet_op(phantom)
projection.show('projection')

scale = 10 / np.max(projection)
noisy_projection = odl.phantom.poisson_noise(projection * scale) / scale
noisy_projection.show('noisy projection')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 60)
# Detector: uniformly sampled, n = (558, 558), min = (-30, -30), max = (30, 30)
detector_partition = odl.uniform_partition([-800, -800], [800, 800],
                                           [200, 200])
# Discrete reconstruction space

# Astra cannot handle axis aligned origin_to_det unless it is aligned
# with the third coordinate axis. See issue #18 at ASTRA's github.
# This is fixed in new versions of astra, with older versions, this could
# give a zero result.
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)

ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda') * 4 / 3.0
ray_trafo_data = ray_trafo(phantom)
scale_ray = 10 / np.max(ray_trafo_data)
noisy_ray_trafo_data = odl.phantom.poisson_noise(ray_trafo_data * scale_ray) / scale_ray
noisy_ray_trafo_data.show()

# Gradient for TV regularization
gradient = odl.Gradient(space)

# Assemble all operators
lin_ops = [pet_op, gradient]
#lin_ops = [ray_trafo, gradient]
#lin_ops = [pet_op]

# Start with a Landweber solution
opnorm = projection.norm() / phantom.norm()
omega = 0.5 / opnorm ** 2

ray_trafo_norm = odl.operator.power_method_opnorm(ray_trafo, maxiter=4)

callback = odl.solvers.CallbackShow('iterates')

x = space.zero()
#odl.solvers.landweber(pet_op, x, noisy_projection, omega=omega, niter=8,
#                      callback=callback)
odl.solvers.landweber(ray_trafo, x, noisy_ray_trafo_data, omega=omega, niter=8,
                      callback=callback)

# Solve
grad_norm = odl.operator.power_method_opnorm(
    gradient, xstart=odl.util.testutils.noise_element(gradient.domain),
    maxiter=4)
tau = 1.0
sigmas = [0.00002, 3.0]
#sigmas = [0.0001]
opnorms = [opnorm, grad_norm]
#opnorms = [ray_trafo_norm, grad_norm]

# Create proximals as needed
#prox_cc_g = [odl.solvers.proximal_cconj_l2_squared(
#    ray_trafo.range, g=noisy_ray_trafo_data),
#             odl.solvers.proximal_cconj_l1(gradient.range, lam=3e-1)]
prox_cc_g = [odl.solvers.proximal_cconj_l2_squared(
    pet_op.range, g=noisy_projection),
             odl.solvers.proximal_cconj_l1(gradient.range, lam=3e-1)]
#prox_cc_g = [odl.solvers.proximal_cconj_l2_squared(pet_op.range, g=noisy_projection)]
prox_f = odl.solvers.proximal_box_constraint(space, 0, 1)

print(tau * sum(s * norm ** 2
                for s, norm in zip(sigmas, opnorms)))

x = ray_trafo.domain.zero()
callback = (odl.solvers.CallbackShow(display_step=1) &
            odl.solvers.CallbackPrintIteration())
odl.solvers.douglas_rachford_pd(x, prox_f, prox_cc_g, lin_ops,
                                tau=tau, sigma=sigmas,
                                niter=200, callback=callback)

x.show('douglas rachford result')
