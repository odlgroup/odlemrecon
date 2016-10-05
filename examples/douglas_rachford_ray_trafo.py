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
import numpy as np

# %% Set up the problem

fov = np.array([590.625, 590.625, 158.625])
shape = [175, 175, 47]
ran_shape = [192, 192, 175]

space = odl.uniform_discr(-fov/2, fov/2, shape)

# Phantom
phantom = odl.phantom.shepp_logan(space, modified=True)
phantom.show('Phantom')

# Geometry
angle_partition = odl.uniform_partition(0, 2 * np.pi, 60)
detector_partition = odl.uniform_partition([-800, -800], [800, 800],
                                           [200, 200])
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition)

# Forward operator - downsampled Ray transform
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
mask = ray_trafo.range.one()
mask[::2] = 0
ray_trafo = mask * ray_trafo * (4. / 3)

# Create noise-free and noisy data (Poisson noise)
data = ray_trafo(phantom)
data.show('Noise-free data')
scale = 10 / np.max(data)  # 10 counts at max
noisy_data = odl.phantom.poisson_noise(data * scale) / scale
noisy_data.show('Noisy data')

# Starting point, either for pre-iteration or optimization scheme
x = space.zero()

# %% Compute starting point for the TV regularization (optional)

# Landweber solution as starting point
opnorm = odl.operator.power_method_opnorm(ray_trafo, maxiter=4)
omega = 0.5 / opnorm ** 2

callback = odl.solvers.CallbackShow('Landweber iterates')
odl.solvers.landweber(ray_trafo, x, noisy_data, omega=omega, niter=8,
                      callback=callback)

# %% Set up the optimization scheme

# Gradient for TV regularization
gradient = odl.Gradient(space)

grad_norm = odl.operator.power_method_opnorm(
    gradient, xstart=odl.util.testutils.noise_element(gradient.domain),
    maxiter=4)

# Set parameters for Douglas-Rachford
tau = 1.0
sigmas = [0.00002, 3.0]
opnorms = [opnorm, grad_norm]

# Check the convergence criterion
c = tau * sum(s * norm ** 2 for s, norm in zip(sigmas, opnorms))
print('Convergence criterion: c must be < 4, got c = {}'.format(c))
assert c < 4

# Proximals of the convex conjugates of g_i. Those appear in the optimzation
# problem as composition g_i o L_i, where L_i are linear operators.
# We have
# g_1 = ||. - noisy_data||^2, L_1 = ray_trafo,
# g_2 = lam * ||.||_1, L_2 = gradient
lin_ops = [ray_trafo, gradient]
prox_cc_g = [odl.solvers.proximal_cconj_l2_squared(ray_trafo.range,
                                                   g=noisy_data),
             odl.solvers.proximal_cconj_l1(gradient.range, lam=3e-1)]

# Proximal of additional term without operator. We use a box constraint.
prox_f = odl.solvers.proximal_box_constraint(space, 0, 1)

callback = (odl.solvers.CallbackShow(display_step=1) &
            odl.solvers.CallbackPrintIteration())
odl.solvers.douglas_rachford_pd(x, prox_f, prox_cc_g, lin_ops,
                                tau=tau, sigma=sigmas,
                                niter=200, callback=callback)

x.show('douglas rachford result')
