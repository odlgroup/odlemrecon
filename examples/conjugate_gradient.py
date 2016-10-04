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

"""Example of PET tomography problem using the GGLS method."""

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
projection.show('projection')

callback = odl.solvers.CallbackShow()

x = space.zero()
odl.solvers.conjugate_gradient_normal(op, x, projection, niter=100,
                                      callback=callback)
