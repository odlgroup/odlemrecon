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
phantom.show()

projection = op(phantom)
projection.show()

callback = odl.solvers.CallbackShow()

x = space.zero()
odl.solvers.conjugate_gradient_normal(op, x, projection, niter=100,
                                      callback=callback)
