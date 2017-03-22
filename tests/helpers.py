# -*- coding: utf-8 -*-
#
'''
Helper functions for PDE consistency tests.
'''
import numpy
import nose

from dolfin import Expression, info, assemble, FunctionSpace, \
    interpolate, plot, interactive, errornorm, dx, Function, \
    VectorFunctionSpace, DirichletBC, project
import sympy as smp
import warnings

from maelstrom.message import Message

try:
    import prettyplotlib as ppl
    from prettyplotlib import plt as pp
except:
    from matplotlib import pyplot as pp


def _truncate_degree(degree, max_degree=10):
    if degree > max_degree:
        warnings.warn(('Expression degree (%r) > maximum degree (%d). '
                       'Truncating.')
                      % (degree, max_degree)
                      )
        return max_degree
    else:
        return degree


def show_timeorder_info(Dt, mesh_sizes, errors):
    '''Performs consistency check for the given problem/method combination and
    show some information about it. Useful for debugging.
    '''
    # Compute the numerical order of convergence.
    orders = {}
    for key in errors:
        orders[key] = _compute_numerical_order_of_convergence(Dt, errors[key])

    # Print the data to the screen
    for i, mesh_size in enumerate(mesh_sizes):
        print
        print('Mesh size %d:' % mesh_size)
        print('dt = %e' % Dt[0]),
        for label, e in errors.items():
            print('   err_%s = %e' % (label, e[i][0])),
        print
        for j in range(len(Dt) - 1):
            print('                 '),
            for label, o in orders.items():
                print('   ord_%s = %e' % (label, o[i][j])),
            print
            print('dt = %e' % Dt[j+1]),
            for label, e in errors.items():
                print('   err_%s = %e' % (label, e[i][j+1])),
            print

    # Create a figure
    for label, err in errors.items():
        pp.figure()
        ax = pp.axes()
        # Plot the actual data.
        for i, mesh_size in enumerate(mesh_sizes):
            pp.loglog(Dt, err[i], '-o', label=mesh_size)
        # Compare with order curves.
        pp.autoscale(False)
        e0 = err[-1][0]
        for o in range(7):
            pp.loglog([Dt[0], Dt[-1]],
                      [e0, e0 * (Dt[-1] / Dt[0]) ** o],
                      color='0.7')
        pp.xlabel('dt')
        pp.ylabel('||%s-%s_h||' % (label, label))
        # pp.title('Method: %s' % method['name'])
        ppl.legend(ax, loc=4)
    pp.show()
    return


def _compute_numerical_order_of_convergence(Dt, errors):
    orders = numpy.empty((errors.shape[0], errors.shape[1] - 1))
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1] - 1):
            orders[i, j] = numpy.log(errors[i][j + 1] / errors[i][j]) \
                / numpy.log(Dt[j + 1] / Dt[j])
    return orders


def _check_time_order(problem, MethodClass, tol=1.0e-10):
    mesh_sizes = [20, 40, 80]
    Dt = [0.5 ** k for k in range(15)]
    errors, order = compute_time_errors(problem, MethodClass, mesh_sizes, Dt)
    # The test is considered passed if the numerical order of
    # convergence matches the expected order in at least the first
    # step in the coarsest spatial discretization, and is not
    # getting worse as the spatial discretizations are refining.
    k = 0
    expected_order = MethodClass['order']
    for i in range(order.shape[0]):
        nose.tools.assert_almost_equal(order[i][k], expected_order,
                                       delta=0.1
                                       )
        while k + 1 < len(order[i]) and \
                abs(order[i][k + 1] - expected_order) < tol:
            k += 1
    return errors


def compute_time_errors(problem, MethodClass, mesh_sizes, Dt):

    mesh_generator, solution, f, mu, rho, cell_type = problem()
    # Translate data into FEniCS expressions.
    sol_u = Expression((smp.printing.ccode(solution['u']['value'][0]),
                        smp.printing.ccode(solution['u']['value'][1])
                        ),
                       degree=_truncate_degree(solution['u']['degree']),
                       t=0.0,
                       cell=cell_type
                       )
    sol_p = Expression(smp.printing.ccode(solution['p']['value']),
                       degree=_truncate_degree(solution['p']['degree']),
                       t=0.0,
                       cell=cell_type
                       )

    fenics_rhs0 = Expression((smp.printing.ccode(f['value'][0]),
                              smp.printing.ccode(f['value'][1])
                              ),
                             degree=_truncate_degree(f['degree']),
                             t=0.0,
                             mu=mu, rho=rho,
                             cell=cell_type
                             )
    # Deep-copy expression to be able to provide f0, f1 for the Dirichlet-
    # boundary conditions later on.
    fenics_rhs1 = Expression(fenics_rhs0.cppcode,
                             degree=_truncate_degree(f['degree']),
                             t=0.0,
                             mu=mu, rho=rho,
                             cell=cell_type
                             )
    # Create initial states.
    p0 = Expression(
        sol_p.cppcode,
        degree=_truncate_degree(solution['p']['degree']),
        t=0.0,
        cell=cell_type
        )

    # Compute the problem
    errors = {'u': numpy.empty((len(mesh_sizes), len(Dt))),
              'p': numpy.empty((len(mesh_sizes), len(Dt)))
              }
    for k, mesh_size in enumerate(mesh_sizes):
        info('')
        info('')
        with Message('Computing for mesh size %r...' % mesh_size):
            mesh = mesh_generator(mesh_size)
            mesh_area = assemble(1.0 * dx(mesh))
            W = VectorFunctionSpace(mesh, 'CG', 2)
            P = FunctionSpace(mesh, 'CG', 1)
            method = MethodClass(W, P,
                                 rho, mu,
                                 theta=1.0,
                                 #theta=0.5,
                                 stabilization=None
                                 #stabilization='SUPG'
                                 )
            u1 = Function(W)
            p1 = Function(P)
            err_p = Function(P)
            divu1 = Function(P)
            for j, dt in enumerate(Dt):
                # Prepare previous states for multistepping.
                u = [Expression(
                    sol_u.cppcode,
                    degree=_truncate_degree(solution['u']['degree']),
                    t=0.0,
                    cell=cell_type
                    ),
                    # Expression(
                    #sol_u.cppcode,
                    #degree=_truncate_degree(solution['u']['degree']),
                    #t=0.5*dt,
                    #cell=cell_type
                    #)
                    ]
                sol_u.t = dt
                u_bcs = [DirichletBC(W, sol_u, 'on_boundary')]
                sol_p.t = dt
                #p_bcs = [DirichletBC(P, sol_p, 'on_boundary')]
                p_bcs = []
                fenics_rhs0.t = 0.0
                fenics_rhs1.t = dt
                method.step(dt,
                            u1, p1,
                            u, p0,
                            u_bcs=u_bcs, p_bcs=p_bcs,
                            f0=fenics_rhs0, f1=fenics_rhs1,
                            verbose=False,
                            tol=1.0e-10
                            )
                sol_u.t = dt
                sol_p.t = dt
                errors['u'][k][j] = errornorm(sol_u, u1)
                # The pressure is only determined up to a constant which makes
                # it a bit harder to define what the error is. For our
                # purposes, choose an alpha_0\in\R such that
                #
                #    alpha0 = argmin ||e - alpha||^2
                #
                # with  e := sol_p - p.
                # This alpha0 is unique and explicitly given by
                #
                #     alpha0 = 1/(2|Omega|) \int (e + e*)
                #            = 1/|Omega| \int Re(e),
                #
                # i.e., the mean error in \Omega.
                alpha = assemble(sol_p * dx(mesh)) \
                    - assemble(p1 * dx(mesh))
                alpha /= mesh_area
                # We would like to perform
                #     p1 += alpha.
                # To avoid creating a temporary function every time, assume
                # that p1 lives in a function space where the coefficients
                # represent actual function values. This is true for CG
                # elements, for example. In that case, we can just add any
                # number to the vector of p1.
                p1.vector()[:] += alpha
                errors['p'][k][j] = errornorm(sol_p, p1)

                show_plots = False
                if show_plots:
                    plot(p1, title='p1', mesh=mesh)
                    plot(sol_p, title='sol_p', mesh=mesh)
                    err_p.vector()[:] = p1.vector()
                    sol_interp = interpolate(sol_p, P)
                    err_p.vector()[:] -= sol_interp.vector()
                    #plot(sol_p - p1, title='p1 - sol_p', mesh=mesh)
                    plot(err_p, title='p1 - sol_p', mesh=mesh)
                    #r = Expression('x[0]', degree=1, cell=triangle)
                    #divu1 = 1 / r * (r * u1[0]).dx(0) + u1[1].dx(1)
                    divu1.assign(project(u1[0].dx(0) + u1[1].dx(1), P))
                    plot(divu1, title='div(u1)')
                    interactive()
    return errors
