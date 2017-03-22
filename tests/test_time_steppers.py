# -*- coding: utf-8 -*-
#
import nose

from dolfin import set_log_level, WARNING, Expression, FunctionSpace, \
    DirichletBC, Function, errornorm, project, plot, interactive, triangle, \
    norm, UnitIntervalMesh, pi, inner, grad, dx, ds, dot, UnitSquareMesh, \
    FacetNormal, interval, RectangleMesh, TrialFunction, TestFunction, \
    assemble, lhs, rhs, MPI

import maelstrom.time_steppers as ts
import sympy as smp
import numpy
import itertools

import helpers

# Turn down the log level to only error messages.
set_log_level(WARNING)
#set_log_level(ERROR)
#set_log_level(0)


def test_generator():
    '''Test order of time discretization.
    '''
    # TODO add test for spatial order
    problems = [
        #problem_sinsin1d,
        #problem_sinsin,
        problem_coscos_cartesian,
        #problem_coscos_cylindrical,
        #problem_stefanboltzmann
        ]
    # Methods together with the expected order of convergence.
    methods = [
        #ts.Dummy,
        ts.ExplicitEuler
        #ts.ImplicitEuler
        #ts.Trapezoidal
        ]
    # Loop over all methods and check the order of convergence.
    for method, problem in itertools.product(methods, problems):
        yield _check_time_order, problem, method


def _check_time_order(problem, method):
    mesh_sizes = [20, 40, 80]
    Dt = [0.5 ** k for k in range(15)]
    errors, order = _compute_time_errors(problem, method, mesh_sizes, Dt)
    # The test is considered passed if the numerical order of convergence
    # matches the expected order in at least the first step in the coarsest
    # spatial discretization, and is not getting worse as the spatial
    # discretizations are refining.
    tol = 0.1
    k = 0
    expected_order = method['order']
    for i in range(order.shape[0]):
        nose.tools.assert_almost_equal(order[i][k], expected_order,
                                       delta=tol
                                       )
        while k + 1 < len(order[i]) \
                and abs(order[i][k + 1] - expected_order) < tol:
            k += 1
    return errors


def _compute_time_errors(problem, method, mesh_sizes, Dt, plot_error=False):
    mesh_generator, solution, ProblemClass, cell_type = problem()
    # Translate data into FEniCS expressions.
    fenics_sol = Expression(smp.printing.ccode(solution['value']),
                            degree=solution['degree'],
                            t=0.0,
                            cell=cell_type
                            )
    # Compute the problem
    errors = {'theta': numpy.empty((len(mesh_sizes), len(Dt)))}
    # Create initial state.
    # Deepcopy the expression into theta0. Specify the cell to allow for
    # more involved operations with it (e.g., grad()).
    theta0 = Expression(fenics_sol.cppcode,
                        degree=solution['degree'],
                        t=0.0,
                        cell=cell_type
                        )
    for k, mesh_size in enumerate(mesh_sizes):
        mesh = mesh_generator(mesh_size)
        V = FunctionSpace(mesh, 'CG', 1)
        theta_approx = Function(V)
        theta0p = project(theta0, V)
        stepper = method(ProblemClass(V))
        if plot_error:
            error = Function(V)
        for j, dt in enumerate(Dt):
            # TODO We are facing a little bit of a problem here, being the
            # fact that the time stepper only accept elements from V as u0.
            # In principle, though, this isn't necessary or required. We
            # could allow for arbitrary expressions here, but then the API
            # would need changing for problem.lhs(t, u).
            # Think about this.
            stepper.step(theta_approx, theta0p,
                         0.0, dt,
                         tol=1.0e-12,
                         verbose=False
                         )
            fenics_sol.t = dt
            #
            # NOTE
            # When using errornorm(), it is quite likely to see a good part
            # of the error being due to the spatial discretization.  Some
            # analyses "get rid" of this effect by (sometimes implicitly)
            # projecting the exact solution onto the discrete function
            # space.
            errors['theta'][k][j] = errornorm(fenics_sol, theta_approx)
            if plot_error:
                error.assign(project(fenics_sol - theta_approx, V))
                plot(error, title='error (dt=%e)' % dt)
                interactive()
    return errors, stepper.name, stepper.order


def _check_space_order(problem, method):
    mesh_generator, solution, weak_F = problem()

    # Translate data into FEniCS expressions.
    fenics_sol = Expression(smp.prining.ccode(solution['value']),
                            degree=solution['degree'],
                            t=0.0
                            )

    # Create initial solution.
    theta0 = Expression(fenics_sol.cppcode,
                        degree=solution['degree'],
                        t=0.0,
                        cell=triangle
                        )

    # Estimate the error component in space.
    # Leave out too rough discretizations to avoid showing spurious errors.
    N = [2 ** k for k in range(2, 8)]
    dt = 1.0e-8
    Err = []
    H = []
    for n in N:
        mesh = mesh_generator(n)
        H.append(MPI.max(mesh.hmax()))
        V = FunctionSpace(mesh, 'CG', 3)
        # Create boundary conditions.
        fenics_sol.t = dt
        #bcs = DirichletBC(V, fenics_sol, 'on_boundary')
        # Create initial state.
        theta_approx = method(V,
                              weak_F,
                              theta0,
                              0.0, dt,
                              bcs=[solution],
                              tol=1.0e-12,
                              verbose=True
                              )
        # Compute the error.
        fenics_sol.t = dt
        Err.append(errornorm(fenics_sol, theta_approx)
                   / norm(fenics_sol, mesh=mesh)
                   )
        print('n: %d    error: %e' % (n, Err[-1]))

    from matplotlib import pyplot as pp
    # Compare with order 1, 2, 3 curves.
    for o in [2, 3, 4]:
        pp.loglog([H[0], H[-1]],
                  [Err[0], Err[0] * (H[-1] / H[0]) ** o],
                  color='0.5'
                  )
    # Finally, the actual data.
    pp.loglog(H, Err, '-o')
    pp.xlabel('h_max')
    pp.ylabel('||u-u_h|| / ||u||')
    pp.show()
    return


def problem_sinsin1d():
    '''sin-sin example.
    '''
    def mesh_generator(n):
        return UnitIntervalMesh(n)
    x = smp.DeferredVector('x')
    t = smp.symbols('t')
    #m = smp.sin(0.5*pi*t)
    m = smp.exp(t) - 0.0
    #theta = m * x * (1-x)
    theta = m * smp.sin(1 * pi * x[0])
    # Produce a matching rhs.
    f_sympy = smp.diff(theta, t) - smp.diff(theta, x[0], 2)
    f = Expression(smp.printing.ccode(f_sympy), degree=numpy.infty, t=0.0)

    # The corresponding operator in weak form.
    def weak_F(t, u_t, u, v):
        # All time-dependent components be set to t.
        f.t = t
        F = - inner(grad(u), grad(v)) * dx \
            + f * v * dx
        return F
    return mesh_generator, theta, weak_F, interval


def problem_sinsin():
    '''sin-sin example.
    '''
    def mesh_generator(n):
        return UnitSquareMesh(n, n, 'left/right')
        #return RectangleMesh(1.0, 0.0, 2.0, 1.0, n, n)
    x = smp.DeferredVector('x')
    t = smp.symbols('t')
    #x, y, t = smp.symbols('x, y, t')
    # Choose the solution something that cannot exactly be expressed by
    # polynomials. Choosing the sine here makes all first-order scheme be
    # second-order accurate since d2sin/dt2 = 0 at t=0.
    #m = smp.sin(0.5*pi*t)
    m = smp.exp(t) - 0.0
    theta = m * x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1])
    #theta = m * smp.sin(1*pi*x) * smp.sin(1*pi*y)
    rho = 5.0
    cp = 2.0
    kappa = 3.0
    # Produce a matching rhs.
    f_sympy = rho * cp * smp.diff(theta, t) \
        - smp.diff(kappa * smp.diff(theta, x[0]), x[0]) \
        - smp.diff(kappa * smp.diff(theta, x[1]), x[1])
    f = Expression(smp.printing.ccode(f_sympy), degree=4, t=0.0)

    # The corresponding operator in weak form.
    def weak_F(t, u_t, u, v):
        # Define the differential equation.
        mesh = v.function_space().mesh()
        n = FacetNormal(mesh)
        # All time-dependent components be set to t.
        f.t = t
        F = - inner(kappa * grad(u), grad(v / (rho * cp))) * dx \
            + inner(kappa * grad(u), n) * v / (rho * cp) * ds \
            + f * v / (rho * cp) * dx
        return F
    return mesh_generator, theta, weak_F, triangle


def problem_coscos_cartesian():
    '''cos-cos example. Inhomogeneous boundary conditions.
    '''
    def mesh_generator(n):
        #mesh = UnitSquareMesh(n, n, 'left/right')
        mesh = RectangleMesh(1.0, 0.0, 2.0, 1.0, n, n, 'left/right')
        return mesh
    t = smp.symbols('t')
    rho = 6.0
    cp = 4.0
    kappa_sympy = smp.exp(t)
    x = smp.DeferredVector('x')
    # Choose the solution something that cannot exactly be expressed by
    # polynomials.
    #theta = smp.sin(t) * smp.sin(pi*x) * smp.sin(pi*y)
    #theta = smp.cos(0.5*pi*t) * smp.sin(pi*x) * smp.sin(pi*y)
    #theta = (smp.exp(t)-1) * smp.cos(3*pi*(x-1.0)) * smp.cos(7*pi*y)
    #theta = (1-smp.cos(t)) * smp.cos(3*pi*(x-1.0)) * smp.cos(7*pi*y)
    #theta = smp.log(1+t) * smp.cos(3*pi*(x-1.0)) * smp.cos(7*pi*y)
    theta = smp.log(2 + t) * smp.cos(pi * (x[0] - 1.0)) * smp.cos(pi * x[1])
    # Produce a matching rhs.
    f_sympy = rho * cp * smp.diff(theta, t) \
        - smp.diff(kappa_sympy * smp.diff(theta, x[0]), x[0]) \
        - smp.diff(kappa_sympy * smp.diff(theta, x[1]), x[1])

    f = Expression(smp.printing.ccode(f_sympy), degree=numpy.infty, t=0.0)
    kappa = Expression(smp.printing.ccode(kappa_sympy), degree=1, t=0.0)

    # The corresponding operator in weak form.
    class HeatEquation(ts.ParabolicProblem):

        def __init__(self, V):
            super(HeatEquation, self).__init__()
            # Define the differential equation.
            self.V = V
            self.rho_cp = rho * cp
            self.sol = Expression(smp.printing.ccode(theta),
                                  degree=numpy.infty,
                                  t=0.0,
                                  cell=triangle
                                  )
            return

        def get_system(self, t):
            kappa.t = t
            f.t = t
            n = FacetNormal(self.V.mesh())
            u = TrialFunction(self.V)
            v = TestFunction(self.V)
            F = inner(kappa * grad(u), grad(v / self.rho_cp)) * dx \
                - inner(kappa * grad(u), n) * v / self.rho_cp * ds \
                - f * v / self.rho_cp * dx
            return assemble(lhs(F)), assemble(rhs(F))

        def get_bcs(self, t):
            self.sol.t = t
            return [DirichletBC(self.V, self.sol, 'on_boundary')]

    return mesh_generator, theta, HeatEquation, triangle


def problem_coscos_cylindrical():
    '''cos-cos example. Inhomogeneous boundary conditions.
    '''
    def mesh_generator(n):
        #mesh = UnitSquareMesh(n, n, 'left/right')
        mesh = RectangleMesh(1.0, 0.0, 2.0, 1.0, n, n, 'left/right')
        return mesh

    t = smp.symbols('t')
    rho = 2.0
    cp = 3.0
    kappa_sympy = smp.exp(t)

    # Cylindrical coordinates.
    x = smp.DeferredVector('x')
    # Solution.
    #theta = smp.sin(t) * smp.sin(pi*(x[0]-1)) * smp.sin(pi*x[1])
    theta = (smp.exp(t) - 1) * smp.cos(pi * (x[0] - 1)) * smp.cos(pi * x[1])
    #theta = smp.log(2+t) * smp.cos(pi*(x[0]-1.0)) * smp.cos(pi*x[1])

    # Convection.
    b_sympy = (-x[1], x[0] - 1)
    #b_sympy = (0.0, 0.0)

    # Produce a matching rhs.
    f_sympy = rho * cp * smp.diff(theta, t) \
        + rho * cp * (b_sympy[0] * smp.diff(theta, x[0])
                      + b_sympy[1] * smp.diff(theta, x[1])
                      ) \
        - 1 / x[0] * smp.diff(x[0]*kappa_sympy * smp.diff(theta, x[0]), x[0]) \
        - smp.diff(kappa_sympy * smp.diff(theta, x[1]), x[1])

    # convert to FEniCS expressions
    f = Expression(smp.printing.ccode(f_sympy), numpy.infty, t=0.0)
    b = Expression((smp.printing.ccode(b_sympy[0]),
                    smp.printing.ccode(b_sympy[1])),
                   degree=1,
                   t=0.0
                   )
    kappa = Expression(smp.printing.ccode(kappa_sympy), degree=1, t=0.0)

    # The corresponding operator in weak form.
    def weak_F(t, u_t, u, v):
        # Define the differential equation.
        mesh = v.function_space().mesh()
        n = FacetNormal(mesh)
        r = Expression('x[0]', degree=1, cell=triangle)
        # All time-dependent components be set to t.
        f.t = t
        b.t = t
        kappa.t = t
        F = - inner(b, grad(u)) * v * dx \
            - 1.0 / (rho * cp) * dot(r * kappa * grad(u), grad(v / r)) * dx \
            + 1.0 / (rho * cp) * dot(r * kappa * grad(u), n) * v / r * ds \
            + 1.0 / (rho * cp) * f * v * dx
        return F
    return mesh_generator, theta, weak_F, triangle


def problem_stefanboltzmann():
    '''Heat equation with Stefan-Boltzmann boundary conditions, i.e.,
    du/dn = u^4 - u_0^4
    '''
    def mesh_generator(n):
        mesh = UnitSquareMesh(n, n, 'left/right')
        return mesh

    t = smp.symbols('t')
    rho = 1.0
    cp = 1.0
    kappa = 1.0
    x = smp.DeferredVector('x')
    # Choose the solution something that cannot exactly be expressed by
    # polynomials.
    #theta = smp.sin(t) * smp.sin(pi*x) * smp.sin(pi*y)
    #theta = smp.cos(0.5*pi*t) * smp.sin(pi*x) * smp.sin(pi*y)
    #theta = (smp.exp(t)-1) * smp.cos(3*pi*(x-1.0)) * smp.cos(7*pi*y)
    #theta = (1-smp.cos(t)) * smp.cos(3*pi*(x-1.0)) * smp.cos(7*pi*y)
    #theta = smp.log(1+t) * smp.cos(3*pi*(x-1.0)) * smp.cos(7*pi*y)
    theta = smp.log(2 + t) * smp.cos(pi * x[0]) * smp.cos(pi * x[1])
    # Produce a matching rhs.
    f_sympy = rho * cp * smp.diff(theta, t) \
        - smp.diff(kappa * smp.diff(theta, x[0]), x[0]) \
        - smp.diff(kappa * smp.diff(theta, x[1]), x[1])
    # Produce a matching u0.
    # u_0^4 = u^4 - du/dn
    # ONLY WORKS IF du/dn==0.
    u0 = theta
    # convert to FEniCS expressions
    f = Expression(smp.printing.ccode(f_sympy), degree=numpy.infty, t=0.0)
    u0 = Expression(smp.printing.ccode(u0), degree=numpy.infty, t=0.0)

    # The corresponding operator in weak form.
    def weak_F(t, u_t, u, v):
        # All time-dependent components be set to t.
        u0.t = t
        f.t = f
        F = - 1.0 / (rho * cp) * kappa * dot(grad(u), grad(v)) * dx \
            + 1.0 / (rho * cp) * kappa * (u*u*u*u - u0*u0*u0*u0) * v * ds \
            + 1.0 / (rho * cp) * f * v * dx
        return F
    return mesh_generator, theta, weak_F, triangle


if __name__ == '__main__':
    # For debugging purposes, show some info.
    #mesh_sizes = [20, 40, 80]
    mesh_sizes = [40, 80, 160]
    Dt = [0.5 ** k for k in range(7)]
    errors, name, _ = _compute_time_errors(
        problem_coscos_cartesian,
        #ts.Dummy,
        #ts.ExplicitEuler,
        #ts.ImplicitEuler,
        ts.Trapezoidal,
        mesh_sizes, Dt,
        )
    helpers.show_timeorder_info(Dt, mesh_sizes, errors)
