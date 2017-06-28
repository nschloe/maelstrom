# -*- coding: utf-8 -*-
#
# TODO update
'''
Numerical solution schemes for the Navier--Stokes equation in cylindrical
coordinates,

.. math::
    \\DeclareMathOperator{\\div}{div}

.. math::
    \\begin{align*}
        &\\rho \\left(\\frac{du}{dt} + (u\\cdot\\nabla)u\\right)
          = -\\nabla p
            + \\mu \\left(
                \\frac{1}{r} \\div(r \\nabla u)
                - e_r \\frac{u_r}{r^2}
                \\right)
            + f,\\\\
        &\\frac{1}{r} \\div(r u) &= 0,
    \\end{align*}

cf.
https://en.wikipedia.org/wiki/Navier%E2%80%93Stokes_equations#Representations_in_3D.
In the weak formulation, we consider integrals in pseudo 3D, resulting in a
weighting with :math:`2\\pi r` of the equations. (The volume element is
:math:`2\\pi r \\text{d}x`.)

The order of the variables is taken to be :math:`(r, z, \\theta)`. This makes
sure that for planar domains, the :math:`x`- and :math:`y`-coordinates are
interpreted as :math:`r`, :math:`z`.
'''

from dolfin import (
    TestFunction, Function, Constant, dot, grad, inner, pi, dx, div, solve,
    derivative, TrialFunction, PETScPreconditioner, PETScKrylovSolver,
    as_backend_type, info, assemble, norm, FacetNormal, sqrt, ds, as_vector,
    NonlinearProblem, NewtonSolver, SpatialCoordinate, project
    )

from . import stabilization as stab
from .message import Message


def _momentum_equation(u, v, p, f, rho, mu, stabilization, my_dx):
    '''Weak form of the momentum equation.
    '''
    # rho and my are Constant() functions
    assert rho.values()[0] > 0.0
    assert mu.values()[0] > 0.0

    # Skew-symmetric formulation.
    # Don't include the boundary term
    #
    #   - mu *inner(r*grad(u2)*n  , v2) * 2*pi*ds.
    #
    # This effectively means that at all boundaries where no sufficient
    # Dirichlet-conditions are posed, we assume grad(u)*n to vanish.
    #
    # The original term
    #    u2[0]/(r*r) * v2[0]
    # doesn't explode iff u2[0]~r and v2[0]~r at r=0. Hence, we need to enforce
    # homogeneous Dirichlet-conditions for n.u at r=0. This corresponds to no
    # flow in normal direction through the symmetry axis -- makes sense.
    # When using the 2*pi*r weighting, we can even be a bit lax on the
    # enforcement on u2[0].
    #
    # For this to be well defined, u[0]/r and u[2]/r must be bounded for r=0,
    # so u[0]~u[2]~r must hold. This either needs to be enforced in the
    # boundary conditions (homogeneous Dirichlet for u[0], u[2] at r=0) or must
    # follow from the dynamics of the system.
    #
    # TODO some more explanation for the following lines of code
    mesh = v.function_space().mesh()
    r = SpatialCoordinate(mesh)[0]
    F = rho * 0.5 * (dot(grad(u) * u, v) - dot(grad(v) * u, u)) \
            * 2*pi*r*my_dx \
        + mu * inner(r * grad(u), grad(v)) * 2 * pi * my_dx  \
        + mu * u[0] / r * v[0] * 2 * pi * my_dx  \
        - dot(f, v) * 2*pi*r * my_dx
    if p:
        F += (p.dx(0) * v[0] + p.dx(1) * v[1]) * 2*pi*r * my_dx
    if len(u) == 3:
        F += rho * (-u[2] * u[2] * v[0] + u[0] * u[2] * v[2]) * 2*pi * my_dx
        F += mu * u[2] / r * v[2] * 2 * pi * my_dx

    if stabilization == 'SUPG':
        # TODO check this part of the code
        #
        # SUPG stabilization has the form
        #
        #     <R, tau*grad(v)*u[0]>
        #
        # with R being the residual in strong form. The choice of tau is
        # subject to research.
        tau = stab.supg2(
                u.function_space().mesh(),
                u,
                mu / rho,
                u.function_space().ufl_element().degree()
                )
        # We need to deal with the term
        #
        #     \int mu * (u2[0]/r**2, 0) * dot(R, grad(v2)*b_tau) 2*pi*r*dx
        #
        # somehow. Unfortunately, it's not easy to construct (u2[0]/r**2,
        # 0), cf.  <https://answers.launchpad.net/dolfin/+question/228353>.
        # Strong residual:
        R = + rho * grad(u) * u * 2 * pi * r \
            - mu * div(r * grad(u)) * 2 * pi \
            - f * 2 * pi * r
        if p:
            R += (p.dx(0) * v[0] + p.dx(1) * v[1]) * 2*pi*r * my_dx

        gv = tau * grad(v) * u
        F += dot(R, gv) * my_dx

        # Manually add the parts of the residual which couldn't be cleanly
        # implemented above.
        F += mu * u[0] / r * 2 * pi * gv[0] * my_dx
        if u.function_space().num_sub_spaces() == 3:
            F += rho * (-u[2] * u[2] * gv[0] + u[0] * u[2] * gv[2]) \
                    * 2*pi*my_dx
            F += mu * u[2] / r * gv[2] * 2*pi * my_dx
    else:
        assert stabilization is None

    return F


def compute_tentative_velocity(
        time_step_method, rho, mu,
        u, p0, dt, u_bcs, f, W,
        my_dx,
        stabilization,
        verbose, tol
        ):
    '''Compute the tentative velocity via

    .. math::
        \\rho (u_0 + (u\\cdot\\nabla)u) =
            \\mu \\frav{1}{r} \\div(r \\nabla u) + \\rho g.
    '''

    class TentativeVelocityProblem(NonlinearProblem):
        def __init__(
                self, ui, time_step_method,
                rho, mu,
                u, p0, dt,
                bcs,
                f,
                my_dx,
                stabilization=False
                ):
            super(TentativeVelocityProblem, self).__init__()

            W = ui.function_space()
            v = TestFunction(W)

            self.bcs = bcs

            r = SpatialCoordinate(ui.function_space().mesh())[0]

            def me(uu, ff):
                return _momentum_equation(
                    uu, v, p0, ff, rho, mu, stabilization, my_dx
                    )

            self.F0 = rho * dot(ui - u[0], v) / dt * 2*pi*r*my_dx
            if time_step_method == 'forward euler':
                self.F0 += me(u[0], f[0])
            elif time_step_method == 'backward euler':
                self.F0 += me(ui, f[1])
            else:
                assert time_step_method == 'crank-nicolson', \
                        'Unknown time stepper \'{}\''.format(time_step_method)
                self.F0 += 0.5 * (me(u[0], f[0]) + me(ui, f[1]))

            self.jacobian = derivative(self.F0, ui)
            self.reset_sparsity = True
            return

        # pylint: disable=unused-argument
        def F(self, b, x):
            # We need to evaluate F at x, so we have to make sure that self.F0
            # is assembled for ui=x. We could use a self.ui and set
            #
            #     self.ui.vector()[:] = x
            #
            # here. One way around this copy is to instantiate this class with
            # the same Function ui that is then used for the solver.solve().
            assemble(
                self.F0,
                tensor=b,
                form_compiler_parameters={'optimize': True}
                )
            for bc in self.bcs:
                bc.apply(b, x)
            return

        def J(self, A, x):
            # We can ignore x; see comment at F().
            assemble(
                self.jacobian,
                tensor=A,
                form_compiler_parameters={'optimize': True}
                )
            for bc in self.bcs:
                bc.apply(A)
            self.reset_sparsity = False
            return

    solver = NewtonSolver()
    solver.parameters['maximum_iterations'] = 5
    solver.parameters['absolute_tolerance'] = tol
    solver.parameters['relative_tolerance'] = 0.0
    solver.parameters['report'] = True
    # The nonlinear term makes the problem generally nonsymmetric.
    solver.parameters['linear_solver'] = 'gmres'
    # If the nonsymmetry is too strong, e.g., if u[0] is large, then AMG
    # preconditioning might not work very well.
    # Use HYPRE-Euclid instead of ILU for parallel computation.
    solver.parameters['preconditioner'] = 'hypre_euclid'
    solver.parameters['krylov_solver']['relative_tolerance'] = tol
    solver.parameters['krylov_solver']['absolute_tolerance'] = 0.0
    solver.parameters['krylov_solver']['maximum_iterations'] = 1000
    solver.parameters['krylov_solver']['monitor_convergence'] = verbose

    ui = Function(W)
    step_problem = TentativeVelocityProblem(
            ui,
            time_step_method,
            rho, mu,
            u, p0, dt,
            u_bcs,
            f,
            my_dx,
            stabilization
            )

    # Take u[0] as initial guess.
    ui.assign(u[0])
    solver.solve(step_problem, ui.vector())

    # Make sure ui is from W. This should happen anyways, but somehow doesn't.
    # TODO find out why not
    ui = project(ui, W)
    # div_u = 1/r * div(r*ui)
    return ui


def compute_pressure(
        P, p0,
        mu, ui,
        u,
        my_dx,
        p_bcs=None,
        rotational_form=False,
        tol=1.0e-10,
        verbose=True
        ):
    '''Solve the pressure Poisson equation

    .. math::

        \\begin{align}
          -\\frac{1}{r} \\div(r \\nabla (p_1-p_0)) =
              -\\frac{1}{r} \\div(r u),\\\\
          \\text{(with boundary conditions)},
        \\end{align}

    for :math:`\\nabla p = u`.

    The pressure correction is based on the update formula

    .. math::
        \\frac{\\rho}{dt} (u_{n+1}-u^*)
            + \\begin{pmatrix}
                \\text{d}\\phi/\\text{d}r\\\\
                \\text{d}\\phi/\\text{d}z\\\\
                \\frac{1}{r} \\text{d}\\phi/\\text{d}\\theta
              \\end{pmatrix}
                = 0

    with :math:`\\phi = p_{n+1} - p^*` and

    .. math::

         \\frac{1}{r} \\frac{\\text{d}}{\\text{d}r} (r u_r^{(n+1)})
       + \\frac{\\text{d}}{\\text{d}z}  (u_z^{(n+1)})
       + \\frac{1}{r} \\frac{\\text{d}}{\\text{d}\\theta} (u_{\\theta}^{(n+1)})
           = 0

    With the assumption that u does not change in the direction
    :math:`\\theta`, one derives

    .. math::

       - \\frac{1}{r}   \\div(r \\nabla \\phi) =
           \\frac{1}{r} \\frac{\\rho}{dt}   \\div(r (u_{n+1} - u^*))\\\\
       - \\frac{1}{r} \\langle n, r \\nabla \\phi\\rangle =
           \\frac{1}{r} \\frac{\\rho}{dt} \\langle n, r (u_{n+1} - u^*)\\rangle

    In its weak form, this is

    .. math::

      \\int r \\langle\\nabla\\phi, \\nabla q\\rangle \\,2 \\pi =
           - \\frac{\\rho}{dt} \\int \\div(r u^*) q \\, 2 \\pi
           - \\frac{\\rho}{dt} \\int_{\\Gamma}
                 \\langle n,  r (u_{n+1}-u^*)\\rangle q \\, 2\\pi.

    (The terms :math:`1/r` cancel with the volume elements :math:`2\\pi r`.)
    If the Dirichlet boundary conditions are applied to both :math:`u^*` and
    :math:`u_n` (the latter in the velocity correction step), the boundary
    integral vanishes.

    If no Dirichlet conditions are given (which is the default case), the
    system has no unique solution; one eigenvalue is 0. This however, does not
    hurt CG convergence if the system is consistent, cf. :cite:`vdV03`. And
    indeed it is consistent if and only if

    .. math::
        \\int_\\Gamma r \\langle n, u\\rangle = 0.

    This condition makes clear that for incompressible Navier-Stokes, one
    either needs to make sure that inflow and outflow always add up to 0, or
    one has to specify pressure boundary conditions.

    Note that, when using a multigrid preconditioner as is done here, the
    coarse solver must be chosen such that it preserves the nullspace of the
    problem.
    '''
    W = ui.function_space()
    r = SpatialCoordinate(W.mesh())[0]

    p = TrialFunction(P)
    q = TestFunction(P)
    a2 = dot(r * grad(p), grad(q)) * 2 * pi * my_dx
    # The boundary conditions
    #     n.(p1-p0) = 0
    # are implicitly included.
    #
    # L2 = -div(r*u) * q * 2*pi*my_dx
    div_u = 1/r * (r * u[0]).dx(0) + u[1].dx(1)
    L2 = -div_u * q * 2*pi*r*my_dx
    if p0:
        L2 += r * dot(grad(p0), grad(q)) * 2*pi*my_dx

    # In the Cartesian variant of the rotational form, one makes use of the
    # fact that
    #
    #     curl(curl(u)) = grad(div(u)) - div(grad(u)).
    #
    # The same equation holds true in cylindrical form. Hence, to get the
    # rotational form of the splitting scheme, we need to
    #
    # rotational form
    if rotational_form:
        # If there is no dependence of the angular coordinate, what is
        # div(grad(div(u))) in Cartesian coordinates becomes
        #
        #     1/r div(r * grad(1/r div(r*u)))
        #
        # in cylindrical coordinates (div and grad are in cylindrical
        # coordinates). Unfortunately, we cannot write it down that
        # compactly since u_phi is in the game.
        # When using P2 elements, this value will be 0 anyways.
        div_ui = 1/r * (r * ui[0]).dx(0) + ui[1].dx(1)
        grad_div_ui = as_vector((div_ui.dx(0), div_ui.dx(1)))
        L2 -= r * mu * dot(grad_div_ui, grad(q)) * 2*pi*my_dx
        # div_grad_div_ui = 1/r * (r * grad_div_ui[0]).dx(0) \
        #     + (grad_div_ui[1]).dx(1)
        # L2 += mu * div_grad_div_ui * q * 2*pi*r*dx
        # n = FacetNormal(Q.mesh())
        # L2 -= mu * (n[0] * grad_div_ui[0] + n[1] * grad_div_ui[1]) \
        #     * q * 2*pi*r*ds

    p1 = Function(P)
    if p_bcs:
        solve(
            a2 == L2, p1,
            bcs=p_bcs,
            solver_parameters={
                'linear_solver': 'iterative',
                'symmetric': True,
                'preconditioner': 'hypre_amg',
                'krylov_solver': {
                    'relative_tolerance': tol,
                    'absolute_tolerance': 0.0,
                    'maximum_iterations': 100,
                    'monitor_convergence': verbose
                    }
                }
            )
    else:
        # If we're dealing with a pure Neumann problem here (which is the
        # default case), this doesn't hurt CG if the system is consistent,
        # cf. :cite:`vdV03`. And indeed it is consistent if and only if
        #
        #   \int_\Gamma r n.u = 0.
        #
        # This makes clear that for incompressible Navier-Stokes, one
        # either needs to make sure that inflow and outflow always add up
        # to 0, or one has to specify pressure boundary conditions.
        #
        # If the right-hand side is very small, round-off errors may impair
        # the consistency of the system. Make sure the system we are
        # solving remains consistent.
        A = assemble(a2)
        b = assemble(L2)
        # Assert that the system is indeed consistent.
        e = Function(P)
        e.interpolate(Constant(1.0))
        evec = e.vector()
        evec /= norm(evec)
        alpha = b.inner(evec)
        normB = norm(b)
        # Assume that in every component of the vector, a round-off error
        # of the magnitude DOLFIN_EPS is present. This leads to the
        # criterion
        #    |<b,e>| / (||b||*||e||) < DOLFIN_EPS
        # as a check whether to consider the system consistent up to
        # round-off error.
        #
        # TODO think about condition here
        # if abs(alpha) > normB * DOLFIN_EPS:
        if abs(alpha) > normB * 1.0e-12:
            # divu = 1 / r * (r * u[0]).dx(0) + u[1].dx(1)
            adivu = assemble(((r * u[0]).dx(0) + u[1].dx(1)) * 2 * pi * my_dx)
            info('\\int 1/r * div(r*u) * 2*pi*r  =  %e' % adivu)
            n = FacetNormal(P.mesh())
            boundary_integral = assemble((n[0] * u[0] + n[1] * u[1])
                                         * 2 * pi * r * ds)
            info('\\int_Gamma n.u * 2*pi*r = %e' % boundary_integral)
            message = (
                'System not consistent! '
                '<b,e> = %g, ||b|| = %g, <b,e>/||b|| = %e.') \
                % (alpha, normB, alpha / normB)
            info(message)
            # # Plot the stuff, and project it to a finer mesh with linear
            # # elements for the purpose.
            # plot(divu, title='div(u_tentative)')
            # # Vp = FunctionSpace(Q.mesh(), 'CG', 2)
            # # Wp = MixedFunctionSpace([Vp, Vp])
            # # up = project(u, Wp)
            # fine_mesh = Q.mesh()
            # for k in range(1):
            #     fine_mesh = refine(fine_mesh)
            # V = FunctionSpace(fine_mesh, 'CG', 1)
            # W = V * V
            # # uplot = Function(W)
            # # uplot.interpolate(u)
            # uplot = project(u, W)
            # plot(uplot[0], title='u_tentative[0]')
            # plot(uplot[1], title='u_tentative[1]')
            # # plot(u, title='u_tentative')
            # interactive()
            # exit()
            raise RuntimeError(message)
        # Project out the roundoff error.
        b -= alpha * evec

        #
        # In principle, the ILU preconditioner isn't advised here since it
        # might destroy the semidefiniteness needed for CG.
        #
        # The system is consistent, but the matrix has an eigenvalue 0.
        # This does not harm the convergence of CG, but when
        # preconditioning one has to make sure that the preconditioner
        # preserves the kernel. ILU might destroy this (and the
        # semidefiniteness). With AMG, the coarse grid solves cannot be LU
        # then, so try Jacobi here.
        # <http://lists.mcs.anl.gov/pipermail/petsc-users/2012-February/012139.html>
        #
        prec = PETScPreconditioner('hypre_amg')
        from dolfin import PETScOptions
        PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
        solver = PETScKrylovSolver('cg', prec)
        solver.parameters['absolute_tolerance'] = 0.0
        solver.parameters['relative_tolerance'] = tol
        solver.parameters['maximum_iterations'] = 100
        solver.parameters['monitor_convergence'] = verbose
        # Create solver and solve system
        A_petsc = as_backend_type(A)
        b_petsc = as_backend_type(b)
        p1_petsc = as_backend_type(p1.vector())
        solver.set_operator(A_petsc)
        solver.solve(p1_petsc, b_petsc)
    return p1


def compute_velocity_correction(
        ui, p0, p1, u_bcs, rho, mu, dt,
        rotational_form, my_dx,
        tol, verbose
        ):
    '''Compute the velocity correction according to

    .. math::

        U = u_0 - \\frac{dt}{\\rho} \\nabla (p_1-p_0).
    '''
    W = ui.function_space()
    P = p1.function_space()

    u = TrialFunction(W)
    v = TestFunction(W)
    a3 = dot(u, v) * my_dx
    phi = Function(P)
    phi.assign(p1)
    if p0:
        phi -= p0
    if rotational_form:
        r = SpatialCoordinate(W.mesh())[0]
        div_ui = 1/r * (r * ui[0]).dx(0) + ui[1].dx(1)
        phi += mu * div_ui
    L3 = dot(ui, v) * my_dx \
        - dt / rho * (phi.dx(0) * v[0] + phi.dx(1) * v[1]) * my_dx
    u1 = Function(W)
    solve(
        a3 == L3, u1,
        bcs=u_bcs,
        solver_parameters={
            'linear_solver': 'iterative',
            'symmetric': True,
            'preconditioner': 'hypre_amg',
            'krylov_solver': {
                'relative_tolerance': tol,
                'absolute_tolerance': 0.0,
                'maximum_iterations': 100,
                'monitor_convergence': verbose
                }
            }
        )
    # u = project(ui - k/rho * grad(phi), V)
    # div_u = 1/r * div(r*u)
    r = SpatialCoordinate(W.mesh())[0]
    div_u1 = 1.0 / r * (r * u1[0]).dx(0) + u1[1].dx(1)
    info('||u||_div = {:e}'.format(sqrt(assemble(div_u1 * div_u1 * my_dx))))
    return u1


def _step(
        dt,
        u, p0,
        W, P,
        u_bcs, p_bcs,
        rho, mu,
        stabilization,
        time_step_method,
        f,
        my_dx,
        rotational_form=False,
        verbose=True,
        tol=1.0e-10,
        ):
    '''General pressure projection scheme as described in section 3.4 of
    :cite:`GMS06`.
    '''
    # dt is a Constant() function
    assert dt.values()[0] > 0.0

    with Message('Computing tentative velocity'):
        ui = compute_tentative_velocity(
                time_step_method, rho, mu,
                u, p0, dt, u_bcs, f, W,
                my_dx,
                stabilization,
                verbose, tol
                )

    with Message('Computing pressure correction'):
        p1 = compute_pressure(
                P, p0,
                mu, ui,
                rho * ui / dt,
                my_dx,
                p_bcs=p_bcs,
                rotational_form=rotational_form,
                tol=tol,
                verbose=verbose
                )

    with Message('Computing velocity correction'):
        u1 = compute_velocity_correction(
            ui, p0, p1, u_bcs, rho, mu, dt,
            rotational_form, my_dx,
            tol, verbose
            )

    return u1, p1


class IPCS(object):
    '''
    Incremental pressure correction scheme; for details see :cite:`GMS06`.
    '''
    order = {
        'velocity': 1,
        'pressure': 1,
        }

    def __init__(self, time_step_method='backward euler', stabilization=False):
        self.time_step_method = time_step_method
        self.stabilization = stabilization
        return

    def step(
            self,
            dt,
            u, p0,
            W, P,
            u_bcs, p_bcs,
            rho, mu,
            f,
            verbose=True,
            tol=1.0e-10,
            my_dx=dx
            ):
        return _step(
            dt,
            u, p0,
            W, P,
            u_bcs, p_bcs,
            rho, mu,
            self.stabilization,
            self.time_step_method,
            f,
            verbose=verbose,
            tol=tol,
            my_dx=my_dx
            )
