#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
'''
Full* simulation of the melt problem.

* Conditions apply.
  1. The effect of the melt movement on the magnetic field is neglected.
     The Lorentz force is precomputed.
  2. The heat transfer from the coils to the melt is not simulated.
     Temperature boundary conditions are set at the melt boundary directly.

A worthwhile read in for the simulation of crystal growth is :cite:`Derby89`.
'''
import os

from dolfin import (
    parameters, Measure, Function, FunctionSpace, Constant, SubMesh, plot,
    interactive, ds, project, XDMFFile, DirichletBC, dot, grad, Expression,
    triangle, DOLFIN_EPS, as_vector, info, norm, assemble, TestFunction,
    TrialFunction, KrylovSolver, MPI, split, NonlinearProblem, derivative,
    inner, TestFunctions, dx, assign, errornorm, interpolate, MixedElement
    )

import numpy
from numpy import pi

import maelstrom.navier_stokes as cyl_ns
import maelstrom.stokes as cyl_stokes
import maelstrom.maxwell as cmx
import maelstrom.heat as cyl_heat
import maelstrom.time_steppers as ts
from maelstrom.message import Message

import problems

# We need to allow extrapolation here since otherwise, the equation systems
# for Maxwell cannot be constructed: They contain the velocity `u` (from
# Navier-Stokes) which is only defined on the workpiece subdomain.
# Cf. <https://answers.launchpad.net/dolfin/+question/210508>.
parameters['allow_extrapolation'] = True
parameters['std_out_all_processes'] = False


class FixedPointSolver(object):
    '''
    Fixed-point iteration for
        F(x) + x = x
    '''
    def __init__(self):
        self.parameters = {
                'maximum_iterations': 0,
                'absolute_tolerance': 0.0,
                'relative_tolerance': 0.0,
                'report': True
                }
        return

    def _report(self, k, nrm, nrm0):
        if self.parameters['report']:
            print(('Fixed-point iteration %d:' % k) +
                  (' r (abs) = %e (tol = %e)'
                   % (nrm, self.parameters['absolute_tolerance'])) +
                  (' r (rel) = %e (tol = %e)'
                   % (nrm/nrm0, self.parameters['relative_tolerance']))
                  )
        return

    def solve(self, problem, vector):
        res = vector.copy()
        problem.F(res, vector)
        nrm = norm(res)
        nrm0 = nrm
        k = 0
        self._report(k, nrm, nrm0)
        while nrm > self.parameters['absolute_tolerance'] \
                or nrm / nrm0 > self.parameters['relative_tolerance']:
            problem.F(res, vector)
            vector[:] += res
            nrm = norm(res)
            k += 1
            self._report(k, nrm, nrm0)
        return


class Stokes(object):

    def __init__(self, u, p, v, q, f):
        super(Stokes, self).__init__()
        self.u = u
        self.p = p
        self.v = v
        self.q = q
        self.f = f
        return

    def F0(self, mu):
        u = self.u
        p = self.p
        v = self.v
        q = self.q
        f = self.f
        mu = Constant(mu)
        # Momentum equation (without the nonlinear term).
        r = Expression('x[0]', degree=1, cell=triangle)
        F0 = mu * inner(r * grad(u), grad(v)) * 2 * pi * dx \
            + mu * u[0] / r * v[0] * 2 * pi * dx \
            - dot(f, v) * 2 * pi * r * dx
        if len(u) == 3:
            # Discard nonlinear component.
            F0 += mu * u[2] / r * v[2] * 2 * pi * dx
        F0 += (p.dx(0) * v[0] + p.dx(1) * v[1]) * 2 * pi * r * dx

        # Incompressibility condition.
        # div_u = 1/r * div(r*u)
        F0 += (1.0 / r * (r * u[0]).dx(0) + u[1].dx(1)) * q \
            * 2 * pi * r * dx
        return F0


def dbcs_to_productspace(W, bcs_list):
    new_bcs = []
    for k, bcs in enumerate(bcs_list):
        for bc in bcs:
            C = bc.function_space().component()
            # pylint: disable=len-as-condition
            if len(C) == 0:
                new_bcs.append(DirichletBC(W.sub(k),
                                           bc.value(),
                                           bc.domain_args[0]))
            elif len(C) == 1:
                new_bcs.append(DirichletBC(W.sub(k).sub(int(C[0])),
                                           bc.value(),
                                           bc.domain_args[0]))
            else:
                raise RuntimeError('Illegal number of subspace components.')
    return new_bcs


class StokesHeat(NonlinearProblem):

    def __init__(self, WPQ,
                 kappa, rho, mu, cp,
                 theta_average,
                 g, extra_force,
                 heat_source,
                 u_bcs, p_bcs,
                 theta_dirichlet_bcs=None,
                 theta_neumann_bcs=None,
                 theta_robin_bcs=None,
                 my_dx=dx,
                 my_ds=ds
                 ):

        theta_dirichlet_bcs = theta_dirichlet_bcs or {}
        theta_neumann_bcs = theta_neumann_bcs or {}
        theta_robin_bcs = theta_robin_bcs or {}

        super(StokesHeat, self).__init__()
        # Translate the boundary conditions into the product space.
        self.bcs = dbcs_to_productspace(
            WPQ,
            [u_bcs, p_bcs, theta_dirichlet_bcs]
            )

        self.uptheta = Function(WPQ)
        u, p, theta = split(self.uptheta)
        v, q, zeta = TestFunctions(WPQ)

        # Right-hand side for momentum equation.
        f = rho(theta) * g
        if extra_force is not None:
            f += as_vector((extra_force[0], extra_force[1], 0.0))

        self.stokes = Stokes(u, p, v, q, f)

        self.heat = cyl_heat.HeatCylindrical(
            WPQ.sub(2), theta, zeta,
            u,
            kappa, rho(theta_average), cp,
            source=heat_source,
            dirichlet_bcs=theta_dirichlet_bcs,
            neumann_bcs=theta_neumann_bcs,
            robin_bcs=theta_robin_bcs,
            my_dx=my_dx,
            my_ds=my_ds
            )

        self.F0 = self.stokes.F0(mu) + self.heat.F0
        self.jacobian = derivative(self.F0, self.uptheta)
        return

    def set_parameter(self, mu):
        self.F0 = self.stokes.F0(mu) + self.heat.F0
        self.jacobian = derivative(self.F0, self.uptheta)
        return

    def F(self, b, x):
        self.uptheta.vector()[:] = x
        assemble(self.F0,
                 tensor=b,
                 form_compiler_parameters={'optimize': True}
                 )
        for bc in self.bcs:
            bc.apply(b, x)
        return

    def J(self, A, x):
        self.uptheta.vector()[:] = x
        assemble(self.jacobian,
                 tensor=A,
                 form_compiler_parameters={'optimize': True}
                 )
        for bc in self.bcs:
            bc.apply(A)
        return


def _average(u):
    '''Computes the average value of a function u over its domain.
    '''
    return assemble(u * dx) \
        / assemble(1.0 * dx(u.function_space().mesh()))


def _construct_initial_state(
        mesh,
        W_element, P_element, Q_element,
        k_wpi, cp_wpi, rho_wpi, mu_wpi,
        joule_wpi,
        u_bcs, p_bcs,
        theta_bcs_d, theta_bcs_n,
        dx_submesh,
        ds_submesh,
        g, extra_force
        ):
    '''Construct an initial state for the Navier-Stokes simulation.
    '''
    W = FunctionSpace(mesh, W_element)
    P = FunctionSpace(mesh, P_element)
    Q = FunctionSpace(mesh, Q_element)

    # theta_average = _average(theta0)
    u0 = interpolate(Constant((0.0, 0.0, 0.0)), W)
    p0 = interpolate(Constant(0.0), P)
    theta0 = interpolate(Constant(1530.0), Q)
    theta0.name = 'temperature'
    theta_average = 1530.0

    if isinstance(rho_wpi, float):
        rho_wpi_const = rho_wpi
    else:
        rho_wpi_const = rho_wpi(theta_average)
    if isinstance(k_wpi, float):
        k_wpi_const = k_wpi
    else:
        k_wpi_const = k_wpi(theta_average)
    if isinstance(cp_wpi, float):
        cp_wpi_const = cp_wpi
    else:
        cp_wpi_const = cp_wpi(theta_average)

    mode = 'newton'
    # mode = 'banach'

    if mode == 'newton':
        WPQ = FunctionSpace(
            mesh, MixedElement([W_element, P_element, Q_element])
            )
        uptheta0 = Function(WPQ)

        # Initial guess
        assign(uptheta0.sub(0), u0)
        assign(uptheta0.sub(1), p0)
        assign(uptheta0.sub(2), theta0)

        stokes_heat_problem = StokesHeat(
            WPQ,
            k_wpi_const,
            rho_wpi,
            mu_wpi(theta_average),
            cp_wpi_const,
            theta_average,
            g, extra_force,
            joule_wpi,
            u_bcs, p_bcs,
            theta_dirichlet_bcs=theta_bcs_d,
            theta_neumann_bcs=theta_bcs_n,
            my_dx=dx_submesh,
            my_ds=ds_submesh
            )
        # solver = FixedPointSolver()
        from dolfin import PETScSNESSolver
        solver = PETScSNESSolver()
        # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/SNES/SNESType.html
        solver.parameters['method'] = 'newtonls'
        # The Jacobian system for Stokes (+heat) are hard to solve.
        # Use LU for now.
        solver.parameters['linear_solver'] = 'lu'
        solver.parameters['maximum_iterations'] = 5
        solver.parameters['absolute_tolerance'] = 1.0e-10
        solver.parameters['relative_tolerance'] = 0.0
        solver.parameters['report'] = True

        solver.solve(stokes_heat_problem, uptheta0.vector())

        # u0, p0, theta0 = split(uptheta0)
        # Create a *deep* copy of u0, p0, theta0 to be able to deal with them
        # as actually separate entities vectors.
        u0, p0, theta0 = uptheta0.split(deepcopy=True)

    else:
        assert mode == 'banach'
        # Solve the coupled heat-Stokes equation approximately. Do this
        # iteratively by solving the heat equation, then solving Stokes with
        # the updated heat, the heat equation with the updated velocity and so
        # forth until the change is 'small'.
        WP = FunctionSpace(mesh, MixedElement([W_element, P_element]))
        # Initialize functions.
        up0 = Function(WP)
        u0, p0 = up0.split()

        theta1 = Function(Q)
        while True:
            heat_problem = cyl_heat.HeatCylindrical(
                Q, TrialFunction(Q), TestFunction(Q),
                b=u0,
                kappa=k_wpi_const,
                rho=rho_wpi_const,
                cp=cp_wpi_const,
                source=joule_wpi,
                dirichlet_bcs=theta_bcs_d,
                neumann_bcs=theta_bcs_n,
                my_dx=dx_submesh,
                my_ds=ds_submesh
                )
            _solve_stationary(heat_problem, theta1, verbose=False)

            f = rho_wpi(theta0) * g
            if extra_force:
                f += as_vector((extra_force[0], extra_force[1], 0.0))

            # Solve problem for velocity, pressure.
            # up1 = up0.copy()
            cyl_stokes.stokes_solve(
                up0,
                mu_wpi(theta_average),
                u_bcs, p_bcs,
                f=f,
                dx=dx_submesh,
                tol=1.0e-10,
                verbose=False,
                maxiter=1000
                )

            plot(u0)
            plot(theta0)

            theta_diff = errornorm(theta0, theta1)
            info('||theta - theta0|| = %e' % theta_diff)
            # info('||u - u0||         = %e' % u_diff)
            # info('||p - p0||         = %e' % p_diff)
            # diff = theta_diff + u_diff + p_diff
            diff = theta_diff
            info('sum = %e' % diff)

            # # Show the iterates.
            # plot(theta0, title='theta0')
            # plot(u0, title='u0')
            # interactive()
            # #exit()
            if diff < 1.0e-10:
                break

            theta0.assign(theta1)

    plot(u0, title='u')
    plot(p0, title='p')
    plot(theta0, title='theta')
    interactive()

    # Create a *deep* copy of u0, p0, to be able to deal with them as actually
    # separate entities.
    # u0, p0 = up0.split(deepcopy=True)
    return u0, p0, theta0


def _compute_lorentz_joule(
        mesh, coils, mu, sigma, omega,
        wpi, submesh_workpiece,
        output_folder, subdomain_indices, subdomains
        ):
    # Function space for magnetic scalar potential, Lorentz force etc.
    V = FunctionSpace(mesh, 'CG', 1)
    # Compute the magnetic field.
    # The Maxwell equations depend on two parameters that change during the
    # computation: (a) the temperature, and (b) the velocity field u0. We
    # assume though that changes in either of the two will only marginally
    # influence the magnetic field. Consequently, we precompute all associated
    # values.
    dx_subdomains = Measure('dx', subdomain_data=subdomains)
    with Message('Computing magnetic field...'):
        Phi, voltages = cmx.compute_potential(
                coils,
                V,
                dx_subdomains,
                mu, sigma, omega,
                convections={}
                # io_submesh=submesh_workpiece
                )
        # Get resulting Lorentz force.
        lorentz_wpi = cmx.compute_lorentz(Phi, omega, sigma[wpi])

        # Show the Lorentz force in the workpiece.
        V_submesh = FunctionSpace(submesh_workpiece, 'CG', 1)
        pl = project(lorentz_wpi, V_submesh * V_submesh)
        pl.rename('Lorentz force', 'Lorentz force')
        filename = os.path.join(output_folder, 'lorentz.xdmf')
        with XDMFFile(submesh_workpiece.mpi_comm(), filename) as f:
            f.parameters['flush_output'] = True
            f.write(pl)

        show_lorentz = False
        if show_lorentz:
            plot(pl, title='Lorentz force')
            interactive()

        # Get Joule heat source.
        joule = cmx.compute_joule(
                Phi, voltages, omega, sigma, mu, subdomain_indices
                )
        show_joule = []
        # show_joule = subdomain_indices
        for ii in show_joule:
            # Show Joule heat source.
            submesh = SubMesh(mesh, subdomains, ii)
            W_submesh = FunctionSpace(submesh, 'CG', 1)
            jp = Function(W_submesh, name='Joule heat source')
            jp.assign(project(joule[ii], W_submesh))
            # jp.interpolate(joule[ii])
            plot(jp)
            interactive()

        joule_wpi = joule[wpi]
    return lorentz_wpi, joule_wpi


def _compute(
        u0, p0, theta0, problem, voltages, T,
        output_folder
        ):
    submesh_workpiece = problem.W.mesh()

    # Define a facet measure on the boundaries. See discussion on
    # <https://bitbucket.org/fenics-project/dolfin/issue/249/facet-specification-doesnt-work-on-ds>.
    ds_workpiece = Measure('ds', subdomain_data=problem.wp_boundaries)

    info('Input voltages:')
    info(repr(voltages))
    coils = []
    if voltages is not None:
        # Merge coil rings with voltages.
        for coil_domain, voltage in zip(problem.coil_domains, voltages):
            coils.append({
                'rings': coil_domain,
                'c_type': 'voltage',
                'c_value': voltage
                })

    subdomain_indices = problem.subdomain_materials.keys()

    theta_average = _average(theta0)

    # Start time, end time, time step.
    t = 0.0
    dt = 1.0e-3
    dt_max = 1.0e-1

    # Standard gravity, <https://en.wikipedia.org/wiki/Standard_gravity>.
    grav = 9.80665
    num_subspaces = problem.W.num_sub_spaces()
    if num_subspaces == 2:
        g = Constant((0.0, -grav))
    elif num_subspaces == 3:
        g = Constant((0.0, -grav, 0.0))
    else:
        raise RuntimeError('Illegal number of subspaces (%d).' % num_subspaces)

    # Compute a few mesh characteristics.
    wpi_area = assemble(1.0 * dx(submesh_workpiece))
    # mesh.hmax() is a local function; get the global hmax.
    hmax_workpiece = MPI.max(submesh_workpiece.mpi_comm(),
                             submesh_workpiece.hmax()
                             )
    # Take the maximum length in x-direction as characteristic length of the
    # domain.
    coords = submesh_workpiece.coordinates()
    char_length = max(coords[:, 0]) - min(coords[:, 0])

    if voltages is None:
        lorentz_wpi = None
        joule_wpi = Constant(0.0)
    else:
        # Build subdomain parameter dictionaries for Maxwell
        mu = {}
        sigma = {}
        for i in subdomain_indices:
            material = problem.subdomain_materials[i]
            mu[i] = material.magnetic_permeability
            if not isinstance(mu[i], float):
                mu[i] = mu[i](theta_average)
            sigma[i] = material.electrical_conductivity
            if not isinstance(sigma[i], float):
                sigma[i] = sigma[i](theta_average)
        # Do the Maxwell dance
        lorentz_wpi, joule_wpi = \
            _compute_lorentz_joule(problem.mesh, coils,
                                   mu, sigma, problem.omega,
                                   problem.wpi, submesh_workpiece,
                                   output_folder,
                                   subdomain_indices,
                                   problem.subdomains
                                   )

    # Prepare some parameters for the Navier-Stokes simulation in the workpiece
    m = problem.subdomain_materials[problem.wpi]
    k_wpi = m.thermal_conductivity
    cp_wpi = m.specific_heat_capacity
    rho_wpi = m.density
    mu_wpi = m.dynamic_viscosity

    u1 = Function(problem.W)
    p1 = Function(problem.P)
    theta1 = Function(problem.Q)

    # Redefine the heat problem with the new u0.
    heat_problem = cyl_heat.HeatCylindrical(
        problem.Q, TrialFunction(problem.Q), TestFunction(problem.Q),
        b=u0,
        kappa=k_wpi,
        rho=rho_wpi(theta_average),
        cp=cp_wpi,
        source=joule_wpi,
        dirichlet_bcs=problem.theta_bcs_d,
        neumann_bcs=problem.theta_bcs_n,
        my_dx=dx(submesh_workpiece),
        my_ds=ds_workpiece
        )

    show_total_force = False
    if show_total_force:
        f = rho_wpi(theta0) * g
        if lorentz_wpi:
            f += as_vector((lorentz_wpi[0], lorentz_wpi[1], 0.0))
        plot(f, mesh=submesh_workpiece, title='Total external force')
        interactive()

    with XDMFFile(submesh_workpiece.mpi_comm(), 'full.xdmf') as outfile:
        outfile.parameters['flush_output'] = True
        outfile.parameters['rewrite_function_mesh'] = False

        def store_and_plot(u, p, theta, t):
            outfile.write(u, t)
            outfile.write(p, t)
            outfile.write(theta, t)
            plot(theta, title='temperature', rescale=True)
            plot(u, title='velocity', rescale=True)
            plot(p, title='pressure', rescale=True)
            # interactive()

        store_and_plot(u0, p0, theta0, t)

        # For time-stepping in buoyancy-driven flows, see
        #
        #     Numerical solution of buoyancy-driven flows;
        #     Einar Rossebø Christensen;
        #     Master's thesis;
        #     <http://www.diva-portal.org/smash/get/diva2:348831/FULLTEXT01.pdf>.
        #
        # Similar to the present approach, one first solves for velocity and
        # pressure, then for temperature.
        #
        heat_stepper = ts.ImplicitEuler(heat_problem)
        ns_stepper = cyl_ns.IPCS(
                problem.W, problem.P,
                rho=rho_wpi(theta_average),
                mu=mu_wpi(theta_average),
                theta=1.0,
                stabilization=None,
                dx=dx(submesh_workpiece)
                # stabilization='SUPG'
                )

        successful_steps = 0
        failed_steps = 0
        while t < T + DOLFIN_EPS:
            info('Successful steps: {}    (failed: {}, total: {})'.format(
                    successful_steps,
                    failed_steps,
                    successful_steps + failed_steps
                 ))
            with Message('Time step {:e} -> {:e}...'.format(t, t + dt)):
                try:
                    # Do one heat time step.
                    with Message('Computing heat...'):
                        # Use HYPRE-Euclid instead of ILU for parallel
                        # computation.
                        heat_stepper.step(
                                theta1,
                                theta0,
                                t, dt,
                                tol=1.0e-12,
                                maxiter=1000,
                                krylov='gmres',
                                preconditioner='hypre_euclid',
                                verbose=False
                                )
                    # Do one Navier-Stokes time step.
                    with Message('Computing flux and pressure...'):
                        # Include proper temperature-dependence here to account
                        # for Boussinesq effect.
                        f0 = rho_wpi(theta0) * g
                        f1 = rho_wpi(theta1) * g
                        if lorentz_wpi is not None:
                            f = as_vector(
                                (lorentz_wpi[0], lorentz_wpi[1], 0.0)
                                )
                            f0 += f
                            f1 += f
                        ns_stepper.step(
                                dt,
                                u1, p1,
                                [u0], p0,
                                u_bcs=problem.u_bcs, p_bcs=problem.p_bcs,
                                f0=f0, f1=f1,
                                verbose=False,
                                tol=1.0e-10
                                )
                except RuntimeError as e:
                    info(e.message)
                    info('Navier--Stokes solver failed to converge. '
                         'Decrease time step from %e to %e and try again.' %
                         (dt, 0.5 * dt)
                         )
                    dt *= 0.5
                    failed_steps += 1
                    continue
                successful_steps += 1

                # Assignments and plotting.
                theta0.assign(theta1)
                u0.assign(u1)
                p0.assign(p1)

                store_and_plot(u0, p0, theta0, t + dt)

                t += dt
                with Message('Diagnostics...'):
                    # Print some general info on the flow in the crucible.
                    umax = get_umax(u0)
                    _print_diagnostics(theta0, umax,
                                       submesh_workpiece, wpi_area,
                                       problem.subdomain_materials,
                                       problem.wpi,
                                       rho_wpi,
                                       mu_wpi,
                                       char_length,
                                       grav)
                    info('')
                with Message('Step size adaptation...'):
                    # Some smooth step-size adaption.
                    target_dt = 0.2 * hmax_workpiece / umax
                    info('previous dt: %e' % dt)
                    info('target dt: %e' % target_dt)
                    # agg is the aggressiveness factor. The distance between
                    # the current step size and the target step size is reduced
                    # by |1-agg|. Hence, if agg==1 then dt_next==target_dt.
                    # Otherwise target_dt is approached more slowly.
                    agg = 0.5
                    dt = min(dt_max,
                             # At most double the step size from step to step.
                             dt * min(2.0, 1.0 + agg * (target_dt - dt) / dt)
                             )
                    info('new dt:    %e' % dt)
                    info('')
                info('')
    return


def get_umax(u):
    # Compute the maximum velocity.
    # First get a vector with the squared norms, then get the maximum value of
    # it. This assumes that the maximum is attained in a node, and that the
    # vectors express the value in certain control points (nodes, edge
    # midpoints etc.).
    usplit = u.split(deepcopy=True)
    vec = usplit[0].vector() * usplit[0].vector()
    for k in range(1, len(u)):
        vec += usplit[k].vector() * usplit[k].vector()
    return numpy.sqrt(norm(vec, 'linf'))


def _print_diagnostics(
        theta0, umax,
        submesh_workpiece,
        wpi_area,
        subdomain_materials,
        wpi,
        rho,
        mu,
        char_length,
        grav
        ):
    av_temperature = assemble(theta0 * dx(submesh_workpiece)) / wpi_area
    vec = theta0.vector()
    temperature_difference = vec.max() - vec.min()
    info('Max temperature: %e' % vec.max())
    info('Min temperature: %e' % vec.min())
    info('Av  temperature: %e' % av_temperature)
    info('')
    info('Max velocity: %e' % umax)
    info('')
    char_velocity = umax
    melt_material = subdomain_materials[wpi]
    rho_const = rho(av_temperature)

    cp = melt_material.specific_heat_capacity
    if not isinstance(cp, float):
        cp = cp(av_temperature)
    k = melt_material.thermal_conductivity
    if not isinstance(k, float):
        k = k(av_temperature)

    mu_const = mu(av_temperature)
    #
    info('Prandtl number: %e'
         % _get_prandtl(mu_const, rho_const, cp, k))
    info('Reynolds number: %e'
         % _get_reynolds(rho_const, mu_const, char_length, char_velocity))
    info('Grashof number: %e'
         % _get_grashof(rho, mu_const, grav,
                        av_temperature,
                        char_length,
                        temperature_difference))
    return


def _get_prandtl(mu, rho, cp, k):
    '''Prandtl number.
    '''
    nu = mu / rho
    return nu * rho * cp / k


def _get_reynolds(rho, mu, char_length, char_velocity):
    '''Reynolds number.
    '''
    return rho * char_velocity * char_length / mu


def _get_grashof(rho, mu, grav, theta_average, char_length, deltaT):
    '''Grashof number.
    '''
    # Volume expansion coefficient: alpha = -V'(T)/V(T).
    # Approximate V'(T).
    dT = 1.0e-2
    vpt = (rho(theta_average + dT) - rho(theta_average - dT)) / (2 * dT)
    volume_expansion = -vpt / rho(theta_average)
    nu = mu / rho(theta_average)
    return volume_expansion * deltaT * char_length ** 3 * grav / nu ** 2


def _solve_stationary(problem, theta, t=0.0, verbose=True, mode='lu'):
    '''Solve the stationary heat equation.
    '''
    A, b = problem.get_system(t=t)
    for bc in problem.get_bcs(t=t):
        bc.apply(A, b)

    if mode == 'lu':
        from dolfin import solve
        solve(A, theta.vector(), b, 'lu')

    elif mode == 'krylov':
        # AMG won't work well if the convection is too strong.
        solver = KrylovSolver('gmres', 'hypre_amg')
        solver.parameters['relative_tolerance'] = 1.0e-12
        solver.parameters['absolute_tolerance'] = 0.0
        solver.parameters['maximum_iterations'] = 142
        solver.parameters['monitor_convergence'] = verbose
        solver.solve(A, theta.vector(), b)
    else:
        raise ValueError('Illegal mode \'%s\'.' % mode)
    return theta


def _parse_args():
    import argparse
    default_dir = '.'
    parser = argparse.ArgumentParser(description='Full simulation.')
    parser.add_argument('directory',
                        type=str,
                        help='directory for storing the output (default: %s)'
                             % default_dir,
                        default=default_dir,
                        nargs='?'
                        )
    return parser.parse_args()


def _gravitational_force(num_subspaces):
    grav = 9.80665
    if num_subspaces == 2:
        return Constant((0.0, -grav))
    elif num_subspaces == 3:
        return Constant((0.0, -grav, 0.0))
    else:
        raise RuntimeError('Illegal number of subspaces (%d).' % num_subspaces)


def test_optimize(num_steps=1, target_time=0.1):
    # The voltage is defined as
    #
    #     v(t) = Im(exp(i omega t) v)
    #          = Im(exp(i (omega t + arg(v)))) |v|
    #          = sin(omega t + arg(v)) |v|.
    #
    # Hence, for a lagging voltage, arg(v) needs to be negative.
    # voltages = None
    #
    Alpha = numpy.linspace(0.0, 2.0, num_steps)
    voltages = [
        38.0 * numpy.exp(-1j * 2 * pi * 2 * 70.0 / 360.0),
        38.0 * numpy.exp(-1j * 2 * pi * 1 * 70.0 / 360.0),
        38.0 * numpy.exp(-1j * 2 * pi * 0 * 70.0 / 360.0),
        25.0 * numpy.exp(-1j * 2 * pi * 0 * 70.0 / 360.0),
        25.0 * numpy.exp(-1j * 2 * pi * 1 * 70.0 / 360.0)
        ]

    # voltages = [0.0, 0.0, 0.0, 0.0, 0.0]
    #
    # voltages = [
    #         25.0 * numpy.exp(-1j * 2*pi * 2 * 70.0/360.0),
    #         25.0 * numpy.exp(-1j * 2*pi * 1 * 70.0/360.0),
    #         25.0 * numpy.exp(-1j * 2*pi * 0 * 70.0/360.0),
    #         38.0 * numpy.exp(-1j * 2*pi * 0 * 70.0/360.0),
    #         38.0 * numpy.exp(-1j * 2*pi * 1 * 70.0/360.0)
    #         ]
    #
    # voltages = [
    #         38.0 * numpy.exp(+1j * 2*pi * 2 * 70.0/360.0),
    #         38.0 * numpy.exp(+1j * 2*pi * 1 * 70.0/360.0),
    #         38.0 * numpy.exp(+1j * 2*pi * 0 * 70.0/360.0),
    #         25.0 * numpy.exp(+1j * 2*pi * 0 * 70.0/360.0),
    #         25.0 * numpy.exp(+1j * 2*pi * 1 * 70.0/360.0)
    #         ]

    problem = problems.Crucible()

    # Solve construct initial state without Lorentz force and Joule heat.
    m = problem.subdomain_materials[problem.wpi]
    k_wpi = m.thermal_conductivity
    cp_wpi = m.specific_heat_capacity
    rho_wpi = m.density
    mu_wpi = m.dynamic_viscosity

    g = _gravitational_force(problem.W.num_sub_spaces())
    submesh_workpiece = problem.W.mesh()
    ds_workpiece = Measure('ds', subdomain_data=problem.wp_boundaries)
    u0, p0, theta0 = _construct_initial_state(
        submesh_workpiece,
        problem.W_element, problem.P_element, problem.Q_element,
        k_wpi, cp_wpi, rho_wpi, mu_wpi,
        Constant(0.0),
        problem.u_bcs, problem.p_bcs,
        problem.theta_bcs_d, problem.theta_bcs_n,
        dx(submesh_workpiece),
        ds_workpiece,
        g,
        extra_force=None
        )

    # Rename the states for plotting and such.
    u0.rename('velocity', 'velocity')
    p0.rename('pressure', 'pressure')
    theta0.rename('temperature', 'temperature')

    args = _parse_args()
    output_folder = args.directory
    for k, alpha in enumerate(Alpha):
        # Scale the voltages
        v = alpha * numpy.array(voltages)
        of = os.path.join(output_folder, 'step%02d' % k)
        if not os.path.exists(of):
            os.makedirs(of)
        _compute(u0, p0, theta0, problem, v, target_time, of)

        # From the second iteration on, only go for at most 60 secs
        target_time = min(60.0, target_time)
    return


if __name__ == '__main__':
    test_optimize(num_steps=51, target_time=600.0)
