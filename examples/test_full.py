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
from dolfin import (
    parameters, Measure, Function, FunctionSpace, Constant, SubMesh, plot,
    interactive, project, XDMFFile, DOLFIN_EPS, as_vector, info, norm,
    assemble, TestFunction, TrialFunction, MPI, dx, interpolate,
    VectorFunctionSpace
    )

import numpy
from numpy import pi

import maelstrom.navier_stokes as cyl_ns
import maelstrom.stokes_heat as stokes_heat
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

    theta_average = 1530.0

    kappa_wpi_const = \
        k_wpi if isinstance(k_wpi, float) else k_wpi(theta_average)
    mu_wpi_const = \
        mu_wpi if isinstance(mu_wpi, float) else mu_wpi(theta_average)
    cp_wpi_const = \
        cp_wpi if isinstance(cp_wpi, float) else cp_wpi(theta_average)

    # initial guess
    u0 = interpolate(Constant((0.0, 0.0, 0.0)), W)
    p0 = interpolate(Constant(0.0), P)
    theta0 = interpolate(Constant(theta_average), Q)
    theta0.rename('temperature', 'temperature')

    u0, p0, theta0 = stokes_heat.solve(
        mesh, W_element, P_element, Q_element,
        u0, p0, theta0,
        kappa_wpi_const, rho_wpi, mu_wpi_const, cp_wpi_const,
        g, extra_force,
        joule_wpi,
        u_bcs, p_bcs,
        theta_bcs_d,
        theta_bcs_n,
        dx_submesh, ds_submesh
        )

    # Create a *deep* copy of u0, p0, to be able to deal with them as actually
    # separate entities.
    # u0, p0 = up0.split(deepcopy=True)
    return u0, p0, theta0


def _compute_lorentz_joule(
        mesh, coils, mu, sigma, omega,
        wpi, submesh_workpiece,
        subdomain_indices, subdomains,
        show=False
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
        lorentz = cmx.compute_lorentz(Phi, omega, sigma[wpi])

        # Show the Lorentz force in the workpiece.
        # W_element = VectorElement('CG', submesh_workpiece.ufl_cell(), 1)
        # First project onto the entire mesh, then onto the submesh; see bug
        # <https://bitbucket.org/fenics-project/dolfin/issues/869/projecting-grad-onto-submesh-error>.
        W = VectorFunctionSpace(mesh, 'CG', 1)
        pl = project(lorentz, W)
        W2 = VectorFunctionSpace(submesh_workpiece, 'CG', 1)
        pl = project(pl, W2)
        pl.rename('Lorentz force', 'Lorentz force')
        with XDMFFile(submesh_workpiece.mpi_comm(), 'lorentz.xdmf') as f:
            f.parameters['flush_output'] = True
            f.write(pl)

        if show:
            plot(pl, title='Lorentz force')
            interactive()

        # Get Joule heat source.
        joule = cmx.compute_joule(
                Phi, voltages, omega, sigma, mu, subdomain_indices
                )

        if show:
            show_joule = subdomain_indices
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

    # To work around bug
    # <https://bitbucket.org/fenics-project/dolfin/issues/869/projecting-grad-onto-submesh-error>.
    # return the projection `pl` and not `lorentz` itself.
    # TODO remove this workaround
    return pl, joule_wpi


def _store_and_plot(outfile, u, p, theta, t):
    outfile.write(u, t)
    outfile.write(p, t)
    outfile.write(theta, t)
    plot(theta, title='temperature', rescale=True)
    plot(u, title='velocity', rescale=True)
    plot(p, title='pressure', rescale=True)
    # interactive()
    return


def _average(u):
    '''Computes the average value of a function u over its domain.
    '''
    return assemble(u * dx) \
        / assemble(1.0 * dx(u.function_space().mesh()))


def _compute(
        u0, p0, theta0, problem, voltages, target_time, show=False
        ):
    submesh_workpiece = problem.W.mesh()

    # Define a facet measure on the boundaries. See discussion on
    # <https://bitbucket.org/fenics-project/dolfin/issue/249/facet-specification-doesnt-work-on-ds>.
    ds_workpiece = Measure('ds', subdomain_data=problem.wp_boundaries)

    subdomain_indices = problem.subdomain_materials.keys()
    theta_average = _average(theta0)

    info('Input voltages:')
    info(repr(voltages))

    if voltages is None:
        lorentz_wpi = None
        joule_wpi = Constant(0.0)
    else:
        # Merge coil rings with voltages.
        coils = [
            {'rings': coil_domain, 'c_type': 'voltage', 'c_value': voltage}
            for coil_domain, voltage in zip(problem.coil_domains, voltages)
            ]
        # Build subdomain parameter dictionaries for Maxwell
        mu_const = {
            i: problem.subdomain_materials[i].magnetic_permeability
            for i in subdomain_indices
            }
        sigma_const = {
            i: problem.subdomain_materials[i].electrical_conductivity
            for i in subdomain_indices
            }
        for i in subdomain_indices:
            mu_const[i] = \
                mu_const[i] if isinstance(mu_const[i], float) \
                else mu_const[i](theta_average)
            sigma_const[i] = \
                sigma_const[i] if isinstance(sigma_const[i], float) \
                else sigma_const[i](theta_average)
        # Do the Maxwell dance
        lorentz_wpi, joule_wpi = _compute_lorentz_joule(
                problem.mesh, coils,
                mu_const, sigma_const, problem.omega,
                problem.wpi, submesh_workpiece,
                subdomain_indices,
                problem.subdomains,
                show=show
                )

    # Start time, time step.
    t = 0.0
    dt = 1.0e-3
    dt_max = 1.0e-1

    # Standard gravity, <https://en.wikipedia.org/wiki/Standard_gravity>.
    grav = 9.80665
    assert problem.W.num_sub_spaces() == 3
    g = Constant((0.0, -grav, 0.0))

    # Compute a few mesh characteristics.
    wpi_area = assemble(1.0 * dx(submesh_workpiece))
    # mesh.hmax() is a local function; get the global hmax.
    hmax_workpiece = MPI.max(
            submesh_workpiece.mpi_comm(), submesh_workpiece.hmax()
            )

    # Take the maximum length in x-direction as characteristic length of the
    # domain.
    coords = submesh_workpiece.coordinates()
    char_length = max(coords[:, 0]) - min(coords[:, 0])

    # Prepare some parameters for the Navier-Stokes simulation in the workpiece
    m = problem.subdomain_materials[problem.wpi]
    k_wpi = m.thermal_conductivity
    cp_wpi = m.specific_heat_capacity
    rho_wpi = m.density
    mu_wpi = m.dynamic_viscosity

    # Redefine the heat problem with the new u0.
    heat_problem = cyl_heat.Heat(
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

        if show:
            _store_and_plot(outfile, u0, p0, theta0, t)

        successful_steps = 0
        failed_steps = 0
        while t < target_time + DOLFIN_EPS:
            info('Successful steps: {}    (failed: {}, total: {})'.format(
                    successful_steps,
                    failed_steps,
                    successful_steps + failed_steps
                 ))
            with Message('Time step {:e} -> {:e}...'.format(t, t + dt)):
                # Do one heat time step.
                with Message('Computing heat...'):
                    # For time-stepping in buoyancy-driven flows, see
                    #
                    # Numerical solution of buoyancy-driven flows;
                    # Einar Rossebø Christensen;
                    # Master's thesis;
                    # <http://www.diva-portal.org/smash/get/diva2:348831/FULLTEXT01.pdf>.
                    #
                    # Similar to the present approach, one first solves for
                    # velocity and pressure, then for temperature.
                    #
                    heat_stepper = ts.ImplicitEuler(heat_problem)
                    ns_stepper = cyl_ns.IPCS(
                            time_step_method='backward euler',
                            stabilization=None
                            )
                    # Use HYPRE-Euclid instead of ILU for parallel computation.
                    theta1 = heat_stepper.step(
                            theta0,
                            t, dt,
                            tol=1.0e-12,
                            maxiter=1000,
                            krylov='gmres',
                            preconditioner='hypre_euclid',
                            verbose=False
                            )
                theta0_average = _average(theta0)
                try:
                    # Do one Navier-Stokes time step.
                    with Message('Computing flux and pressure...'):
                        # Include proper temperature-dependence here to account
                        # for the Boussinesq effect.
                        f0 = rho_wpi(theta0) * g
                        f1 = rho_wpi(theta1) * g
                        if lorentz_wpi is not None:
                            f = as_vector(
                                (lorentz_wpi[0], lorentz_wpi[1], 0.0)
                                )
                            f0 += f
                            f1 += f
                        for b in problem.u_bcs:
                            print(b.function_space())
                        exit(1)
                        u1, p1 = ns_stepper.step(
                                dt,
                                {0: u0}, p0,
                                problem.W, problem.P,
                                problem.u_bcs, problem.p_bcs,
                                rho_wpi(theta0_average),
                                mu_wpi(theta0_average),
                                f={0: f0, 1: f1},
                                tol=1.0e-10,
                                my_dx=dx(submesh_workpiece)
                                )
                        exit(1)
                except RuntimeError as e:
                    print('>>>>>>>>>>>>>>>>>>>>>>>>>')
                    print(e)
                    print('<<<<<<<<<<<<<<<<<<<<<<<<<')
                    exit(1)
                    info(e.message)
                    info(
                        'Navier--Stokes solver failed to converge. '
                        'Decrease time step from {:e} to {:e} and try again.'
                        .format(dt, 0.5 * dt)
                        )
                    dt *= 0.5
                    failed_steps += 1
                    continue
                successful_steps += 1

                # Assignments and plotting.
                theta0.assign(theta1)
                u0.assign(u1)
                p0.assign(p1)

                if show:
                    _store_and_plot(outfile, u0, p0, theta0, t + dt)

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
                    info('previous dt: {:e}'.format(dt))
                    info('target dt: {:e}'.format(target_dt))
                    # agg is the aggressiveness factor. The distance between
                    # the current step size and the target step size is reduced
                    # by |1-agg|. Hence, if agg==1 then dt_next==target_dt.
                    # Otherwise target_dt is approached more slowly.
                    agg = 0.5
                    dt = min(dt_max,
                             # At most double the step size from step to step.
                             dt * min(2.0, 1.0 + agg * (target_dt - dt) / dt)
                             )
                    info('new dt:    {:e}'.format(dt))
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
    info('Max temperature: {:e}'.format(vec.max()))
    info('Min temperature: {:e}'.format(vec.min()))
    info('Av  temperature: {:e}'.format(av_temperature))
    info('')
    info('Max velocity: {:e}'.format(umax))
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
    info('Prandtl number: {:e}'.format(
        _get_prandtl(mu_const, rho_const, cp, k)
        ))
    info('Reynolds number: {:e}'.format(
        _get_reynolds(rho_const, mu_const, char_length, char_velocity)
        ))
    info('Grashof number: {:e}'.format(
         _get_grashof(
             rho, mu_const, grav,
             av_temperature, char_length, temperature_difference
             )
         ))
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


def test_optimize(num_steps=1, target_time=0.1, show=False):
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

    g = Constant((0.0, -9.80665, 0.0))
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

    project(u0, problem.W)

    if show:
        plot(u0, title='u')
        plot(p0, title='p')
        plot(theta0, title='theta')
        interactive()

    # Rename the states for plotting and such.
    u0.rename('velocity', 'velocity')
    p0.rename('pressure', 'pressure')
    theta0.rename('temperature', 'temperature')

    for k, alpha in enumerate(Alpha):
        # Scale the voltages
        v = alpha * numpy.array(voltages)
        _compute(u0, p0, theta0, problem, v, target_time, show=show)

        # From the second iteration on, only go for at most 60 secs
        target_time = min(60.0, target_time)
    return


if __name__ == '__main__':
    test_optimize(num_steps=51, target_time=600.0, show=True)
