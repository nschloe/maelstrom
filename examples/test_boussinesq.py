#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
'''
Coupled solve of the Navier--Stokes and the heat equation
in cylindrical coordinates.
'''
import problems

import maelstrom.navier_stokes as cyl_ns
import maelstrom.time_steppers as ts

from dolfin import (
    parameters, Measure, Constant, Function, XDMFFile, mpi_comm_world,
    SpatialCoordinate, DOLFIN_EPS, begin, end, dot, grad, pi, project, plot,
    norm, Expression, triangle
    )

# We need to allow extrapolation here since otherwise, the equation systems
# for Maxwell cannot be constructed: They contain the velocity `u` (from
# Navier-Stokes) which is only defined on the workpiece subdomain.
# Cf. <https://answers.launchpad.net/dolfin/+question/210508>.
parameters['allow_extrapolation'] = True

GMSH_EPS = 1.0e-15


def test():
    # mesh, V, Q, u_bcs, p_bcs, heat_boundary, left_boundary, right_boundary, \
    #     lower_boundary, upper_boundary = _domain_ballintube()
    # mesh, subdomains, subdomain_materials, wpi, V, Q, P, u_bcs, p_bcs, \
    #     heater_boundary = _domain_peter()
    # mesh, subdomains, subdomain_materials, wpi, V, Q, P, u_bcs, p_bcs, \
    #     background_temp, heater_bcs = probs.crucible()

    problem = problems.Crucible()

    dx = Measure('dx', subdomain_data=problem.subdomains)

    subdomain_indices = problem.subdomain_materials.keys()

    # Density depends on temperature.
    rho = {}
    mu = {}
    cp = {}
    kappa = {}
    for k in subdomain_indices:
        material = problem.subdomain_materials[k]
        rho[k] = material.density
        try:
            cp[k] = material.specific_heat_capacity
        except AttributeError:
            pass
        try:
            kappa[k] = material.thermal_conductivity
        except AttributeError:
            pass

    material = problem.subdomain_materials[problem.wpi]
    mu = {problem.wpi: material.dynamic_viscosity}

    # Start time, end time, time step.
    t = 0.0
    T = 1200.0
    dt = 1.0e-4

    g = Constant((0.0, -9.81))

    # Initial states.
    u_1 = Function(problem.W, name='velocity')
    u_1.interpolate(Constant((0.0, 0.0, 0.0)))

    p_1 = Function(problem.P, name='pressure')
    p_1.interpolate(Constant(0.0))

    theta_1 = Function(problem.Q, name='temperature')
    theta_1.interpolate(Constant(problem.background_temp))

    with XDMFFile(mpi_comm_world(), 'boussinesq.xdmf') as xdmf_file:
        xdmf_file.parameters['flush_output'] = True
        xdmf_file.parameters['rewrite_function_mesh'] = False

        xdmf_file.write(u_1, t)
        xdmf_file.write(p_1, t)
        xdmf_file.write(theta_1, t)

        r = SpatialCoordinate(problem.mesh)[0]

        while t < T + DOLFIN_EPS:
            begin('Time step %e -> %e...' % (t, t+dt))

            begin('Computing heat...')
            # Stabilization in the workpiece.
            # rho_cp = rho[wpi](1550.0) * cp[wpi](1550.0)
            try:
                k = kappa[problem.wpi](1550.0)
            except TypeError:  # 'float' object is not callable
                k = kappa[problem.wpi]
            # tau = stab.supg2(
            #     Q.mesh(),  # TODO what to put here?
            #     u_1,
            #     k/rho_cp,
            #     Q.ufl_element().degree()
            #     )
            # u_tau = stab.supg(u_1,
            #                   k/rho_cp,
            #                   Q.ufl_element().degree()
            #                   )
            # Build right-hand side F of heat equation such that u' = F(u).

            def weak_F(t, u_t, trial, v):
                # TODO reevaluate
                # Don't use zero() or 0 to avoid errors as described in
                # <https://bitbucket.org/fenics-project/dolfin/issue/44/assemble-0-vectors>.
                # Use Expression instead of Constant to work around the error
                # <https://bitbucket.org/fenics-project/dolfin/issue/38/constant-expressions-dont-use-the-cell>.
                # Also, explicitly both RHS and LHS to something that doesn't
                # evaluate to an empty form.
                F = trial \
                    * Expression('0.0', cell=triangle) * v * 2*pi*r*dx(0) \
                    + Expression('0.0', cell=triangle) * v * 2*pi*r*dx(0)
                for i in subdomain_indices:
                    # Take all parameters at 1550K.
                    rho_cp = rho[i](1550.0) * cp[i](1550.0)
                    k = kappa[i](1550.0)
                    F -= k * r * dot(grad(trial), grad(v/rho_cp)) * 2*pi*dx(i)
                # Add convection.
                F -= dot(u_1, grad(trial)) * v * 2*pi*r*dx(problem.wpi)
                # # Add SUPG stabilization.
                # rho_cp = rho[wpi](background_temp)*cp[wpi]
                # k = kappa[wpi](background_temp)
                # Rdx = u_t * 2*pi*r*dx(wpi) \
                #     + dot(u_1, grad(trial)) * 2*pi*r*dx(wpi) \
                #     - 1.0/(rho_cp) * div(k*r*grad(trial)) * 2*pi*dx(wpi)
                # #F -= dot(tau*u_1, grad(v)) * Rdx
                # #F -= tau * inner(u_1, grad(v)) * 2*pi*r*dx(wpi)
                # #plot(tau, mesh=V.mesh(), title='u_tau')
                # #interactive()
                # #F -= tau * v * 2*pi*r*dx(wpi)
                # #F -= tau * Rdx
                return F
            theta = ts.implicit_euler_step(
                problem.Q,
                weak_F,
                theta_1,
                t, dt,
                sympy_bcs=problem.heater_bcs,
                tol=1.0e-12,
                lhs_multiplier=2*pi*r,
                verbose=False,
                form_compiler_parameters={
                    'quadrature_rule': 'vertex',
                    'quadrature_degree': 1,
                    },
                )
            end()

            # Do one Navier-Stokes time step.
            begin('Computing flux and pressure...')
            try:
                u, p = cyl_ns.ipcs_step(
                    problem.W, problem.P, dt,
                    mu[problem.wpi](problem.background_temp),
                    rho[problem.wpi](problem.background_temp),
                    u_1, problem.u_bcs,
                    p_1, problem.p_bcs,
                    f0=rho[problem.wpi](theta_1) * g,
                    f1=rho[problem.wpi](theta) * g,
                    theta=1.0,
                    stabilization=True,
                    tol=1.0e-10
                    )
            except RuntimeError as e:
                print(e.message)
                print('Navier--Stokes solver failed to converge. '
                      'Decrease time step from %e to %e and try again.' %
                      (dt, 0.5*dt)
                      )
                dt *= 0.5
                end()
                end()
                end()
                continue
            end()

            # Assignments and plotting.
            theta_1.assign(theta)
            u_1.assign(u)
            p_1.assign(p)

            # Save
            xdmf_file.write(theta_1, t+dt)
            xdmf_file.write(u_1, t+dt)
            xdmf_file.write(p_1, t+dt)

            plot(theta_1, title='temperature', rescale=True)
            plot(u_1, title='velocity', rescale=True)
            plot(p_1, title='pressure', rescale=True)
            # interactive()
            t += dt

            # Time update.
            begin('Step size adaptation...')
            u1, u2 = u_1.split()
            unorm = project(
                    abs(u1) + abs(u2),
                    problem.P,
                    form_compiler_parameters={'quadrature_degree': 4}
                    )
            unorm = norm(unorm.vector(), 'linf')
            # print('||u||_inf = %e' % unorm)
            unorm = max(unorm, DOLFIN_EPS)
            # http://scicomp.stackexchange.com/questions/2927/estimating-the-courant-number-for-the-navier-stokes-equations-under-differing-re
            rho_mu = rho[problem.wpi](problem.background_temp) \
                / mu[problem.wpi](problem.background_temp)
            target_dt = min(
                0.5*problem.mesh.hmin()/unorm,
                0.5*problem.mesh.hmin()**2 * rho_mu
                )
            print('previous dt: %e' % dt)
            print('target dt:  %e' % target_dt)
            # alpha is the aggressiveness factor. The distance between the
            # current step size and the target step size is reduced by
            # |1-alpha|. Hence, if alpha==1 then dt_next==target_dt. Otherwise
            # target_dt is approached more slowly.
            alpha = 0.5
            dt_max = 1.0e1
            dt = min(dt_max,
                     # At most double the step size from step to step.
                     dt * min(2.0, 1.0 + alpha*(target_dt - dt)/dt)
                     )
            print('new dt:    %e' % dt)
            end()
            end()
    return


if __name__ == '__main__':
    test()
