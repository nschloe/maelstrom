#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
'''
Coupled solve of the Navier--Stokes and the heat equation
in cylindrical coordinates.
'''
from __future__ import print_function

from dolfin import (
    parameters, Constant, Function, XDMFFile, mpi_comm_world, DOLFIN_EPS,
    begin, end, project, plot, norm, as_vector, Measure, FunctionSpace
    )
import parabolic

import maelstrom
import maelstrom.navier_stokes as cyl_ns

import problems


# We need to allow extrapolation here since otherwise, the equation systems
# for Maxwell cannot be constructed: They contain the velocity `u` (from
# Navier-Stokes) which is only defined on the workpiece subdomain.
# Cf. <https://answers.launchpad.net/dolfin/+question/210508>.
parameters['allow_extrapolation'] = True

GMSH_EPS = 1.0e-15


def test(target_time=0.1):
    problem = problems.Crucible()

    # dx = Measure('dx', subdomain_data=problem.subdomains)

    material = problem.subdomain_materials[problem.wpi]
    rho = material.density
    cp = material.specific_heat_capacity
    kappa = material.thermal_conductivity
    mu = material.dynamic_viscosity

    average_temp = 1520.0

    # Start time, end time, time step.
    t = 0.0
    dt = 1.0e-4

    g = Constant((0.0, -9.81, 0.0))

    # Initial states.
    # TODO proper intitialization with StokesHeat
    u0 = Function(problem.W, name='velocity')
    u0.interpolate(Constant((0.0, 0.0, 0.0)))
    p0 = Function(problem.P, name='pressure')
    p0.interpolate(Constant(0.0))

    my_ds = Measure('ds')(subdomain_data=problem.wp_boundaries)

    heat_problem = maelstrom.heat.Heat(
                problem.Q,
                kappa, rho(average_temp), cp,
                convection=None,
                source=Constant(0.0),
                dirichlet_bcs=problem.theta_bcs_d,
                neumann_bcs=problem.theta_bcs_n,
                my_ds=my_ds
                )
    theta0 = heat_problem.solve_stationary()
    theta0.rename('theta', 'temperature')

    with XDMFFile(mpi_comm_world(), 'boussinesq.xdmf') as xdmf_file:
        xdmf_file.parameters['flush_output'] = True
        xdmf_file.parameters['rewrite_function_mesh'] = False

        xdmf_file.write(u0, t)
        xdmf_file.write(p0, t)
        xdmf_file.write(theta0, t)

        while t < target_time + DOLFIN_EPS:
            begin('Time step %e -> %e...' % (t, t+dt))

            begin('Computing heat...')

            heat_problem = maelstrom.heat.Heat(
                        problem.Q,
                        kappa, rho(average_temp), cp,
                        # Only take the first two components of the convection
                        convection=as_vector([u0[0], u0[1]]),
                        source=Constant(0.0),
                        # dirichlet_bcs=problem.theta_bcs_d_strict,
                        dirichlet_bcs=problem.theta_bcs_d,
                        neumann_bcs=problem.theta_bcs_n,
                        my_ds=my_ds
                        )
            stepper = parabolic.ImplicitEuler(heat_problem)
            theta1 = stepper.step(theta0, t, dt)
            end()

            # Do one Navier-Stokes time step.
            begin('Computing flux and pressure...')
            stepper = cyl_ns.IPCS(
                    time_step_method='backward euler',
                    stabilization=None
                    )
            try:
                u1, p1 = stepper.step(
                        dt,
                        {0: u0}, p0,
                        problem.W, problem.P,
                        problem.u_bcs, problem.p_bcs,
                        rho(problem.background_temp),
                        mu(problem.background_temp),
                        f={0: rho(theta0)*g, 1: rho(theta1)*g},
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
            theta0.assign(theta1)
            u0.assign(u1)
            p0.assign(p1)

            # Save
            xdmf_file.write(theta0, t+dt)
            xdmf_file.write(u0, t+dt)
            xdmf_file.write(p0, t+dt)

            plot(theta0, title='temperature', rescale=True)
            plot(u0, title='velocity', rescale=True)
            plot(p0, title='pressure', rescale=True)
            # interactive()
            t += dt

            # Time update.
            begin('Step size adaptation...')
            ux, uy, uz = u0.split()
            unorm = project(
                    abs(ux) + abs(uy) + abs(uz),
                    problem.P,
                    form_compiler_parameters={'quadrature_degree': 4}
                    )
            unorm = norm(unorm.vector(), 'linf')
            # print('||u||_inf = %e' % unorm)
            unorm = max(unorm, DOLFIN_EPS)
            # http://scicomp.stackexchange.com/questions/2927/estimating-the-courant-number-for-the-navier-stokes-equations-under-differing-re
            rho_mu = rho(problem.background_temp) / mu(problem.background_temp)
            target_dt = min(
                0.5*problem.mesh.hmin() / unorm,
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
    test(target_time=1200.0)
