#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Navier-Stokes testbed.
"""
from __future__ import print_function

from dolfin import (
    parameters,
    Constant,
    Function,
    XDMFFile,
    DOLFIN_EPS,
    plot,
    project,
    norm,
    MPI,
)
import pytest

import maelstrom.navier_stokes as cyl_ns
import maelstrom.stokes as cyl_stokes

import problems

parameters["allow_extrapolation"] = True


@pytest.mark.parametrize(
    "problem",
    [
        problems.Lid_driven_cavity(),
        # problems.Rotating_lid(),
        problems.Ball_in_tube(),
    ],
)
def test(problem, max_num_steps=2, show=False):
    # # Density depends on temperature.
    # material = 'water'
    # rho = params[material]['density'](293.0)
    # mu = params[material]['dynamic viscosity'](293.0)

    rho = 1.0
    mu = 1.0

    # Start time, end time, time step.
    t = 0.0
    T = 8.0
    dt = 1.0e-5
    dt_max = 1.0e-1

    num_subspaces = problem.W.num_sub_spaces()

    if num_subspaces == 2:
        # g = Constant((0.0, 0.0))
        g = Constant((0.0, -9.81))
    elif num_subspaces == 3:
        # g = Constant((0.0, 0.0, 0.0))
        g = Constant((0.0, -9.81, 0.0))
    else:
        raise RuntimeError("Illegal number of subspaces ({}).".format(num_subspaces))

    initial_stokes = False
    if initial_stokes:
        u0, p0 = cyl_stokes.solve(
            problem.W,
            problem.P,
            mu,
            rho,
            problem.u_bcs,
            problem.p_bcs,
            f=rho * g,
            tol=1.0e-10,
            maxiter=2000,
        )
    else:
        # Initial states.
        u0 = Function(problem.W, name="velocity")
        u0.vector().zero()
        p0 = Function(problem.P, name="pressure")
        p0.vector().zero()

    filename = "navier_stokes.xdmf"
    with XDMFFile(MPI.comm_world, filename) as xdmf_file:
        xdmf_file.parameters["flush_output"] = True
        xdmf_file.parameters["rewrite_function_mesh"] = False

        if show:
            xdmf_file.write(u0, t)
            xdmf_file.write(p0, t)

        stepper = cyl_ns.IPCS(time_step_method="backward euler")
        steps = 0
        while t < T + DOLFIN_EPS and steps < max_num_steps:
            steps += 1
            print("Time step {:e} -> {:e}...".format(t, t + dt))
            try:
                u1, p1 = stepper.step(
                    Constant(dt),
                    {0: u0},
                    p0,
                    problem.W,
                    problem.P,
                    problem.u_bcs,
                    problem.p_bcs,
                    Constant(rho),
                    Constant(mu),
                    f={0: rho * g, 1: rho * g},
                    tol=1.0e-10,
                )
            except RuntimeError:
                print(
                    "Navier--Stokes solver failed to converge. "
                    "Decrease time step from {} to {} and try again.".format(
                        dt, 0.5 * dt
                    )
                )
                dt *= 0.5
                continue

            u0.assign(u1)
            p0.assign(p1)

            if show:
                # Save to files.
                xdmf_file.write(u0, t + dt)
                xdmf_file.write(p0, t + dt)
                # Plotting for some reason takes up a lot of memory.
                plot(u0, title="velocity", rescale=True)
                plot(p0, title="pressure", rescale=True)
                # interactive()

            print("Step size adaptation...")
            # unorm = project(abs(u[0]) + abs(u[1]) + abs(u[2]),
            #                 P,
            #                 form_compiler_parameters={'quadrature_degree': 4}
            #                 )
            unorm = project(
                norm(u1), problem.P, form_compiler_parameters={"quadrature_degree": 4}
            )
            unorm = norm(unorm.vector(), "linf")
            # print('||u||_inf = {:e}'.format(unorm))
            # Some smooth step-size adaption.
            target_dt = 0.2 * problem.mesh.hmax() / unorm
            print("current dt: {:e}".format(dt))
            print("target dt:  {:e}".format(target_dt))
            # alpha is the aggressiveness factor. The distance between the
            # current step size and the target step size is reduced by
            # |1-alpha|. Hence, if alpha==1 then dt_next==target_dt. Otherwise
            # target_dt is approached slowlier.
            alpha = 0.5
            dt = min(
                dt_max,
                # At most double the step size from step to step.
                dt * min(2.0, 1.0 + alpha * (target_dt - dt) / dt),
            )
            print("next dt:    {:e}".format(dt))
            t += dt
    return


if __name__ == "__main__":
    test(
        # problems.Lid_driven_cavity()
        # problems.Rotating_lid()
        problems.Ball_in_tube(),
        max_num_steps=1000,
        show=True,
    )
