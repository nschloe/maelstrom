#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Full* simulation of the melt problem.

* Conditions apply.
  1. The effect of the melt movement on the magnetic field is neglected.
     The Lorentz force is precomputed.
  2. The heat transfer from the coils to the melt is not simulated.
     Temperature boundary conditions are set at the melt boundary directly.

A worthwhile read in for the simulation of crystal growth is :cite:`Derby89`.
"""
import warnings

import numpy
import pytest
from dolfin import (
    DOLFIN_EPS,
    MPI,
    Constant,
    FunctionSpace,
    Measure,
    XDMFFile,
    as_vector,
    assemble,
    dx,
    interpolate,
    norm,
    parameters,
    plot,
)
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from numpy import pi

import maelstrom.heat as cyl_heat
import maelstrom.navier_stokes as cyl_ns
import maelstrom.stokes_heat as stokes_heat
import matplotlib.pyplot as plt
import parabolic
import problems
from maelstrom.helpers import average
from test_maxwell import get_lorentz_joule

# Ignore the deprecation warning, see
# https://www.allanswered.com/post/lknbq/assemble-quadrature-representation-vs-uflacs/
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)

# We need to allow extrapolation here since otherwise, the equation systems
# for Maxwell cannot be constructed: They contain the velocity `u` (from
# Navier-Stokes) which is only defined on the workpiece subdomain.
# Cf. <https://answers.launchpad.net/dolfin/+question/210508>.
parameters["allow_extrapolation"] = True
parameters["std_out_all_processes"] = False


def _construct_initial_state(
    mesh,
    W_element,
    P_element,
    Q_element,
    kappa,
    cp,
    rho,
    mu,
    heat_source,
    u_bcs,
    p_bcs,
    theta_dirichlet_bcs,
    theta_neumann_bcs,
    dx_submesh,
    ds_submesh,
    g,
    extra_force,
):
    """Construct an initial state for the Navier-Stokes simulation.
    """
    Q = FunctionSpace(mesh, Q_element)

    # Finding a steady-state solution of the coupled Stokes-Heat problem hasn't
    # been successfull -- perhaps because there is no steady state. A
    # reasonable approach is to first solve the heat equation without
    # convection, then the Stokes problem with the uplift force from the
    # heat and density distribution.

    # initial guess
    theta_average = 1530.0
    theta0 = interpolate(Constant(theta_average), Q)

    kappa_const = kappa if isinstance(kappa, float) else kappa(theta_average)
    mu_const = mu if isinstance(mu, float) else mu(theta_average)
    cp_const = cp if isinstance(cp, float) else cp(theta_average)

    u0, p0, theta0 = stokes_heat.solve_fixed_point(
        mesh,
        W_element,
        P_element,
        Q_element,
        theta0,
        kappa_const,
        rho,
        mu_const,
        cp_const,
        g,
        extra_force,
        heat_source,
        u_bcs,
        p_bcs,
        theta_dirichlet_bcs,
        theta_neumann_bcs,
        my_dx=dx_submesh,
        my_ds=ds_submesh,
        max_iter=1,
        tol=1.0e-8,
    )

    # u0, p0, theta0 = stokes_heat.solve(
    #     mesh, W_element, P_element, Q_element,
    #     u0, p0, theta0,
    #     kappa_wpi_const, rho_wpi, mu_wpi_const, cp_wpi_const,
    #     g, extra_force,
    #     joule_wpi,
    #     u_bcs, p_bcs,
    #     theta_bcs_d,
    #     theta_bcs_n,
    #     dx_submesh, ds_submesh
    #     )

    u0.rename("u", "velocity")
    p0.rename("p", "pressure")
    theta0.rename("theta", "temperature")

    # Create a *deep* copy of u0, p0, to be able to deal with them as actually
    # separate entities.
    # u0, p0 = up0.split(deepcopy=True)
    return u0, p0, theta0


def _store(outfile, u, p, theta, t):
    outfile.write(u, t)
    outfile.write(p, t)
    outfile.write(theta, t)
    return


def _plot(p, theta):
    plot(theta, title="temperature", rescale=True)
    # matplotlib can only show 2d plots; this does not include 3d vector fields
    # plot(u, title='velocity', rescale=True)
    plot(p, title="pressure", rescale=True)
    return


@pytest.mark.parametrize("with_voltage", [False, True])
def test_boussinesq(with_voltage, target_time=1.0e-2, show=False):
    """Simple boussinesq test; no Maxwell involved.
    """
    problem = problems.Crucible()

    # Solve construct initial state without Lorentz force and Joule heat.
    m = problem.subdomain_materials[problem.wpi]
    k_wpi = m.thermal_conductivity
    cp_wpi = m.specific_heat_capacity
    rho_wpi = m.density
    mu_wpi = m.dynamic_viscosity

    g = Constant((0.0, -9.80665, 0.0))
    submesh_workpiece = problem.W.mesh()
    ds_workpiece = Measure("ds", subdomain_data=problem.wp_boundaries)
    u0, p0, theta0 = _construct_initial_state(
        submesh_workpiece,
        problem.W_element,
        problem.P_element,
        problem.Q_element,
        k_wpi,
        cp_wpi,
        rho_wpi,
        mu_wpi,
        Constant(0.0),
        problem.u_bcs,
        problem.p_bcs,
        problem.theta_bcs_d,
        problem.theta_bcs_n,
        dx(submesh_workpiece),
        ds_workpiece,
        g,
        extra_force=None,
    )

    if with_voltage:
        voltages = [
            38.0 * numpy.exp(-1j * 2 * pi * 2 * 70.0 / 360.0),
            38.0 * numpy.exp(-1j * 2 * pi * 1 * 70.0 / 360.0),
            38.0 * numpy.exp(-1j * 2 * pi * 0 * 70.0 / 360.0),
            25.0 * numpy.exp(-1j * 2 * pi * 0 * 70.0 / 360.0),
            25.0 * numpy.exp(-1j * 2 * pi * 1 * 70.0 / 360.0),
        ]
        lorentz, joule, _ = get_lorentz_joule(problem, voltages, show=show)
    else:
        lorentz = None
        joule = Constant(0.0)

    u1, _, theta1 = _compute_boussinesq(
        problem,
        u0,
        p0,
        theta0,
        lorentz=lorentz,
        joule=joule,
        target_time=target_time,
        show=show,
    )

    if with_voltage:
        # p is only defined up to a constant
        ref = 0.001069306823450495
        assert abs(norm(u1, "L2") - ref) < 1.0e-3 * ref
        ref = 86.96271240592199
        assert abs(norm(theta1, "L2") - ref) < 1.0e-3 * ref
    else:
        ref = 0.001069283700624996
        assert abs(norm(u1, "L2") - ref) < 1.0e-3 * ref
        ref = 86.96274092197706
        assert abs(norm(theta1, "L2") - ref) < 1.0e-3 * ref

    return


def _compute_boussinesq(
    problem, u0, p0, theta0, lorentz, joule, target_time=0.1, show=False
):
    # Define a facet measure on the boundaries. See discussion on
    # <https://bitbucket.org/fenics-project/dolfin/issue/249/facet-specification-doesnt-work-on-ds>.
    ds_workpiece = Measure("ds", subdomain_data=problem.wp_boundaries)

    submesh_workpiece = problem.W.mesh()

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
    hmax_workpiece = MPI.max(submesh_workpiece.mpi_comm(), submesh_workpiece.hmax())

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

    theta_average = average(theta0)

    # show_total_force = True
    # if show_total_force:
    #     f = rho_wpi(theta0) * g
    #     if lorentz:
    #         f += as_vector((lorentz[0], lorentz[1], 0.0))
    #     tri = plot(f, mesh=submesh_workpiece, title='Total external force')
    #     plt.colorbar(tri)
    #     plt.show()

    with XDMFFile(submesh_workpiece.mpi_comm(), "all.xdmf") as outfile:
        outfile.parameters["flush_output"] = True
        outfile.parameters["rewrite_function_mesh"] = False

        _store(outfile, u0, p0, theta0, t)
        if show:
            _plot(p0, theta0)
            plt.show()

        successful_steps = 0
        failed_steps = 0
        while t < target_time + DOLFIN_EPS:
            print(
                "Successful steps: {}    (failed: {}, total: {})".format(
                    successful_steps, failed_steps, successful_steps + failed_steps
                )
            )
            print("Time step {:e} -> {:e}...".format(t, t + dt))
            # Do one heat time step.
            print("Computing heat...")
            # Redefine the heat problem with the new u0.
            heat_problem = cyl_heat.Heat(
                problem.Q,
                kappa=k_wpi,
                rho=rho_wpi(theta_average),
                cp=cp_wpi,
                convection=u0,
                source=joule,
                dirichlet_bcs=problem.theta_bcs_d,
                neumann_bcs=problem.theta_bcs_n,
                my_dx=dx(submesh_workpiece),
                my_ds=ds_workpiece,
            )

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
            heat_stepper = parabolic.ImplicitEuler(heat_problem)

            ns_stepper = cyl_ns.IPCS(time_step_method="backward euler")
            theta1 = heat_stepper.step(theta0, t, dt)

            theta0_average = average(theta0)
            try:
                # Do one Navier-Stokes time step.
                print("Computing flux and pressure...")
                # Include proper temperature-dependence here to account
                # for the Boussinesq effect.
                f0 = rho_wpi(theta0) * g
                f1 = rho_wpi(theta1) * g
                if lorentz is not None:
                    f = as_vector((lorentz[0], lorentz[1], 0.0))
                    f0 += f
                    f1 += f
                u1, p1 = ns_stepper.step(
                    Constant(dt),
                    {0: u0},
                    p0,
                    problem.W,
                    problem.P,
                    problem.u_bcs,
                    problem.p_bcs,
                    # Make constant TODO
                    Constant(rho_wpi(theta0_average)),
                    Constant(mu_wpi(theta0_average)),
                    f={0: f0, 1: f1},
                    tol=1.0e-10,
                    my_dx=dx(submesh_workpiece),
                )
            except RuntimeError as e:
                print(e.args[0])
                print(
                    "Navier--Stokes solver failed to converge. "
                    "Decrease time step from {:e} to {:e} and try again.".format(
                        dt, 0.5 * dt
                    )
                )
                dt *= 0.5
                failed_steps += 1
                continue
            successful_steps += 1

            # Assignments and plotting.
            theta0.assign(theta1)
            u0.assign(u1)
            p0.assign(p1)

            _store(outfile, u0, p0, theta0, t + dt)
            if show:
                _plot(p0, theta0)
                plt.show()

            t += dt
            print("Diagnostics...")
            # Print some general info on the flow in the crucible.
            umax = get_umax(u0)
            _print_diagnostics(
                theta0,
                umax,
                submesh_workpiece,
                wpi_area,
                problem.subdomain_materials,
                problem.wpi,
                rho_wpi,
                mu_wpi,
                char_length,
                grav,
            )
            print()
            print("Step size adaptation...")
            # Some smooth step-size adaption.
            target_dt = 0.2 * hmax_workpiece / umax
            print("previous dt: {:e}".format(dt))
            print("target dt: {:e}".format(target_dt))
            # agg is the aggressiveness factor. The distance between
            # the current step size and the target step size is reduced
            # by |1-agg|. Hence, if agg==1 then dt_next==target_dt.
            # Otherwise target_dt is approached more slowly.
            agg = 0.5
            dt = min(
                dt_max,
                # At most double the step size from step to step.
                dt * min(2.0, 1.0 + agg * (target_dt - dt) / dt),
            )
            print("new dt:    {:e}".format(dt))
            print()
            print()
    return u0, p0, theta0


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
    return numpy.sqrt(norm(vec, "linf"))


def _print_diagnostics(
    theta0,
    umax,
    submesh_workpiece,
    wpi_area,
    subdomain_materials,
    wpi,
    rho,
    mu,
    char_length,
    grav,
):
    av_temperature = assemble(theta0 * dx(submesh_workpiece)) / wpi_area
    vec = theta0.vector()
    temperature_difference = vec.max() - vec.min()
    print("Max temperature: {:e}".format(vec.max()))
    print("Min temperature: {:e}".format(vec.min()))
    print("Av  temperature: {:e}".format(av_temperature))
    print()
    print("Max velocity: {:e}".format(umax))
    print()
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
    print("Prandtl number: {:e}".format(_get_prandtl(mu_const, rho_const, cp, k)))
    print(
        "Reynolds number: {:e}".format(
            _get_reynolds(rho_const, mu_const, char_length, char_velocity)
        )
    )
    print(
        "Grashof number: {:e}".format(
            _get_grashof(
                rho, mu_const, grav, av_temperature, char_length, temperature_difference
            )
        )
    )
    return


def _get_prandtl(mu, rho, cp, k):
    """Prandtl number.
    """
    nu = mu / rho
    return nu * rho * cp / k


def _get_reynolds(rho, mu, char_length, char_velocity):
    """Reynolds number.
    """
    return rho * char_velocity * char_length / mu


def _get_grashof(rho, mu, grav, theta_average, char_length, deltaT):
    """Grashof number.
    """
    # Volume expansion coefficient: alpha = -V'(T)/V(T).
    # Approximate V'(T).
    dT = 1.0e-2
    vpt = (rho(theta_average + dT) - rho(theta_average - dT)) / (2 * dT)
    volume_expansion = -vpt / rho(theta_average)
    nu = mu / rho(theta_average)
    return volume_expansion * deltaT * char_length ** 3 * grav / nu ** 2


if __name__ == "__main__":
    test_boussinesq(target_time=60.0, with_voltage=True, show=False)
