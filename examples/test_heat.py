#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from __future__ import print_function

import numpy
from dolfin import Constant, Function, Measure, XDMFFile, dx, plot, project

import maelstrom
import matplotlib.pyplot as plt
import parabolic
import problems


def _parameter_quest():
    """Find parameter sets fitting crucible data.
    """

    # Create the set of parameter values to search.
    flux0 = [500.0 * k for k in range(11)]
    flux1 = [500.0 * k for k in range(11)]
    from itertools import product

    search_space = product(flux0, flux1)

    control_values = [
        # bottom left
        ((0.0, 0.366), 1580.0),
        # top left
        ((0.0, 0.411), 1495.0),
        # top middle
        ((0.038, 0.411), 1511.0),
        # top right
        ((0.076, 0.411), 1540.0),
    ]
    tol = 20.0

    first = True
    best_match = None
    best_match_norm = None
    for p in search_space:
        print(p)
        # for boundary conditions
        flux = {"upper": p[0], "crucible": p[1]}

        theta = test_stationary_solve(flux)

        # Check if temperature values match with crucible blueprint.
        dev = numpy.array([theta(c[0]) - c[1] for c in control_values])
        print("Deviations from control temperatures:")
        print(dev)
        dev_norm = numpy.linalg.norm(dev)
        if not best_match or dev_norm < best_match_norm:
            print("New best match! {} (||dev|| = {:e})".format(p, dev_norm))
            best_match = p
            best_match_norm = dev_norm

        if all(abs(dev) < tol):
            print("Success! {}".format(p))
            print("Temperature at control points (with reference values):")
            for c in control_values:
                print(
                    "({:e}, {:e}):  {:e}   ({:e})".format(
                        c[0][0], c[0][1], theta(c[0]), c[1]
                    )
                )
        print()

        if first:
            theta_1 = theta
            first = False
        else:
            theta_1.assign(theta)
        plot(theta_1, rescale=True)
    return


def test_stationary_solve(show=False):

    problem = problems.Crucible()

    boundaries = problem.wp_boundaries

    average_temp = 1551.0

    material = problem.subdomain_materials[problem.wpi]
    rho = material.density(average_temp)
    cp = material.specific_heat_capacity
    kappa = material.thermal_conductivity

    my_ds = Measure("ds")(subdomain_data=boundaries)

    convection = None
    heat = maelstrom.heat.Heat(
        problem.Q,
        kappa,
        rho,
        cp,
        convection,
        source=Constant(0.0),
        dirichlet_bcs=problem.theta_bcs_d,
        neumann_bcs=problem.theta_bcs_n,
        robin_bcs=problem.theta_bcs_r,
        my_dx=dx,
        my_ds=my_ds,
    )
    theta_reference = heat.solve_stationary()
    theta_reference.rename("theta", "temperature")

    if show:
        # with XDMFFile('temperature.xdmf') as f:
        #     f.parameters['flush_output'] = True
        #     f.parameters['rewrite_function_mesh'] = False
        #     f.write(theta_reference)
        tri = plot(theta_reference)
        plt.colorbar(tri)
        plt.show()

    assert abs(maelstrom.helpers.average(theta_reference) - 1551.0) < 1.0e-1

    return theta_reference


def test_time_step():
    problem = problems.Crucible()

    boundaries = problem.wp_boundaries

    # The melting point of GaAs is 1511 K.
    average_temp = 1520.0

    f = Constant(0.0)

    material = problem.subdomain_materials[problem.wpi]
    rho = material.density(average_temp)
    cp = material.specific_heat_capacity
    kappa = material.thermal_conductivity

    my_ds = Measure("ds")(subdomain_data=boundaries)

    # from dolfin import DirichletBC
    convection = None
    heat = maelstrom.heat.Heat(
        problem.Q,
        kappa,
        rho,
        cp,
        convection,
        source=Constant(0.0),
        dirichlet_bcs=problem.theta_bcs_d,
        neumann_bcs=problem.theta_bcs_n,
        robin_bcs=problem.theta_bcs_r,
        my_dx=dx,
        my_ds=my_ds,
    )

    # create time stepper
    # stepper = parabolic.ExplicitEuler(heat)
    stepper = parabolic.ImplicitEuler(heat)
    # stepper = parabolic.Trapezoidal(heat)

    theta0 = project(Constant(average_temp), problem.Q)
    # theta0 = heat.solve_stationary()
    theta0.rename("theta0", "temperature")

    theta1 = Function(problem.Q)
    theta1 = Function(problem.Q)

    t = 0.0
    dt = 1.0e-3
    end_time = 10 * dt

    with XDMFFile("temperature.xdmf") as f:
        f.parameters["flush_output"] = True
        f.parameters["rewrite_function_mesh"] = False

        f.write(theta0, t)
        while t < end_time:
            theta1.assign(stepper.step(theta0, t, dt))
            theta0.assign(theta1)
            t += dt
            #
            f.write(theta0, t)

    assert abs(maelstrom.helpers.average(theta0) - 1519.81) < 1.0e-2

    return


if __name__ == "__main__":
    # # for boundary conditions
    # heat_transfer_coefficient = {
    #         'upper': 50.0,
    #         'upper left': 300.0,
    #         'crucible': 15.0
    #         }
    # T = {'upper': 1480.0,
    #      'upper left': 1500.0,
    #      'crucible': 1660.0}

    test_stationary_solve(show=True)
    # test_time_step()
