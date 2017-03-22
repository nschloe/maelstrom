#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from maelstrom.message import Message


def poor_man(problem, state, parameter_value, target_value, solver):

    with Message('ISolve for parameter value %e' % parameter_value):
        solver.solve(problem, state.vector())

    delta = (target_value - parameter_value) * 1.0e-1
    while abs(target_value - parameter_value) > 1.0e-13:
        if abs(delta) > abs(target_value - parameter_value):
            # Don't overshoot
            delta = target_value - parameter_value
        parameter_value += delta
        print
        print
        with Message('Solve for parameter value %e (target: %e)'
                     % (parameter_value, target_value)):
            print
            problem.set_parameter(parameter_value)
            try:
                # Store the result in a temporary vector in case
                # the solve fails.
                tmp = state.vector().copy()
                solver.solve(problem, tmp)
                state.vector()[:] = tmp.copy()
                # Increase the step if it was successful
                delta *= 1.1
            except RuntimeError:
                # Try again with half the delta.
                parameter_value -= delta
                delta *= 0.5

    return
