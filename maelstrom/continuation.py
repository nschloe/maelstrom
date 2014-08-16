#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of Maelstrom.
#
#  Maelstrom is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Maelstrom is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Maelstrom.  If not, see <http://www.gnu.org/licenses/>.
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
            except RuntimeError:
                # Try again with half the delta.
                parameter_value -= delta
                delta *= 0.5

    return
