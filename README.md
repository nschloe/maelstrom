# maelstrom

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/maelstrom.svg)](https://circleci.com/gh/nschloe/maelstrom)
[![codecov](https://codecov.io/gh/nschloe/maelstrom/branch/master/graph/badge.svg)](https://codecov.io/gh/nschloe/maelstrom)
[![Documentation Status](https://readthedocs.org/projects/maelstrom/badge/?version=latest)](https://readthedocs.org/projects/maelstrom/?badge=latest)

maelstrom is a numerical software tool for the solution of magnetohydrodynamics
problems in cylindrical coordinates.
As such, maelstrom includes time integrators for the heat equation, for the
Navier--Stokes equations, and a stationary solver for the Maxwell equations,
each in cylindrial coordinates.

### Some details on the problem

The goal is to compute the flux of a liquid metal under the influence of a
magnetic field, modeled by

  * the heat equation,
  * Maxwell's equations, and
  * the Navier-Stokes equations.

Heat and Navier-Stokes are coupled by buoyancy, heat and Maxwell by the Joule
effect, and Maxwell and Navier-Stokes by current induction and the Lorentz
force.

To simplify matters, it is assumed that the effect of the material flux does
not influence the electric and magnetic fields, i.e., the current induction
from moving molten metal in a magnetic field is neglected. This decouples
Maxwell's equations from the other two. Essentially, the task breaks down to

 * computing Joule heating and Lorentz force, given a voltage distribution in
   coils, and given those two quantities
 * computing the the resulting material flux inside a container.

### Solving Maxwell's equations

Derivation of the involved formulas is best taken from [the
documentation](https://maelstrom.readthedocs.io/en/master/maelstrom.maxwell.html).




### Testing

To run the voropy unit tests, check out this repository and type
```
pytest
```

### License

maelstrom is published under the MIT license. See the file LICENSE for detailed
information.
