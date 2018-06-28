# -*- coding: utf-8 -*-
#
import warnings

from dolfin import (
    dx,
    ds,
    dot,
    grad,
    pi,
    assemble,
    lhs,
    rhs,
    SpatialCoordinate,
    TrialFunction,
    TestFunction,
    KrylovSolver,
    Function,
    assemble_system,
    LUSolver,
    div,
    as_vector,
)
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

from . import stabilization as stab

# Ignore the deprecation warning, see
# https://www.allanswered.com/post/lknbq/assemble-quadrature-representation-vs-uflacs/
warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)


def F(
    u,
    v,
    kappa,
    rho,
    cp,
    convection,
    source,
    r,
    neumann_bcs,
    robin_bcs,
    my_dx,
    my_ds,
    stabilization,
):
    """
    Compute

    .. math::

        F(u) =
            \\int_\\Omega \\kappa r
                \\langle\\nabla u, \\nabla \\frac{v}{\\rho c_p}\\rangle
                \\, 2\\pi \\, \\text{d}x
            + \\int_\\Omega \\langle c, \\nabla u\\rangle v
                \\, 2\\pi r\\,\\text{d}x
            - \\int_\\Omega \\frac{1}{\\rho c_p} f v
                \\, 2\\pi r \\,\\text{d}x\\\\
            - \\int_\\Gamma r \\kappa \\langle n, \\nabla T\\rangle v
                \\frac{1}{\\rho c_p} 2\\pi \\,\\text{d}s
            - \\int_\\Gamma  r \\kappa  \\alpha (u - u_0) v
                \\frac{1}{\\rho c_p} \\, 2\\pi \\,\\text{d}s,

    used for time-stepping

    .. math::

        u' = F(u).
    """
    rho_cp = rho * cp

    F0 = kappa * r * dot(grad(u), grad(v / rho_cp)) * 2 * pi * my_dx

    # F -= dot(b, grad(u)) * v * 2*pi*r * dx_workpiece(0)
    if convection is not None:
        c = as_vector([convection[0], convection[1]])
        F0 += dot(c, grad(u)) * v * 2 * pi * r * my_dx

    # Joule heat
    F0 -= source * v / rho_cp * 2 * pi * r * my_dx

    # Neumann boundary conditions
    for k, n_grad_T in neumann_bcs.items():
        F0 -= r * kappa * n_grad_T * v / rho_cp * 2 * pi * my_ds(k)

    # Robin boundary conditions
    for k, value in robin_bcs.items():
        alpha, u0 = value
        F0 -= r * kappa * alpha * (u - u0) * v / rho_cp * 2 * pi * my_ds(k)

    if stabilization == "supg":
        # Add SUPG stabilization.
        assert convection is not None
        # TODO u_t?
        R = (
            -div(kappa * r * grad(u)) / rho_cp * 2 * pi
            + dot(c, grad(u)) * 2 * pi * r
            - source / rho_cp * 2 * pi * r
        )
        mesh = v.function_space().mesh()
        element_degree = v.ufl_element().degree()
        tau = stab.supg(mesh, convection, kappa, element_degree)
        F0 += R * tau * dot(convection, grad(v)) * my_dx
    else:
        assert stabilization is None

    return F0


class Heat(object):
    """
    Class for interfacing the parabolic library for time stepping.

    Note that the mass matrix :math:`M` is computed using a vertex quadrature
    scheme such that it does not have off-diagonal entries. The usual form with
    positive off-diagonal entries makes the mass matrix a non-M-matrix, leading
    to oscillations whenever the temperature gradient is sharp. See
    :cite:`GR2007` for background.
    """

    def __init__(
        self,
        Q,
        kappa,
        rho,
        cp,
        convection,
        source,
        dirichlet_bcs=None,
        neumann_bcs=None,
        robin_bcs=None,
        my_dx=dx,
        my_ds=ds,
        stabilization=None,
    ):
        super(Heat, self).__init__()
        self.Q = Q

        dirichlet_bcs = dirichlet_bcs or []
        neumann_bcs = neumann_bcs or {}
        robin_bcs = robin_bcs or {}

        self.convection = convection

        u = TrialFunction(Q)
        v = TestFunction(Q)

        # If there are sharp temperature gradients, numerical oscillations may
        # occur. This happens because the resulting matrix is not an M-matrix,
        # caused by the fact that A1 puts positive elements in places other
        # than the main diagonal. To prevent that, it is suggested by
        # Großmann/Roos to use a vertex-centered discretization for the mass
        # matrix part.
        # Check
        # https://bitbucket.org/fenics-project/ffc/issues/145/uflacs-error-for-vertex-quadrature-scheme
        #
        self.M = assemble(
            u * v * dx,
            form_compiler_parameters={
                "representation": "quadrature",
                "quadrature_rule": "vertex",
            },
        )

        mesh = Q.mesh()
        r = SpatialCoordinate(mesh)[0]
        self.F0 = F(
            u,
            v,
            kappa,
            rho,
            cp,
            convection,
            source,
            r,
            neumann_bcs,
            robin_bcs,
            my_dx,
            my_ds,
            stabilization,
        )

        self.dirichlet_bcs = dirichlet_bcs

        self.A, self.b = assemble_system(-lhs(self.F0), rhs(self.F0))
        return

    # pylint: disable=unused-argument
    def eval_alpha_M_beta_F(self, alpha, beta, u, t):
        """Evaluate  :code:`alpha * M * u + beta * F(u, t)`.
        """
        uvec = u.vector()
        # Convert to proper `float`s to avoid accidental conversion to
        # numpy.arrays, cf.
        # <https://bitbucket.org/fenics-project/dolfin/issues/874/genericvector-numpyfloat-numpyarray-not>
        alpha = float(alpha)
        beta = float(beta)
        return alpha * (self.M * uvec) + beta * (self.A * uvec + self.b)

    def solve_alpha_M_beta_F(self, alpha, beta, b, t):
        """Solve  :code:`alpha * M * u + beta * F(u, t) = b`  with Dirichlet
        conditions.
        """
        matrix = alpha * self.M + beta * self.A

        # See above for float conversion
        right_hand_side = -float(beta) * self.b.copy()
        if b:
            right_hand_side += b

        for bc in self.dirichlet_bcs:
            bc.apply(matrix, right_hand_side)

        # TODO proper preconditioner for convection
        if self.convection:
            # Use HYPRE-Euclid instead of ILU for parallel computation.
            # However, this PC sometimes fails.
            # solver = KrylovSolver('gmres', 'hypre_euclid')
            # Fallback:
            solver = LUSolver()
        else:
            solver = KrylovSolver("gmres", "hypre_amg")
            solver.parameters["relative_tolerance"] = 1.0e-13
            solver.parameters["absolute_tolerance"] = 0.0
            solver.parameters["maximum_iterations"] = 100
            solver.parameters["monitor_convergence"] = True

        solver.set_operator(matrix)

        u = Function(self.Q)
        solver.solve(u.vector(), right_hand_side)
        return u

    def solve_stationary(self):
        """Solve the stationary problem :code:`F(u, t) = 0`  with Dirichlet
        conditions.
        """
        return self.solve_alpha_M_beta_F(alpha=0.0, beta=1.0, b=None, t=0.0)
