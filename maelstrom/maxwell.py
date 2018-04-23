# -*- coding: utf-8 -*-
#
'''
.. math::
    \\DeclareMathOperator{\\div}{div}
    \\DeclareMathOperator{\\curl}{curl}

The equation system defined here are largely based on :cite:`Cha97`, with the
addition of the material flux :math:`u` of the liquid.

Given an electric field :math:`E`, a magnetic field :math:`B`, and a material
flux :math:`u`, the current density :math:`J` in the material is given
by

.. math::
     J = \\sigma (E + u \\times B).

where :math:`\\sigma` is the electrical conductivity.

This leads to

.. math::
     \\curl(\\sigma^{-1} J - u\\times B + \\text{i} \\omega A) = 0.

Assuming that :math:`B` is given by the potential :math:`\\phi`,
:math:`B = \\curl(\\phi e_{\\theta})`, we end up with

.. math::
    u \\times B &= u \\times \\curl(\\phi e_{\\theta}) \\\\
                &= u \\times \\left(
                    - \\frac{\\text{d}\\phi}{\\text{d}z} e_r
                    + \\frac{1}{r} \\frac{\\text{d}(r\\phi)}{\\text{d}r} e_z
                    \\right) \\\\
                &= - u_z \\frac{\\text{d}\\phi}{\\text{d}z} e_{\\theta}
                   + u_{\\theta} \\frac{\\text{d}\\phi}{\\text{d}z} e_z
                   - u_r \\frac{1}{r}
                     \\frac{\\text{d}(r\\phi)}{\\text{d}r} e_{\\theta}
                   + u_{\\theta} \\frac{1}{r}
                     \\frac{\\text{d}(r\\phi)}{\\text{d}r} e_r.

(Note that :math:`\\curl` is taken in cylindrial coordinates.)

Following Chaboudez, this eventually leads to the equation system

.. math::
    \\begin{cases}
    - \\div\\left(\\frac{1}{\\mu r} \\nabla(r\\phi)\\right)
    - \\sigma u_\\theta \\div\\left(\\frac{1}{r}\\nabla(r\\phi)\\right)
    + \\sigma \\left\\langle
          (u_r, u_z)^T,
          \\frac{1}{r}\\nabla(r\\phi)
          \\right\\rangle
    + \\text{i} \\sigma \\omega \\phi
    = \\frac{\\sigma v_k}{2\\pi r}    \\quad\\text{in } \\Omega,\\\\
    n\\cdot\\left(- \\frac{1}{\\mu r} \\nabla(r\\phi)\\right) = 0
      \\quad\\text{on }\\Gamma \\setminus \\{r=0\\}\\\\
    \\phi = 0    \\quad\\text{ for } r=0.
    \\end{cases}

The differential operators are interpreted like 2D for :math:`r` and :math:`z`.

For the weak formulation, the volume elements :math:`2\\pi r\\,\\text{d}x` are
used. This corresponds to the full 3D rotational formulation and also makes
sure that at least the diffusive term is nice and symmetric. Additionally, it
avoids dividing by :math:`r` in the convections and the right hand side.

Here with no convection in direction :math:`\\theta`:

.. math::
       \\int_\\Omega
           \\div\\left(\\frac{1}{\\mu r} \\nabla(r u)\\right) (2\\pi r v)
     + \\langle b, \\nabla(r u)\\rangle 2\\pi v
     + \\text{i} \\sigma \\omega u 2 \\pi r v
   = \\int_\\Omega \\sigma v_k v.
'''
from dolfin import (
    info, DOLFIN_EPS, DirichletBC, Function, KrylovSolver, dot, grad, pi,
    TrialFunctions, TestFunctions, assemble, Constant, project, FunctionSpace,
    SpatialCoordinate, mpi_comm_world
    )
import numpy


def solve(V, dx,
          Mu, Sigma,  # dictionaries
          omega,
          f_list,  # list of dictionaries
          convections,  # dictionary
          f_degree=None,
          bcs=None,
          tol=1.0e-12,
          verbose=False):
    '''Solve the complex-valued time-harmonic Maxwell system in 2D cylindrical
    coordinates

    .. math::
         \\div\\left(\\frac{1}{\\mu r} \\nabla(r\\phi)\\right)
         + \\left\\langle u, \\frac{1}{r} \\nabla(r\\phi)\\right\\rangle
         + i \\sigma \\omega \\phi
            = f

    with finite elements.

    :param V: function space for potentials

    :param dx: measure

    :param Mu: mu per subdomain
    :type Mu: dictionary

    :param Sigma: sigma per subdomain
    :type Sigma: dictionary

    :param omega: current frequency
    :type omega: float

    :param f_list: list of right-hand sides for each of which a solution will
                   be computed

    :param convections: convection, defined per subdomain
    :type convections: dictionary

    :param bcs: Dirichlet boundary conditions

    :param tol: relative solver tolerance
    :type tol: float

    :param verbose: solver verbosity
    :type verbose: boolean

    :rtype: list of functions
    '''
    # For the exact solution of the magnetic scalar potential, see
    # <http://www.physics.udel.edu/~jim/PHYS809_10F/Class_Notes/Class_26.pdf>.
    # Here, the value of \\phi along the rotational axis is specified as
    #
    #    phi(z) = 2 pi I / c * (z/|z| - z/sqrt(z^2 + a^2))
    #
    # where 'a' is the radius of the coil. This expression contradicts what is
    # specified by [Chaboudez97]_ who claim that phi=0 is the natural value
    # at the symmetry axis.
    #
    # For more analytic expressions, see
    #
    #     Simple Analytic Expressions for the Magnetic Field of a Circular
    #     Current Loop;
    #     James Simpson, John Lane, Christopher Immer, and Robert Youngquist;
    #     <http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20010038494_2001057024.pdf>.
    #

    # Check if boundary conditions on phi are explicitly provided.
    if not bcs:
        # Create Dirichlet boundary conditions.
        # In the cylindrically symmetric formulation, the magnetic vector
        # potential is given by
        #
        #    A = e^{i omega t} phi(r,z) e_{theta}.
        #
        # It is natural to demand phi=0 along the symmetry axis r=0 to avoid
        # discontinuities there.
        # Also, this makes sure that the system is well-defined (see comment
        # below).
        #
        def xzero(x, on_boundary):
            return on_boundary and abs(x[0]) < DOLFIN_EPS
        ee = V.ufl_element() * V.ufl_element()
        VV = FunctionSpace(V.mesh(), ee)
        bcs = DirichletBC(VV, (0.0, 0.0), xzero)
        #
        # Concerning the boundary conditions for the rest of the system:
        # At the other boundaries, it is not uncommon (?) to set so-called
        # impedance boundary conditions; see, e.g.,
        #
        # Chaboudez et al.,
        # Numerical Modeling in Induction Heating for Axisymmetric
        # Geometries,
        # IEEE Transactions on Magnetics, vol. 33, no. 1, Jan 1997,
        # <http://www.esi-group.com/products/casting/publications/Articles_PDF/InductionaxiIEEE97.pdf>.
        #
        # or
        #
        # <ftp://ftp.math.ethz.ch/pub/sam-reports/reports/reports2010/2010-39.pdf>.
        #
        # TODO review those, references don't seem to be too accurate
        # Those translate into Robin-type boundary conditions (and are in fact
        # sometimes called that, cf.
        # https://en.wikipedia.org/wiki/Robin_boundary_condition).
        # The classical reference is
        #
        #   Impedance boundary conditions for imperfectly conducting surfaces,
        #   T.B.A. Senior,
        #   <http://link.springer.com/content/pdf/10.1007/BF02920074>.
        #
        # class OuterBoundary(SubDomain):
        #     def inside(self, x, on_boundary):
        #         return on_boundary and abs(x[0]) > DOLFIN_EPS
        # boundaries = MeshFunction(
        #     'size_t', mesh,
        #     mesh.topology().dim() - 1
        #     )
        # boundaries.set_all(0)
        # outer_boundary = OuterBoundary()
        # outer_boundary.mark(boundaries, 1)
        # ds = Measure('ds')[boundaries]
        # #n = FacetNormal(mesh)
        # #a += - 1.0/Mu[i] * dot(grad(r*ur), n) * vr * ds(1) \
        # #     - 1.0/Mu[i] * dot(grad(r*ui), n) * vi * ds(1)
        # #L += - 1.0/Mu[i] * 1.0 * vr * ds(1) \
        # #     - 1.0/Mu[i] * 1.0 * vi * ds(1)
        # # This is -n.grad(r u) = u:
        # a += 1.0/Mu[i] * ur * vr * ds(1) \
        #    + 1.0/Mu[i] * ui * vi * ds(1)

    # Create the system matrix, preconditioner, and the right-hand sides.
    # For preconditioners, there are two approaches. The first one, described
    # in
    #
    #     Algebraic Multigrid for Complex Symmetric Systems;
    #     D. Lahaye, H. De Gersem, S. Vandewalle, and K. Hameyer;
    #     <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=877730>
    #
    # doesn't work too well here.
    # The matrix P, created in build_system(), provides a better alternative.
    # For more details, see documentation in build_system().
    #
    A, P, b_list, _, W = build_system(
        V, dx,
        Mu, Sigma,
        omega,
        f_list,
        f_degree,
        convections,
        bcs
        )

    # prepare solver
    # Don't use 'amg', since that defaults to `ml_amg` if available which
    # crashes
    # <https://bitbucket.org/fenics-project/docker/issues/61/petsc-vectorfunctionspace-amg-malloc>.
    solver = KrylovSolver('gmres', 'hypre_amg')
    solver.set_operators(A, P)

    # The PDE for A has huge coefficients (order 10^8) all over. Hence, if
    # relative residual is set to 10^-6, the actual residual will still be of
    # the order 10^2. While this isn't too bad (after all the equations are
    # upscaled by a large factor), one can choose a very small relative
    # tolerance here to get a visually pleasing residual norm.
    solver.parameters['relative_tolerance'] = tol
    solver.parameters['absolute_tolerance'] = 0.0
    solver.parameters['maximum_iterations'] = 100
    solver.parameters['report'] = verbose
    solver.parameters['monitor_convergence'] = verbose

    phi_list = []
    for k, b in enumerate(b_list):
        phi_list.append(Function(W))
        phi_list[-1].rename('phi{}'.format(k), 'phi{}'.format(k))
        solver.solve(phi_list[-1].vector(), b)

    return phi_list


def build_system(V, dx,
                 Mu, Sigma,
                 omega,
                 f_list,
                 f_degree,
                 convections,
                 bcs):
    '''Build FEM system for

    .. math::
         \\div\\left(\\frac{1}{\\mu r} \\nabla(r\\phi)\\right)
         + \\left\\langle u, \\frac{1}{r} \\nabla(r\\phi)\\right\\rangle
         + \\text{i} \\sigma \\omega \\phi
            = f

    by multiplying with :math:`2\\pi r v` and integrating over the domain and
    the preconditioner given by :cite:`KL2012`.
    '''
    r = SpatialCoordinate(V.mesh())[0]

    subdomain_indices = Mu.keys()

    ee = V.ufl_element() * V.ufl_element()
    W = FunctionSpace(V.mesh(), ee)

    # Bilinear form.
    ur, ui = TrialFunctions(W)
    vr, vi = TestFunctions(W)

    # build right-hand sides
    b_list = []
    for f in f_list:
        L = (
            + Constant(0.0) * vr * dx(0)
            + Constant(0.0) * vi * dx(0)
            )
        for i, fval in f.items():
            L += (
                + fval[0] * vr * 2*pi*r * dx(i, degree=f_degree)
                + fval[1] * vi * 2*pi*r * dx(i, degree=f_degree)
                )
        b_list.append(assemble(L))

    # div(1/(mu r) grad(r phi)) + i sigma omega phi
    #
    # Split up diffusive and reactive terms to be able to assemble them with
    # different FFC parameters.
    #
    # Make omega a constant function to avoid rebuilding the equation system
    # when omega changes.
    om = Constant(omega)
    a1 = Constant(0.0) * ur * vr * dx(0)
    a2 = Constant(0.0) * ur * vr * dx(0)
    for i in subdomain_indices:
        # The term 1/r looks like it might cause problems. The dubious term is
        #
        #  1/r d/dr (r u_r) = u_r + 1/r du_r/dr,
        #
        # so we have to make sure that 1/r du_r/dr is bounded for all trial
        # functions u. This is guaranteed when taking Dirichlet boundary
        # conditions at r=0.
        sigma = Constant(Sigma[i])
        a1 += (
            + 1.0 / (Mu[i] * r) * dot(grad(r * ur), grad(r * vr)) * 2*pi*dx(i)
            + 1.0 / (Mu[i] * r) * dot(grad(r * ui), grad(r * vi)) * 2*pi*dx(i)
            )
        a2 += (
            - om * sigma * ui * vr * 2*pi*r * dx(i)
            + om * sigma * ur * vi * 2*pi*r * dx(i)
            )
        # Don't do anything at the interior boundary. Taking the Poisson
        # problem as an example, the weak formulation is
        #
        #     \int \Delta(u) v = -\int grad(u).grad(v) + \int_ n.grad(u) v.
        #
        # If we have 'artificial' boundaries through the domain, we would
        # like to make sure that along those boundaries, the equation is
        # exactly what it would be without the them. The important case to
        # look at are the trial and test functions which are nonzero on
        # the boundary. It is clear that the integral along the interface
        # boundary has to be omitted.

    # Add the convective component for the workpiece,
    #   a += <u, 1/r grad(r phi)> *2*pi*r*dx
    for i, conv in convections.items():
        a1 += (
            + dot(conv, grad(r * ur)) * vr * 2 * pi * dx(i)
            + dot(conv, grad(r * ui)) * vi * 2 * pi * dx(i)
            )

    force_m_matrix = False
    if force_m_matrix:
        A1 = assemble(a1)
        A2 = assemble(
            a2,
            form_compiler_parameters={
                'quadrature_rule': 'vertex',
                'quadrature_degree': 1
                })
        A = A1 + A2
    else:
        # Assembling the thing into one single object makes it possible to
        # extract .data() for conversion to SciPy's sparse types later.
        A = assemble(a1 + a2)

    # Compute the preconditioner as described in
    #
    #     A robust preconditioned MINRES-solver for time-periodic eddy-current
    #     problems;
    #     M. Kolmbauer, U. Langer;
    #     <http://www.numa.uni-linz.ac.at/Publications/List/2012/2012-02.pdf>.
    #
    # For the real-imag system
    #
    #     ( K  M )
    #     (-M  K ),
    #
    # Kolmbauer and Langer suggest the preconditioner
    #
    #     ( K+M        )
    #     (     -(K+M) ).
    #
    # The diagonal blocks can, for example, be solved with standard AMG
    # methods.
    p1 = Constant(0.0) * ur * vr * dx(0)
    p2 = Constant(0.0) * ur * vr * dx(0)
    # Diffusive terms.
    for i in subdomain_indices:
        p1 += (
            + 1.0 / (Mu[i]*r) * dot(grad(r*ur), grad(r*vr)) * 2 * pi * dx(i)
            - 1.0 / (Mu[i]*r) * dot(grad(r*ui), grad(r*vi)) * 2 * pi * dx(i)
            )
        p2 += (
            + om * Constant(Sigma[i]) * ur * vr * 2 * pi * r * dx(i)
            - om * Constant(Sigma[i]) * ui * vi * 2 * pi * r * dx(i)
            )
    P = assemble(p1 + p2)

    # build mass matrix
    # mm = sum([(ur * vr + ui * vi) * 2*pi*r * dx(i)
    #           for i in subdomain_indices
    #           ])
    mm = Constant(0.0) * ur * vr * dx(0)
    for i in subdomain_indices:
        mm += (
            + ur * vr * 2*pi*r * dx(i)
            + ui * vi * 2*pi*r * dx(i)
            )
    M = assemble(mm)

    # Apply boundary conditions.
    if bcs:
        bcs.apply(A)
        bcs.apply(P)
        bcs.apply(M)
        for b in b_list:
            bcs.apply(b)

    # helpers.show_matrix(A)
    # print(helpers.get_eigenvalues(A))
    # helpers.show_matrix(M)
    # helpers.show_matrix(P)

    return A, P, b_list, M, W


# def prescribe_current(A, b, coil_rings, current):
#     '''Get the voltage coefficients c_l with the total current prescribed.
#     '''
#     A[coil_rings][:] = 0.0
#     for i in coil_rings:
#         A[i][i] = 1.0
#     # The current must equal in all coil rings.
#     b[coil_rings] = current
#     return A, b


def prescribe_voltage(A, b, coil_rings, voltage, v_ref, J):
    '''Get the voltage coefficients :math:`c_l` with the total voltage
    prescribed.
    '''
    # The currents must equal in all coil rings.
    for k in range(len(coil_rings) - 1):
        i = coil_rings[k]
        i1 = coil_rings[k + 1]
        A[i][:] = J[i][:] - J[i1][:]
        b[i] = 0.0
    # sum c_k * v_ref == V
    i = coil_rings[-1]
    A[i][:] = 0.0
    A[i][coil_rings] = v_ref
    b[i] = voltage
    return A, b


# def prescribe_power(A, b, coil_rings, total_power, v_ref, J):
#     '''Get the voltage coefficients c_l with the total power prescribed.
#     '''
#     raise RuntimeError('Not yet implemented.')
#     # There are different notions of power for AC current; for an overview,
#     # see [1]. With
#     #
#     #     v(t) = V exp(i omega t),
#     #     i(t) = I exp(i omega t),
#     #
#     # V, I\in\C, we have
#     #
#     #     p(t) = v(t) * i(t).
#     #
#     # The time-average over one period is
#     #
#     #     P = 1/2 Re(V I*)
#     #       = Re(V_RMS I_RMS*)
#     #
#     # with the root-mean-square (RMS) quantities. This corresponds with the
#     # _real power_ in [1].
#     # When assuming that the voltage is real-valued, the power is
#     #
#     #    P = V/2 Re(I).
#     #
#     # [1] <https://en.wikipedia.org/wiki/AC_power>.
#     #
#     voltage = v_ref
#     A, b = prescribe_voltage(A, b, coil_rings, voltage, v_ref, J)
#
#     # Unconditionally take J[0] here. -- It shouldn't make a difference.
#      alpha = numpy.sqrt(
#          2 * total_power / (v_ref * numpy.sum(J[0][:] * c).real)
#          )
#     # We would like to scale the solution with alpha. For this, scale the
#     # respective part of the right-hand side.
#     b[coils] *= alpha
#     return A, b


def compute_potential(coils, V, dx, mu, sigma, omega, convections,
                      verbose=True,
                      io_submesh=None):
    '''Compute the magnetic potential :math:`\\Phi` with
    :math:`A = \\exp(\\text{i} \\omega t) \\Phi e_{\\theta}` for a number of
    coils.
    '''
    # Index all coil rings consecutively, starting with 0.
    # This makes them easier to handle for the equation system.
    physical_indices = []
    new_coils = []
    k = 0
    for coil in coils:
        new_coils.append([])
        for coil_ring in coil['rings']:
            new_coils[-1].append(k)
            physical_indices.append(coil_ring)
            k += 1

    # Set arbitrary reference voltage.
    v_ref = 1.0

    r = SpatialCoordinate(V.mesh())[0]

    # Compute reference potentials for all coil rings.
    # Prepare the right-hand sides according to :cite:`Cha97`.
    f_list = []
    for k in physical_indices:
        # Real an imaginary parts.
        f_list.append({k: (v_ref * sigma[k] / (2 * pi * r), Constant(0.0))})
    # Solve.
    phi_list = solve(
        V, dx,
        mu, sigma,
        omega,
        f_list,
        convections,
        tol=1.0e-12,
        verbose=True
        )

    # Write out these `phi`s to files.
    if io_submesh:
        V_submesh = FunctionSpace(io_submesh, 'CG', 1)
        W_submesh = V_submesh * V_submesh
        from dolfin import interpolate, XDMFFile
        for k, phi in enumerate(phi_list):
            # Restrict to workpiece submesh.
            phi_out = interpolate(phi, W_submesh)
            phi_out.rename('phi{:02d}'.format(k), 'phi{:02d}'.format(k))
            # Write to file
            with XDMFFile(mpi_comm_world(), 'phi{:02d}.xdmf'.format(k)) as xdmf_file:
                xdmf_file.write(phi_out)
            # plot(phi_out)
            # interactive()

    # Compute weights for the individual coils.
    # First get the voltage--coil-current mapping.
    J = get_voltage_current_matrix(
        phi_list, physical_indices, dx,
        sigma,
        omega,
        v_ref
        )

    num_coil_rings = len(phi_list)
    A = numpy.empty((num_coil_rings, num_coil_rings), dtype=J.dtype)
    b = numpy.empty(num_coil_rings, dtype=J.dtype)
    for k, coil in enumerate(new_coils):
        weight_type = coils[k]['c_type']
        target_value = coils[k]['c_value']
        # if weight_type == 'current':
        #     A, b = prescribe_current(A, b, coil, target_value)
        assert weight_type == 'voltage'
        A, b = prescribe_voltage(A, b, coil, target_value, v_ref, J)

    # # TODO write out the equation system to a file
    # if io_submesh:
    #     numpy.savetxt('matrix.dat', A)

    # Solve the system for the weights.
    weights = numpy.linalg.solve(A, b)

    # # Prescribe total power.
    # target_total_power = 4.0e3
    # # Compute all coils with reference voltage.
    # num_coil_rings = J.shape[0]
    # A = numpy.empty((num_coil_rings, num_coil_rings), dtype=J.dtype)
    # b = numpy.empty(num_coil_rings)
    # for k, coil in enumerate(new_coils):
    #     target_value = v_ref
    #     A, b = prescribe_voltage(A, b, coil, target_value, v_ref, J)
    # weights = numpy.linalg.solve(A, b)
    # preliminary_voltages = v_ref * weights
    # preliminary_currents = numpy.dot(J, weights)
    # # Compute resulting total power.
    # total_power = 0.0
    # for coil_loops in new_coils:
    #     # Currents should be the same all over the entire coil,
    #     # so take currents[coil_loops[0]].
    #     total_power += 0.5 \
    #                  * numpy.sum(preliminary_voltages[coil_loops]) \
    #                  * preliminary_currents[coil_loops[0]].real
    #                  # TODO no abs here
    # # Scale all voltages by necessary factor.
    # weights *= numpy.sqrt(target_total_power / total_power)

    if verbose:
        info('')
        info('Resulting voltages,   V/sqrt(2):')
        voltages = v_ref * weights
        info('   {}'.format(abs(voltages) / numpy.sqrt(2)))
        info('Resulting currents,   I/sqrt(2):')
        currents = numpy.dot(J, weights)
        info('   {}'.format(abs(currents) / numpy.sqrt(2)))
        info('Resulting apparent powers (per coil):')
        for coil_loops in new_coils:
            # With
            #
            #     v(t) = Im(exp(i omega t) v),
            #     i(t) = Im(exp(i omega t) i),
            #
            # the average apparent power over one period is
            #
            #     P_av = omega/(2 pi) int_0^{2 pi/omega} v(t) i(t)
            #          = 1/2 Re(v i*).
            #
            # Currents should be the same all over, so take currents[coil[0]].
            #
            alpha = sum(voltages[coil_loops]) \
                * currents[coil_loops[0]].conjugate()
            power = 0.5 * alpha.real
            info('   {}'.format(power))
        info('')

    # Compute Phi as the linear combination \sum C_i*phi_i.
    # The function Phi is guaranteed to fulfill the PDE as well (iff the
    # the boundary conditions are linear in phi too).
    #
    # Form $\sum_l c_l \\phi_l$.
    # https://answers.launchpad.net/dolfin/+question/214172
    #
    # Unfortunately, one cannot just use
    #     Phi[0].vector()[:] += c.real * phi[0].vector()
    # since phi is from the FunctionSpace V*V and thus .vector() is not
    # available for the individual components.
    #
    Phi = [
        Constant(0.0),
        Constant(0.0)
        ]
    for phi, c in zip(phi_list, weights):
        # Phi += c * phi
        Phi[0] += c.real * phi[0] - c.imag * phi[1]
        Phi[1] += c.imag * phi[0] + c.real * phi[1]

    # Project the components down to V. This makes various subsequent
    # computations with Phi faster.
    Phi[0] = project(Phi[0], V)
    Phi[0].rename('Re(Phi)', 'Re(Phi)')
    Phi[1] = project(Phi[1], V)
    Phi[1].rename('Im(Phi)', 'Im(Phi)')
    return Phi, voltages


def get_voltage_current_matrix(phi, physical_indices, dx,
                               Sigma,
                               omega,
                               v_ref):
    '''Compute the matrix that relates the voltages with the currents in the
    coil rings. (The relationship is indeed linear.)

    This is according to :cite:`KP02`.

    The entry :math:`J_{k,l}` in the resulting matrix is the contribution of
    the potential generated by coil :math:`l` to the current in coil :math:`k`.
    '''
    mesh = phi[0].function_space().mesh()

    r = SpatialCoordinate(mesh)[0]

    num_coil_rings = len(phi)
    J = numpy.empty((num_coil_rings, num_coil_rings), dtype=numpy.complex)
    for l, pi0 in enumerate(physical_indices):
        partial_phi_r, partial_phi_i = phi[l].split()
        for k, pi1 in enumerate(physical_indices):
            # -1i*omega*int_{coil_k} sigma phi.
            int_r = assemble(Sigma[pi1] * partial_phi_r * dx(pi1))
            int_i = assemble(Sigma[pi1] * partial_phi_i * dx(pi1))
            J[k][l] = -1j * omega * (int_r + 1j * int_i)
        # v_ref/(2*pi) * int_{coil_l} sigma/r.
        # 1/r doesn't explode since we only evaluate it in the coils where
        # r!=0.
        # For assemble() to work, a mesh needs to be supplied either implicitly
        # by the integrand, or explicitly. Since the integrand doesn't contain
        # mesh information here, pass it through explicitly.
        J[l][l] += (
            v_ref / (2 * pi) * assemble(Sigma[pi0] / r * dx(pi0))
            )
    return J


# pylint: disable=unused-argument
def compute_joule(Phi, voltages,
                  omega, Sigma, Mu,
                  subdomain_indices):
    '''
    See, e.g., equation (2.17) in :cite:`Cha97`.

    In a time-harmonic approximation with

    ..math::

        \\begin{align}
        A &= \\Re(a exp(\\text{i} \\omega t)),\\\\
        B &= \\Re(b exp(\\text{i} \\omega t)),
        \\end{align}

    the time-average of :math:`A\\cdot B` over one period is

    ..math::

       \\overline{A\\cdot B} = \\frac{1}{2} \\Re(a \\cdot b^*)

    see http://www.ece.rutgers.edu/~orfanidi/ewa/ch01.pdf.
    In particular,

    ..math::

       \\overline{A\\cdot A} = \\frac{1}{2} \\|a\\|^2.

    Consequently, we can compute the average source term over one period
    as

    ..math::

        s = \\frac{1}{2} \\|j\\|^2 / \\sigma = \\frac{1}{2} \\|E\\|^2 \\sigma.

    (Not using :math:`j` avoids explicitly dividing by :math:`\\sigma` which is
    0 at nonconductors.)
    '''
    # j_r = {}
    # j_i = {}
    # E_r =  omega*Phi_i + rhs_r
    # E_i = -omega*Phi_r + rhs_i
    # plot(E_r)
    # plot(E_i)
    # interactive()
    # exit()
    # The Joule heating source is
    # https://en.wikipedia.org/wiki/Joule_heating#Differential_Form
    #
    #   P = J.E =  \\sigma E.E.
    #
    # joule_source = zero() * dx(0)
    joule_source = {}
    mesh = Phi[0].function_space().mesh()
    r = SpatialCoordinate(mesh)[0]
    for i in subdomain_indices:
        # See, e.g., equation (2.17) in
        #
        #  Numerical modeling in induction heating for axisymmetric geometries,
        #  Chaboudez et al.,
        #  IEEE Transactions of magnetics, vol. 33, no. 1, January 1997.
        #
        # In a time-harmonic approximation with
        #     A = Re(a exp(i omega t)),
        #     B = Re(b exp(i omega t)),
        # the time-average of $A\\cdot B$ over one period is
        #
        #    \\overline{A\\cdot B} = 1/2 Re(a \\cdot b*)
        #
        # see <http://www.ece.rutgers.edu/~orfanidi/ewa/ch01.pdf>.
        # In particular,
        #
        #    \\overline{A\\cdot A} = 1/2 ||a||^2
        #
        # Consequently, we can compute the average source term over one period
        # as
        #
        #     source = 1/2 ||j||^2 / sigma = 1/2 * ||E||^2 * sigma.
        #
        # (Not using j avoids explicitly dividing by sigma which is 0 at
        # nonconductors.)
        #
        # TODO check this part
        E_r = +omega * Phi[1]
        E_i = -omega * Phi[0]
        if i in voltages:
            E_r += voltages[i].real / (2*pi*r)
            E_i += voltages[i].imag / (2*pi*r)
        # Make Sigma[i] a Constant since it could be 0 and then render the
        # entire Expression 0 (float).
        joule_source[i] = 0.5 * Constant(Sigma[i]) * (E_r*E_r + E_i*E_i)

    # # Alternative computation.
    # joule_source = zero() * dx(0)
    # for i in subdomain_indices:
    #     joule_source += 1.0/(Mu[i]*r) * dot(grad(r*Phi_r),grad(v)) * dx(i)

    # # And the third way (for omega==0)
    # joule_source = zero() * dx(0)
    # for i in subdomain_indices:
    #     if i in C:
    #         joule_source += (
    #             Sigma[i] * voltages[i].real / (2*pi*r) * v * dx(i)
    #             )
    # u = TrialFunction(V)
    # sol = Function(V)
    # solve(u*v*dx() == joule_source, sol,
    #       bcs = DirichletBC(V, 0.0, 'on_boundary'))
    # plot(sol)
    # interactive()
    # exit()
    return joule_source


def compute_lorentz(Phi, omega, sigma):
    '''In a time-harmonic discretization with quantities

    .. math::

        \\begin{align}
            A &= \\Re(a \\exp(\\text{i} \\omega t)),\\\\
            B &= \\Re(b \\exp(\\text{i} \\omega t)),
        \\end{align}

    the time-average of :math:`A\\times B` over one period is

    .. math::
        \\overline{A\\times B} = \\frac{1}{2} \\Re(a \\times b^*),

    see http://www.ece.rutgers.edu/~orfanidi/ewa/ch01.pdf.
    Since the Lorentz force generated by the current :math:`J` in the magnetic
    field :math:`B` is

    .. math::
        F_L = J \\times B,

    its time average is

    .. math::
       \\overline{F_L} = \\frac{1}{2} \\Re(j \\times b^*).

    With

    .. math::
       J &= \\Re(\\exp(\\text{i} \\omega t) j e_{\\theta}),\\\\
       B &= \\Re\\left(
           \\exp(i \\omega t) \\left(
             -\\frac{\\text{d}\\phi}{\\text{d}z} e_r
             + \\frac{1}{r} \\frac{\\text{d}(r\\phi)}{\\text{d}r} e_z
           \\right)
           \\right),

    we have

    .. math::
       \\overline{F_L}
           &= \\frac{1}{2} \\Re\\left(j \\frac{d\\phi^*}{dz} e_z
              + \\frac{j}{r} \\frac{d(r\\phi^*)}{dr} e_r\\right)\\\\
           &= \\frac{1}{2}
              \\Re\\left(\\frac{j}{r} \\nabla(r\\phi^*)\\right)\\\\

    In the workpiece, we can assume

    .. math::
        j = -\\text{i} \\sigma \\omega \\phi

    which gives

    .. math::
       \\begin{align*}
       \\overline{F_L}
           &= \\frac{\\sigma\\omega}{2r} \\Im\\left(
                  \\phi \\nabla(r \\phi^*)
                  \\right)\\\\
           &= \\frac{\\sigma\\omega}{2r} \\left(
                \\Im(\\phi) \\nabla(r \\Re(\\phi))
               -\\Re(\\phi) \\nabla(r \\Im(\\phi))
               \\right)
       \\end{align*}
    '''
    mesh = Phi[0].function_space().mesh()
    r = SpatialCoordinate(mesh)[0]
    return 0.5 * sigma * omega / r * (
        + Phi[1] * grad(r * Phi[0])
        - Phi[0] * grad(r * Phi[1])
        )
