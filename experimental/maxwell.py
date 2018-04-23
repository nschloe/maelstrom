def _adaptive_mesh_refinement(dx, phi, mu, sigma, omega, conv, voltages):
    from dolfin import cells, refine
    eta = _error_estimator(dx, phi, mu, sigma, omega, conv, voltages)
    mesh = phi.function_space().mesh()
    level = 0
    TOL = 1.0e-4
    E = sum([e * e for e in eta])
    E = sqrt(MPI.sum(E))
    info('Level {:d}: E = {:g} (TOL = {:g})'.format(level, E, TOL))
    # Mark cells for refinement
    REFINE_RATIO = 0.5
    cell_markers = MeshFunction('bool', mesh, mesh.topology().dim())
    eta_0 = sorted(eta, reverse=True)[int(len(eta) * REFINE_RATIO)]
    eta_0 = MPI.max(eta_0)
    for c in cells(mesh):
        cell_markers[c] = eta[c.index()] > eta_0
    # Refine mesh
    mesh = refine(mesh, cell_markers)
    # Plot mesh
    plot(mesh)
    interactive()
    exit()
    # # Compute error indicators
    # K = array([c.volume() for c in cells(mesh)])
    # R = numpy.array([
    #     abs(source([c.midpoint().x(), c.midpoint().y()]))
    #     for c in cells(mesh)
    #     ])
    # gam = h*R*sqrt(K)
    return


def _error_estimator(dx, phi, mu, sigma, omega, conv, voltages):
    '''Simple error estimator from

        A posteriori error estimation and adaptive mesh-refinement techniques;
        R. Verfürth;
        Journal of Computational and Applied Mathematics;
        Volume 50, Issues 1-3, 20 May 1994, Pages 67-83;
        <https://www.sciencedirect.com/science/article/pii/0377042794902909>.

    The strong PDE is

        - div(1/(mu r) grad(rphi)) + <u, 1/r grad(rphi)> + i sigma omega phi
      = sigma v_k / (2 pi r).
    '''
    from dolfin import cells
    mesh = phi.function_space().mesh()
    # Assemble the cell-wise residual in DG space
    DG = FunctionSpace(mesh, 'DG', 0)
    # get residual in DG
    v = TestFunction(DG)
    R = _residual_strong(dx, v, phi, mu, sigma, omega, conv, voltages)
    r_r = assemble(R[0])
    r_i = assemble(R[1])
    r = r_r * r_r + r_i * r_i
    visualize = True
    if visualize:
        # Plot the cell-wise residual
        u = TrialFunction(DG)
        a = zero() * dx(0)
        subdomain_indices = mu.keys()
        for i in subdomain_indices:
            a += u * v * dx(i)
        A = assemble(a)
        R2 = Function(DG)
        solve(A, R2.vector(), r)
        plot(R2, title='||R||^2')
        interactive()
    K = r.array()
    info('{:r}'.format(K))
    h = numpy.array([c.diameter() for c in cells(mesh)])
    eta = h * numpy.sqrt(K)
    return eta


