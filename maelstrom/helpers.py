# -*- coding: utf-8 -*-
#
from dolfin import (
        as_backend_type, DirichletBC, assemble, dx
        )
import matplotlib.pyplot as plt
import scipy.linalg


def show_matrix(A):
    A = as_backend_type(A)
    A_matrix = A.sparray()

    # colormap
    cmap = plt.cm.gray_r
    A_dense = A_matrix.todense()
    # A_r = A_dense[0::2][0::2]
    # A_i = A_dense[1::2][0::2]
    cmap.set_bad('r')
    # im = plt.imshow(
    #     abs(A_dense), cmap=cmap, interpolation='nearest', norm=LogNorm()
    #     )
    plt.imshow(abs(A_dense), cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.show()
    return


def get_eigenvalues(A):
    A = as_backend_type(A)
    A_matrix = A.sparray()
    return scipy.linalg.eigvals(A_matrix.todense())


def dbcs_to_productspace(W, bcs_list):
    new_bcs = []
    for k, bcs in enumerate(bcs_list):
        for bc in bcs:
            C = bc.function_space().component()
            # pylint: disable=len-as-condition
            if len(C) == 0:
                new_bcs.append(DirichletBC(W.sub(k),
                                           bc.value(),
                                           bc.domain_args[0]))
            else:
                assert len(C) == 1, 'Illegal number of subspace components.'
                new_bcs.append(
                        DirichletBC(
                            W.sub(k).sub(int(C[0])),
                            bc.value(),
                            bc.domain_args[0]
                            ))

    return new_bcs


def average(u):
    '''Computes the average value of a function u over its domain.
    '''
    return assemble(u * dx) \
        / assemble(1.0 * dx(u.function_space().mesh()))
