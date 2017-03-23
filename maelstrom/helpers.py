# -*- coding: utf-8 -*-
#
from dolfin import (
        as_backend_type
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
