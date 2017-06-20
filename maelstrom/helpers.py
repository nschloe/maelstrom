# -*- coding: utf-8 -*-
#
from dolfin import DirichletBC, assemble, dx


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
