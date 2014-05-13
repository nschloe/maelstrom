#!/usr/bin/env python
#
#  Copyright (c) 2012--2014, Nico Schlömer, <nico.schloemer@gmail.com>
#  All rights reserved.
#
#  This file is part of PyMHD.
#
#  PyMHD is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  PyMHD is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with PyMHD.  If not, see <http://www.gnu.org/licenses/>.
#
'''
Gmsh to FEniCS file converter. This is better than the regular
dolfin-convert since it preserves physical region names.
'''
import numpy as np


def _main():
    args = _parse_args()
    _read_gmsh(args.input)
    return


def _read_gmsh(filename):
    # A Gmsh file is structured like
    #
    # $MeshFormat
    # 2.2 0 8
    # $EndMeshFormat
    # $PhysicalNames
    # ...
    # $EndPhysicalNames
    #
    # In general:
    #
    # $Tag
    # ...
    # $EndTag
    #

    # regexp for $Tag
    import re
    start_tag = '\$(\w+)'

    f = open(filename, 'r')
    line = f.readline()
    while line:
        a = re.match(start_tag, line)
        assert a
        tag = a.group(1)

        if tag == 'MeshFormat':
            line = f.readline()
            # We could check the format version here.

        elif tag == 'PhysicalNames':
            line = f.readline()
            num_physical_names = int(line)
            physical_names = np.empty(num_physical_names,
                                      dtype=np.dtype([('dim', int),
                                                      ('id', int),
                                                      ('name', str)]))
            for k in xrange(num_physical_names):
                data = f.readline().split()
                physical_names[k] = (int(data[0]), int(data[1]), data[2])

        elif tag == 'Nodes':
            line = f.readline()
            num_nodes = int(line)
            nodes = np.empty(num_nodes,
                             dtype=np.dtype([('id', int),
                                             ('coords', (float, 3))]))
            for k in xrange(num_nodes):
                data = f.readline().split()
                nodes[k] = (int(data[0]),
                            [float(data[1]), float(data[2]), float(data[3])]
                            )

        elif tag == 'Elements':
            line = f.readline()
            num_elements = int(line)
            elements = np.empty(num_elements,
                                dtype=np.dtype([('id', int),
                                                ('tag', int),
                                                ('nodes', (int, 4))]))
            # 1 4 2 1 55 7969 8002 7844 8005
            # Actually:
            # ID, numnodes, numtags, tag1, tag2,..., node1, node2,...
            # This is hardcoded for numnodes==4, numtags==2.
            for k in xrange(num_elements):
                data = f.readline().split()
                elements[k] = \
                    (int(data[0]),
                     int(data[3]),
                     [int(data[5]), int(data[6]), int(data[7]), int(data[8])]
                     )
        else:
            raise RuntimeError('Unknown tag \'%s\'.' % tag)

        # Make sure the end tag is here
        assert re.match('\$End%s' % tag, f.readline())
        # and read the next line:
        line = f.readline()

    return nodes, elements, physical_names


def _parse_args():
    '''Parse input arguments.'''
    import argparse
    parser = argparse.ArgumentParser(description='Convert Gmsh files '
                                                 'to FEniCS.')
    parser.add_argument('input',
                        help='Input Gmsh file'
                        )
    parser.add_argument('output',
                        help='Output FEniCS file'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    _main()
