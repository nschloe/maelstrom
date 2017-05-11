# -*- coding: utf-8 -*-
#
'''
Read Tecplot data.
'''

import numpy
import re


def read(filename):
    # Read the meta data.
    # It is principally structured like
    #
    #   NAME = content
    #
    # Items aren't separated by anything else than whitespace. Hence, the only
    # character that provides structure here is the equality sign "=".

    c = {}
    previous_key = None

    f = open(filename, 'r')
    line = f.readline()
    while line:
        out_key_value = re.match('([A-Z][A-Z ]*)\s*=\s*"([^"]*)"\s*', line)
        out_value = re.match('\s*"([^"]*)"\s*', line)
        if out_key_value:
            key = out_key_value.group(1).strip()
            value = out_key_value.group(2)
            if re.match('ZONE.*', key):
                # Special treatment for zones.
                c[key] = _read_zone(f, c['VARIABLES'])
                c[key]['title'] = value
            else:
                c[key] = value
            previous_key = key
        elif out_value:
            # Only a value present in this line. It must belong to the previous
            # key.
            value = out_value.group(1)
            try:
                c[previous_key].append(value)
            except RuntimeError:
                # Convert previous key-value to key-listofvalues.
                previous_value = c[previous_key]
                c[previous_key] = [previous_value, value]
        # Read next line.
        line = f.readline()
    f.close()
    return c


def _read_zone(f, variable_names):
    '''Read ZONE data from a Tecplot file.
    '''
    zone = {}

    print('Reading zone header...')
    line = f.readline()
    while line:  # zone_header and line:
        # Read the zone header.
        # Break up the line at commas and read individually.
        all_units_success = True
        units = line.split(',')
        for unit in units:
            re_key_value = '\s*([A-Z][A-Za-z]*)=(.*)'
            out = re.match(re_key_value, unit)
            if out:
                key = out.group(1)
                value = out.group(2)
                if key == 'STRANDID' or key == 'Nodes' or key == 'Elements':
                    value = int(value)
                elif key == 'SOLUTIONTIME':
                    value = float(value)
                zone[key] = value
            else:
                all_units_success = False
                break
        if not all_units_success:
            # we must be in the numerical data section already
            break
        line = f.readline()

    print('Reading zone data...')
    # Fill in the numerical data into an array.
    num_nodes = zone['Nodes']
    # data = numpy.empty((num_nodes, num_colums))
    # We're in a ZONE and the pattern doesn't match KEY=value. This must mean
    # we're dealing with numerical values now.  Check out what DT says and
    # build the appropriate regex.
    dt = zone['DT']
    # Strip leading and trailing brackets.
    dt = dt.strip('() ')
    # Convert the Tecplot DT (data type) to an array of numpy data types.
    data = {}
    tp_datatypes = dt.split()
    num_columns = len(tp_datatypes)
    assert(num_columns == len(variable_names))
    for l, tp_dt in enumerate(tp_datatypes):
        name = variable_names[l]
        if tp_dt == 'SINGLE':
            data[name] = numpy.empty(num_nodes, dtype=float)
        else:
            raise RuntimeError('Unknown Tecplot data type \'%s\'.' % tp_dt)
    # Build the regex for every data line.
    SINGLE_regex = '[-+]?[0-9]\.[0-9]+E[-+][0-9][0-9]'
    dt = ' ' + dt.replace('SINGLE', '(' + SINGLE_regex + ')')
    # Read all node data.
    for k in range(num_nodes):
        out = re.match(dt, line)
        assert(out)
        for l in range(num_columns):
            name = variable_names[l]
            data[name][k] = out.group(l+1)
        line = f.readline()
    # Copy over the data.
    zone['node data'] = data

    # Read elements (element connectivity).
    num_elements = zone['Elements']
    if zone['ZONETYPE'] == 'FELineSeg':
        num_nodes_per_element = 2
    else:
        raise RuntimeError('Invalid ZONETYPE \'%s\'.' % zone['ZONETYPE'])
    data = numpy.empty((num_nodes, num_nodes_per_element), dtype=int)
    element_regex = ' ([0-9]+)+\s+([0-9]+)'
    for k in range(num_elements):
        out = re.match(element_regex, line)
        assert(out)
        for l in range(num_nodes_per_element):
            data[k][l] = out.group(l+1)
        line = f.readline()
    zone['element data'] = data

    return zone


def read_with_vtk(filename):
    import vtk
    reader = vtk.vtkTecplotReader()
    reader.SetFileName(filename)
    reader.Update()
    exit()
    return
