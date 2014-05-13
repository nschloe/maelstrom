#!/bin/bash
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
# Launches a full computation in an appropriate subdirectory.

if [ -z "$1" ]; then
    echo "Error!"
    echo "Please provide a string that describes the experiment."
    echo "Abort."
    exit 1
fi

dirname=results-`date -u +"%Y-%m-%d-%H.%M"`-$1
mkdir $dirname
cd $dirname
screen -L -d -m ../full-cylindrical
cd ..
