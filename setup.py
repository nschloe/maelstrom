# -*- coding: utf-8 -*-
#
import os
from distutils.core import setup
import codecs


def read(fname):
    return codecs.open(os.path.join(os.path.dirname(__file__), fname),
                       encoding='utf-8'
                       ).read()

setup(name='maelstrom',
      packages=['maelstrom'],
      version='0.1.0',
      description='Numerical solution of magnetohydrodynamics problems',
      long_description=read('README.md'),
      author='Nico Schlömer',
      author_email='nico.schloemer@gmail.com',
      url='https://github.com/nschloe/maelstrom/',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Mathematics'
          ],
      )
