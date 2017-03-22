# -*- coding: utf-8 -*-
#
'''
Allows using DOLFIN's indented block messages with 'with', i.e.,

    with Message('hello world'):
        # do something
'''
from dolfin import begin, end


class Message:

    def __init__(self, string):
        self.string = string
        return

    def __enter__(self):
        begin(self.string)
        return

    def __exit__(self, type, value, traceback):
        end()
        return
