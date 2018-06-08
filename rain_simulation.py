#! /usr/bin/env python
# -*- coding: utf-8 -*-

import collections as co
import numpy as np


Vector = co.namedtuple('Vector', ['x', 'y'])

VECTOR_DIM = 2


def import_norm_vector_arr(file_name):
    arr = np.loadtxt(file_name)
    height, width = arr.shape
    return arr.reshape(height, width/VECTOR_DIM, VECTOR_DIM)


def main():
    file_name = 'ascii_fig.png.norm'
    norm_vec_arr = import_norm_vector_arr(file_name)
    print norm_vec_arr.shape
    print norm_vec_arr


if __name__ == '__main__':
    main()
