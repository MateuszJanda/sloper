#!/usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
Ad maiorem Dei gloriam
"""

import argparse
import numpy as np


def main():
    args = interpret_args()

    shape = calc_shape(args.file_name)
    ascii_arr = copy_to_array(args.file_name, shape)
    marksers = mark_contours(ascii_arr)
    contures_arr = copy_contures(ascii_arr, marksers)
    show_contures(contures_arr)


def calc_shape(file_name):
    width = 0
    height = 0
    with open(file_name, 'r') as f:
        for line in f:
            if len(line) > width:
                width = len(line)
            height += 1

    return height, width


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def interpret_args():
    parser = argparse.ArgumentParser(
        description='Drill ASCII figure and and leave the contours.\n'
                    'Last version you can find on github.com/MateuszJanda/sloper\n'
                    'Mateusz Janda (c) <mateusz janda at gmail com>',
        usage='Please try to use -h, --help for more informations',
        epilog='Example:\n'
               '$ python driller.py -i ascii_data/logo.txt\n',
        formatter_class=CustomFormatter)

    parser.add_argument('-i', '--input', metavar='file', dest='file_name',
        help='ASCII figure in text file.')

    args = parser.parse_args()

    return args



def copy_to_array(file_name, shape):
    ascii_arr = np.full(shape, fill_value= ' ')

    with open(file_name, 'r') as f:
        num = 0
        for line in f:
            line = line.rstrip()
            ascii_arr[num, :] = [c for c in line] + [' ' for _ in range(shape[1] - len(line))]
            num += 1

    return ascii_arr


def mark_contours(ascii_arr):
    marksers = np.zeros_like(ascii_arr)

    # Vertically
    for x in range(ascii_arr.shape[1]):
        begin = True
        last_y = 0
        for y in range(ascii_arr.shape[0]):
            if begin and ascii_arr[y, x] != ' ':
                marksers[y, x] = 1
                begin = False
                continue

            if not begin and ascii_arr[y, x] != ' ':
                last_y = y
                continue
        marksers[last_y, x] = 1

    # Horizontally
    for y in range(ascii_arr.shape[0]):
        begin = True
        last_x = 0
        for x in range(ascii_arr.shape[1]):
            if begin and ascii_arr[y, x] != ' ':
                marksers[y, x] = 1
                begin = False
                continue

            if not begin and ascii_arr[y, x] != ' ':
                last_x = x
                continue
        marksers[y, last_x] = 1

    return marksers


def copy_contures(ascii_arr, marksers):
    contures_arr = np.full_like(ascii_arr, fill_value=' ')
    for y in range(marksers.shape[0]):
        for x in range(marksers.shape[1]):
            if marksers[y, x]:
                contures_arr[y, x] = ascii_arr[y, x]

    return contures_arr


def show_contures(contures_arr):
    for y in range(contures_arr.shape[0]):
        print(''.join(contures_arr[y, :].tolist()))


if __name__ == '__main__':
    main()
