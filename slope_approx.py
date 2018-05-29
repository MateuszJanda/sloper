#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import collections as co
import itertools as it
import numpy as np
import copy
import cv2


CALIBRATION_AREA_SIZE = 40
BLACK = 0
WHITE = 255

Point = co.namedtuple('Point', ['x', 'y'])
Size = co.namedtuple('Size', ['width', 'height'])


def calibration_data(img):
    """
    Calculate calibration data .
    Return:
        top-left point where cell start
        cell size (width, height)
    """
    under_tl, under_br = underscore_pos(img)
    roof = roof_pos(img, under_tl, under_br)
    sep_height = separator_height(img, under_tl, under_br)

    width = width = under_br.x - under_tl.x + 1
    height = under_br.y - roof.y + sep_height
    start_pt = Point(under_tl.x, under_br.y - height)

    cell_size = Size(width, height)
    print('Cell top-left: ' + str(start_pt) + ', cell size: ' + str(cell_size))
    return start_pt, cell_size


def underscore_pos(img):
    """ Calculate underscore "_" position [top-left point, bottom-right point] """
    under_tl = None
    for x, y in it.product(range(CALIBRATION_AREA_SIZE), range(CALIBRATION_AREA_SIZE)):
        if img[y, x] != BLACK:
            under_tl = Point(x, y)
            break

    tmp = None
    for x in range(under_tl.x, CALIBRATION_AREA_SIZE):
        if img[under_tl.y, x] == BLACK:
            break
        tmp = Point(x, under_tl.y)

    under_rb = None
    for y in range(tmp.y, CALIBRATION_AREA_SIZE):
        if img[y, tmp.x] == BLACK:
            break
        under_rb = Point(tmp.x, y)

    print('Underscore top left: ', under_tl)
    print('Underscore bottom right: ', under_rb)
    return under_tl, under_rb


def roof_pos(img, under_pos1, under_pos2):
    """ Calculate roof sign "^" position - only the pick """
    roof = Point(0, CALIBRATION_AREA_SIZE)
    width = under_pos2.x - under_pos1.x + 1

    for x in range(under_pos2.x + 1, under_pos2.x + width):
        for y in range(CALIBRATION_AREA_SIZE):
            if img[y, x] != BLACK and y < roof.y:
                roof = Point(x, y)

    print('Roof pos: ', roof)
    return roof


def separator_height(img, under_pos1, under_pos2):
    """ Calculate separator area between underscore "_" and bottom roof sign "^" """
    roof = Point(0, CALIBRATION_AREA_SIZE)
    width = under_pos2.x - under_pos1.x + 1

    for x in range(under_pos1.x, under_pos1.x + width):
        for y in range(under_pos2.y + 1, CALIBRATION_AREA_SIZE):
            if img[y, x] != BLACK and y < roof.y:
                roof = Point(x, y)

    height = roof.y - under_pos2.y

    print('Separator height: ', height)
    return height


def draw_filled_cell(img, start_pt, cell_size):
    """ Just for debug purpose, will cell with color """
    for x in range(start_pt.x, start_pt.x + cell_size.width):
        for y in range(start_pt.y, start_pt.y + cell_size.height):
            img[y, x] ^= 158


def draw_net(img, start_pt, cell_size):
    """ Just for debug purpose draw net """
    BLUE = (255, 0, 0)
    end_y = start_pt.y + ((img.shape[0] - start_pt.y) // cell_size.height) * cell_size.height
    end_x = start_pt.x + ((img.shape[1] - start_pt.x) // cell_size.width) * cell_size.width
    print end_x, img.shape[1]
    for x in range(start_pt.x, end_x + 1, cell_size.width):
        # print x
        cv2.line(img, (x, start_pt.y), (x, end_y), BLUE, 1)
    # cv2.line(img, (253, start_pt.y), (253, height), BLUE, 1)

    for y in range(start_pt.y, end_y + 1, cell_size.height):
        cv2.line(img, (start_pt.x, y), (end_x, y), BLUE, 1)
    # cv2.line(img, (start_pt.x, 10), (253, 10), BLUE, 1)


def draw_dots(img, start_pt, cell_size):
    count = 0
    braille_cell_size = Size(2.0, 4.0)
    field_size = Size(cell_size.width/braille_cell_size.width, cell_size.height/braille_cell_size.height)

    height, width, _ = img.shape
    new_width = ((width - start_pt.x) // cell_size.width) * cell_size.width
    stepx = (new_width / cell_size.width) * braille_cell_size.width

    new_height = ((height - start_pt.y) // cell_size.height) * cell_size.height
    stepy = (new_height / cell_size.height) * braille_cell_size.height

    for x in np.linspace(start_pt.x, width, stepx):
        for y in np.linspace(start_pt.y, height, stepy):
            # print x, y
            # if count % 2 == 0:
            if True:
                # roi = img[y:y+field_size.height, x:x+field_size.width]
                print str(int(x)) + '  ' + str(int(x+field_size.width)) + ' - ' + str(int(y)) + ' ' + str( int(y+field_size.height))
                img[int(y):int(y+field_size.height), int(x):int(x+field_size.width)] ^= 69

            count += 1



    pass


def erase_calibration_area(img):
    """ Erase calibration are from image """
    cv2.rectangle(img, (0, 0), (CALIBRATION_AREA_SIZE, CALIBRATION_AREA_SIZE), BLACK, cv2.FILLED)


def find_nearest(head_cnt, contours, min_dist=15):
    """ Find nearest contour to current head contour """
    best_cnt = None
    for cnt in contours:
        for head_pos, cnt_pos in it.product(head_cnt, cnt):
            dist = np.linalg.norm(head_pos-cnt_pos)

            if abs(dist) < min_dist:
                min_dist = abs(dist)
                best_cnt = cnt

    return best_cnt


def connect_nearby_contours(gray_img):
    """
    Connect nearby contours (ASCII characters)
    See also:
    https://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them
    http://answers.opencv.org/question/169492/accessing-all-points-of-a-contour/
    """
    _, contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Countours start from the bottom
    last = contours.pop(0)
    chain = [last]
    while len(contours) > 0:
        cnt = find_nearest(last, contours)

        if cnt is None:
            raise(Excpetion('Error! Countours length: ' + len(contours)))

        chain.append(cnt)
        for i in range(len(contours)):
            if np.all(cnt == contours[i]):
                contours.pop(i)
                break
        last = cnt

    unified = [np.vstack(chain)]
    cv2.drawContours(gray_img, unified, -1, WHITE, 1)


def main():
    orig_img = cv2.imread('ascii_fig.png', cv2.IMREAD_GRAYSCALE)

    # Image should have white characters and black background
    _, gray_img= cv2.threshold(src=orig_img, thresh=30, maxval=255, type=cv2.THRESH_BINARY)

    start_pt, cell_size = calibration_data(gray_img)
    erase_calibration_area(gray_img)

    connect_nearby_contours(gray_img)

    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    draw_filled_cell(color_img, start_pt, cell_size)
    draw_net(color_img, start_pt, cell_size)
    # draw_dots(color_img, start_pt, cell_size)
    cv2.imshow('color_img', color_img)

    cv2.imshow('orig_img', orig_img)
    cv2.imshow('gray_img', gray_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
