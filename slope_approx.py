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


def calibration_data(img):
    """
    Calculate calibration area.
    Return top-left point where cell start, cell width, cell hight
    """
    under_pos1, under_pos2 = underscore_pos(img)
    roof = roof_pos(img, under_pos1, under_pos2)
    sep_high = separator_high(img, under_pos1, under_pos2)

    width = width = under_pos2.x - under_pos1.x + 1
    high = under_pos2.y - roof.y + sep_high
    start_pt = Point(under_pos1.x, under_pos2.y - high)

    print('Cell top-left: ' + str(start_pt) + ', size: ' + str((width, high)))
    return start_pt, width, high


def underscore_pos(img):
    """ Calculate underscore "_" position [top-left point, bottom-right point]"""
    pt1 = None
    for x, y in it.product(range(CALIBRATION_AREA_SIZE), range(CALIBRATION_AREA_SIZE)):
        if img[y, x] != BLACK:
            pt1 = Point(x, y)
            break

    tmp = None
    for x in range(pt1.x, CALIBRATION_AREA_SIZE):
        if img[pt1.y, x] == BLACK:
            break
        tmp = Point(x, pt1.y)

    pt2 = None
    for y in range(tmp.y, CALIBRATION_AREA_SIZE):
        if img[y, tmp.x] == BLACK:
            break
        pt2 = Point(tmp.x, y)

    print('Underscore top left: ', pt1)
    print('Underscore bottom right: ', pt2)
    return pt1, pt2


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


def separator_high(img, under_pos1, under_pos2):
    """ Calculate separator area between underscore "_" and bottom roof sign "^" """
    roof = Point(0, CALIBRATION_AREA_SIZE)
    width = under_pos2.x - under_pos1.x + 1

    for x in range(under_pos1.x, under_pos1.x + width):
        for y in range(under_pos2.y + 1, CALIBRATION_AREA_SIZE):
            if img[y, x] != BLACK and y < roof.y:
                roof = Point(x, y)

    high = roof.y - under_pos2.y

    print('Separator high: ', high)
    return high


def draw_filled_cell(img, start_pt, width, high):
    """ Just for debug purpose, will cell with color """
    for x in range(start_pt.x, start_pt.x + width):
        for y in range(start_pt.y, start_pt.y + high):
            img[y, x] ^= 158


def draw_net(img, start_pt, width, high):
    """ Just for debug purpose draw net """
    BLUE = (255, 0, 0)
    for x in range(start_pt.x, img.shape[1], width):
        cv2.line(img, (x, start_pt.y), (x, img.shape[0]), BLUE, 1)

    for y in range(start_pt.y, img.shape[0], high):
        cv2.line(img, (start_pt.x, y), (img.shape[1], y), BLUE, 1)


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

    start_pt, width, high = calibration_data(gray_img)
    erase_calibration_area(gray_img)

    connect_nearby_contours(gray_img)

    color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    draw_filled_cell(color_img, start_pt, width, high)
    draw_net(color_img, start_pt, width, high)
    cv2.imshow('color_img', color_img)

    cv2.imshow('orig_img', orig_img)
    cv2.imshow('gray_img', gray_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
