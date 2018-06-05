#! /usr/bin/env python
# -*- coding: utf-8 -*-

import collections as co
import itertools as it
import numpy as np
import math
import copy
import cv2


Point = co.namedtuple('Point', ['x', 'y'])
Size = co.namedtuple('Size', ['width', 'height'])


CALIBRATION_AREA_SIZE = 40
BLACK = 0
WHITE = 255
BRAILLE_CELL_SIZE = Size(2, 4)


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

    width = under_br.x - under_tl.x + 1
    height = under_br.y - roof.y + sep_height
    cell_size = Size(width, height)

    start_pt = Point(under_tl.x, under_br.y - height + 1)
    end_pt = Point(start_pt.x + ((img.shape[1] - start_pt.x) // cell_size.width) * cell_size.width,
        start_pt.y + ((img.shape[0] - start_pt.y) // cell_size.height) * cell_size.height)

    print('Cell top-left: ' + str(start_pt))
    print('Cell bottom-right: ' +  str(end_pt))
    print('Cell size: ' + str(cell_size))
    return start_pt, end_pt, cell_size


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

    print('Underscore top-left: ' + str(under_tl))
    print('Underscore bottom-right: ' + str(under_rb))
    return under_tl, under_rb


def roof_pos(img, under_tl, under_br):
    """ Calculate roof sign "^" position - only the pick """
    roof = Point(0, CALIBRATION_AREA_SIZE)
    width = under_br.x - under_tl.x + 1

    for x in range(under_br.x + 1, under_br.x + width):
        for y in range(CALIBRATION_AREA_SIZE):
            if img[y, x] != BLACK and y < roof.y:
                roof = Point(x, y)

    print('Roof pos: ' + str(roof))
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

    print('Separator height: ' + str(height))
    return height


def erase_calibration_area(img):
    """ Erase calibration are from image """
    cv2.rectangle(img, (0, 0), (CALIBRATION_AREA_SIZE, CALIBRATION_AREA_SIZE), BLACK, cv2.FILLED)


def draw_filled_cell(img, pt, cell_size):
    """ Just for debug purpose, will cell with color """
    for x in range(pt.x, pt.x + cell_size.width):
        for y in range(pt.y, pt.y + cell_size.height):
            img[y, x] ^= 158


def draw_grid(img, start_pt, end_pt, cell_size):
    """ Just for debug purpose draw grid that separate cells """
    BLUE = (255, 0, 0)
    for x in range(start_pt.x, end_pt.x + 1, cell_size.width):
        cv2.line(img, (x, start_pt.y), (x, end_pt.y), BLUE, 1)

    for y in range(start_pt.y, end_pt.y + 1, cell_size.height):
        cv2.line(img, (start_pt.x, y), (end_pt.x, y), BLUE, 1)


def draw_braille_dots(out_img, braille_arr, start_pt, end_pt, cell_size):
    """ Just for debug purpose - draw braille dots in cell """
    dot_field_size = Size(cell_size.width//BRAILLE_CELL_SIZE.width, cell_size.height//BRAILLE_CELL_SIZE.height)

    for cx, x in enumerate(range(start_pt.x, end_pt.x, cell_size.width)):
        for cy, y in enumerate(range(start_pt.y, end_pt.y, cell_size.height)):

            for bx, by in it.product(range(BRAILLE_CELL_SIZE.width), range(BRAILLE_CELL_SIZE.height)):
                if braille_arr[cy*BRAILLE_CELL_SIZE.height + by, cx*BRAILLE_CELL_SIZE.width + bx]:
                    center = Point(x+dot_field_size.width*bx + dot_field_size.width//2,
                        y+dot_field_size.height*by + dot_field_size.height//2)
                    cv2.circle(out_img, center, 2, (0, 0, 255), -1)


def draw_contour(img, contour):
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for c in contour:
        out[c.y, c.x] = (0, 0, 255)

    cv2.imshow('out', out)


def braille_array(img, start_pt, end_pt, cell_size):
    """ Extract braille data - dots that cover chars (any pixel in dot field is none zero) in all cell """
    height = ((end_pt.y - start_pt.y) // cell_size.height) * BRAILLE_CELL_SIZE.height
    width = ((end_pt.x - start_pt.x) // cell_size.width) * BRAILLE_CELL_SIZE.width
    braille_arr = np.zeros(shape=[height, width], dtype=img.dtype)

    for bx, x in enumerate(range(start_pt.x, end_pt.x, cell_size.width)):
        for by, y in enumerate(range(start_pt.y, end_pt.y, cell_size.height)):
            cell = img[y:y+cell_size.height, x:x+cell_size.width]

            braille_cell = braille_in_cell(cell, cell_size)
            x1, x2 = bx*BRAILLE_CELL_SIZE.width, bx*BRAILLE_CELL_SIZE.width+BRAILLE_CELL_SIZE.width
            y1, y2 = by*BRAILLE_CELL_SIZE.height, by*BRAILLE_CELL_SIZE.height+BRAILLE_CELL_SIZE.height
            braille_arr[y1:y2, x1:x2] = braille_cell

    return braille_arr


def braille_in_cell(cell, cell_size):
    """ Extract braille data - dots that cover chars in cell """
    dot_field_size = Size(cell_size.width//BRAILLE_CELL_SIZE.width, cell_size.height//BRAILLE_CELL_SIZE.height)
    braille_cell = np.zeros([BRAILLE_CELL_SIZE.height, BRAILLE_CELL_SIZE.width], dtype=cell.dtype)

    for bx, x in enumerate(np.linspace(0, cell_size.width, BRAILLE_CELL_SIZE.width, endpoint=False)):
        for by, y in enumerate(np.linspace(0, cell_size.height, BRAILLE_CELL_SIZE.height, endpoint=False)):
            y1, y2 = int(y), int(y)+dot_field_size.height
            x1, x2 = int(x), int(x)+dot_field_size.width
            dot_field = cell[y1:y2, x1:x2]

            if dot_field.any():
                braille_cell[by, bx] = WHITE
            else:
                braille_cell[by, bx] = BLACK

    return braille_cell


def connect_nearby_contours(img):
    """
    Connect nearby contours (ASCII characters)
    See also:
    https://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them
    http://answers.opencv.org/question/169492/accessing-all-points-of-a-contour/
    """
    gray_img = copy.copy(img)
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

    cont_img = np.zeros_like(gray_img)
    unified = [np.vstack(chain)]
    cv2.drawContours(cont_img, unified, -1, WHITE, 1)

    return cont_img


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


def contour_points(img, start_pt, end_pt):
    cont_img = copy.copy(img)
    _, contours, _ = cv2.findContours(cont_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return [Point(c[0, 0], c[0, 1]) for c in np.vstack(contours)]


def aprox(contour, start_pt, end_pt, cell_size):
    height = ((end_pt.y - start_pt.y)//cell_size.height) * BRAILLE_CELL_SIZE.height
    width = ((end_pt.x - start_pt.x)//cell_size.width) * BRAILLE_CELL_SIZE.width
    vector_arr = np.zeros(shape=[height, width, 2], dtype=np.float32)

    first_point, last_point = None, None
    for c in contour:
        if not first_point:
            first_point = c

        if in_dot_boundry(first_point, c, start_pt, cell_size):
            last_point = c
        elif last_point:
            print 'jest'
            norm_vec = calculate_norm_vector(first_point, last_point)
            pos = array_pos(first_point, start_pt, cell_size)
            vector_arr[pos.y, pos.x] = norm_vec

            first_point = c
            last_point = None


def in_dot_boundry(pt, test_pt, start_pt, cell_size):
    w = cell_size.width/float(BRAILLE_CELL_SIZE.width)
    h = cell_size.height/float(BRAILLE_CELL_SIZE.height)
    x = start_pt.x + ((pt.x - start_pt.x)//w) * w
    y = start_pt.y + ((pt.y - start_pt.y)//h) * h
    tl_pt = Point(int(x), int(y))
    br_pt = Point(int(x + w), int(y + h))
    print tl_pt, br_pt, test_pt

    if tl_pt.x <= test_pt.x < br_pt.x and tl_pt.y <= test_pt.y < br_pt.y:
        print 'ok'
    else:
        print 'nie ok'
    return tl_pt.x <= test_pt.x < br_pt.x and tl_pt.y <= test_pt.y < br_pt.y


def calculate_norm_vector(pt1, pt2):
    # calculation tangent line (ax + by + c = 0) to points
    if pt2.x - pt1.x == 0:
        a = 1.0
        b = 0.0
    else:
        a = (pt2.y - pt1.y)/float(pt2.x - pt1.x)
        b = 1.0

    # normalized perpendicular vector to line (ax + by + c = 0) equal to v = [-a, b]
    mag = math.sqrt(a**2 + b**2)
    if pt1.x <= pt2.x or pt1.y < pt2.y:
        return np.array([-a/mag, b/mag])

    return np.array([a/mag, -b/mag])


def array_pos(pt, start_pt, cell_size):
    # x = ((pt.x - start_pt.x)//cell_size.width) * BRAILLE_CELL_SIZE.width + (pt.x - start_pt.x)%BRAILLE_CELL_SIZE.width
    # y = ((pt.y - start_pt.y)//cell_size.height) * BRAILLE_CELL_SIZE.height + (pt.y - start_pt.y)%BRAILLE_CELL_SIZE.height
    w = cell_size.width/float(BRAILLE_CELL_SIZE.width)
    h = cell_size.height/float(BRAILLE_CELL_SIZE.height)
    x = (pt.x - start_pt.x)//w
    y = (pt.y - start_pt.y)//h
    return Point(int(x), int(y))


def export_contour_img(file_name, img, start_pt, end_pt):
    """ Export contour image to file """
    out_img = img[start_pt.y:end_pt.y, start_pt.x:end_pt.x]
    np.savetxt(file_name+'.contour', out_img, fmt='%02x')


def export_braille_data(file_name, braille_arr):
    """ Export braille data to file """
    np.savetxt(file_name+'.braille', braille_arr, fmt='%02x')


def main():
    file_name = 'ascii_fig.png'
    orig_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    # Processing should be on the image with a black background and white foreground
    _, gray_img= cv2.threshold(src=orig_img, thresh=30, maxval=255, type=cv2.THRESH_BINARY)

    start_pt, end_pt, cell_size = calibration_data(gray_img)
    erase_calibration_area(gray_img)

    cont_img = connect_nearby_contours(gray_img)
    export_contour_img(file_name, cont_img, start_pt, end_pt)

    contour = contour_points(cont_img, start_pt, end_pt)
    draw_contour(orig_img, contour)
    aprox(contour, start_pt, end_pt, cell_size)

    braille_arr = braille_array(gray_img, start_pt, end_pt, cell_size)
    export_braille_data(file_name, braille_arr)

    debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    # draw_filled_cell(orig_img, start_pt, cell_size)
    draw_braille_dots(debug_img, braille_arr, start_pt, end_pt, cell_size)
    draw_grid(debug_img, start_pt, end_pt, cell_size)

    cv2.imshow('debug_img', debug_img)
    cv2.imshow('orig_img', orig_img)
    cv2.imshow('cont_img', cont_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
