#! /usr/bin/env python
# -*- coding: utf-8 -*-

import collections as co
import itertools as it
import numpy as np
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


def draw_net(img, start_pt, end_pt, cell_size):
    """ Just for debug purpose draw net """
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


def contour_moor_neighborhood(cont_img, start_pt, end_pt):
    """
    https://en.wikipedia.org/wiki/Moore_neighborhood
    """
    for y, x in it.product(reversed(range(start_pt.y, end_pt.y)), range(start_pt.x, end_pt.x)):
        if cont_img[y, x] == WHITE:
            start = Point(x, y)
            break
        else:
            backtrack = Point(x, y)

    contour = [start]
    current_boundry = start
    current = next_clockwise(current_boundry, backtrack)

    while current != start:
        if cont_img[current.y, current.x] == WHITE:
            contour.append(current)
            backtrack = current_boundry
            current_boundry = current
            current = next_clockwise(current_boundry, backtrack)
        else:
            backtrack = current
            current = next_clockwise(current_boundry, backtrack)

    return contour


def next_clockwise(current_boundry, backtrack):
    if backtrack.x < current_boundry.x:
        if backtrack.y >= current_boundry.y:
            return Point(backtrack.x, backtrack.y-1)
        else:
            return Point(backtrack.x+1, backtrack.y)
    elif backtrack.x == current_boundry.x:
        if backtrack.y < current_boundry.y:
            return Point(backtrack.x+1, backtrack.y)
        else:
            return Point(backtrack.x-1, backtrack.y)
    else:
        if backtrack.y <= current_boundry.y:
            return Point(backtrack.x, backtrack.y+1)
        else:
            return Point(backtrack.x-1, backtrack.y)


def aprox(contour, delta=2):
    return

    for i, c in enumerate(contour):
        points = [c,contour[i+3]]
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        # y = ax + b
        a, b = np.linalg.lstsq(A, y_coords)[0]
        # perpendicular vector v=[A, B] to line Ax + By + C = 0
        vec = [a, 1]
        # normalize
    pass

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

    contour = contour_moor_neighborhood(cont_img, start_pt, end_pt)
    aprox(contour)


    braille_arr = braille_array(gray_img, start_pt, end_pt, cell_size)
    export_braille_data(file_name, braille_arr)

    debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    # draw_filled_cell(orig_img, start_pt, cell_size)
    draw_braille_dots(debug_img, braille_arr, start_pt, end_pt, cell_size)
    draw_net(debug_img, start_pt, end_pt, cell_size)

    cv2.imshow('debug_img', debug_img)
    cv2.imshow('orig_img', orig_img)
    cv2.imshow('cont_img', cont_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
