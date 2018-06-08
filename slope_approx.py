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
Grid = co.namedtuple('Grid', ['start', 'end', 'cell_size'])


CALIBRATION_AREA_SIZE = 40
BLACK = 0
WHITE = 255
BRAILLE_CELL_SIZE = Size(2, 4)


def grid_data(img):
    """
    Calculate grid data .
    Return:
        top-left point where first cell (grid) start
        bottom-right point where last cell (grid) end
        cell size (width, height)
    """
    under_tl_pt, under_br_pt = underscore_pos(img)
    roof_pt = roof_pos(img, under_tl_pt, under_br_pt)
    sep_height = separator_height(img, under_tl_pt, under_br_pt)

    width = under_br_pt.x - under_tl_pt.x + 1
    height = under_br_pt.y - roof_pt.y + sep_height
    cell_size = Size(width, height)

    start_pt = Point(under_tl_pt.x, under_br_pt.y - height + 1)
    end_pt = Point(start_pt.x + ((img.shape[1] - start_pt.x) // cell_size.width) * cell_size.width,
        start_pt.y + ((img.shape[0] - start_pt.y) // cell_size.height) * cell_size.height)

    grid = Grid(start_pt, end_pt, cell_size)
    print('Cell top-left: ' + str(grid.start))
    print('Cell bottom-right: ' +  str(grid.end))
    print('Cell size: ' + str(grid.cell_size))
    return grid


def underscore_pos(img):
    """ Calculate underscore "_" position [top-left point, bottom-right point] """
    under_tl_pt = None
    for x, y in it.product(range(CALIBRATION_AREA_SIZE), range(CALIBRATION_AREA_SIZE)):
        if img[y, x] != BLACK:
            under_tl_pt = Point(x, y)
            break

    tmp = None
    for x in range(under_tl_pt.x, CALIBRATION_AREA_SIZE):
        if img[under_tl_pt.y, x] == BLACK:
            break
        tmp = Point(x, under_tl_pt.y)

    under_br_pt = None
    for y in range(tmp.y, CALIBRATION_AREA_SIZE):
        if img[y, tmp.x] == BLACK:
            break
        under_br_pt = Point(tmp.x, y)

    print('Underscore top-left: ' + str(under_tl_pt))
    print('Underscore bottom-right: ' + str(under_br_pt))
    return under_tl_pt, under_br_pt


def roof_pos(img, under_tl_pt, under_br_pt):
    """ Calculate roof sign "^" position - only the pick """
    roof_pt = Point(0, CALIBRATION_AREA_SIZE)
    width = under_br_pt.x - under_tl_pt.x + 1

    for x in range(under_br_pt.x + 1, under_br_pt.x + width):
        for y in range(CALIBRATION_AREA_SIZE):
            if img[y, x] != BLACK and y < roof_pt.y:
                roof_pt = Point(x, y)

    print('Roof pos: ' + str(roof_pt))
    return roof_pt


def separator_height(img, under_tl_pt, under_br_pt):
    """ Calculate separator area between underscore "_" and bottom roof sign "^" """
    roof_pt = Point(0, CALIBRATION_AREA_SIZE)
    width = under_br_pt.x - under_tl_pt.x + 1

    for x in range(under_tl_pt.x, under_tl_pt.x + width):
        for y in range(under_br_pt.y + 1, CALIBRATION_AREA_SIZE):
            if img[y, x] != BLACK and y < roof_pt.y:
                roof_pt = Point(x, y)

    height = roof_pt.y - under_br_pt.y

    print('Separator height: ' + str(height))
    return height


def erase_calibration_area(img):
    """ Erase calibration are from image """
    cv2.rectangle(img, (0, 0), (CALIBRATION_AREA_SIZE, CALIBRATION_AREA_SIZE), BLACK, cv2.FILLED)


def draw_filled_cell(img, pt, grid):
    """ Just for debug purpose, fill cell with color """
    for x in range(pt.x, pt.x + grid.cell_size.width):
        for y in range(pt.y, pt.y + grid.cell_size.height):
            img[y, x] ^= 158


def draw_grid(img, grid):
    """ Just for debug purpose draw grid that separate cells """
    BLUE = (255, 0, 0)
    for x in range(grid.start.x, grid.end.x + 1, grid.cell_size.width):
        cv2.line(img, (x, grid.start.y), (x, grid.end.y), BLUE, 1)

    for y in range(grid.start.y, grid.end.y + 1, grid.cell_size.height):
        cv2.line(img, (grid.start.x, y), (grid.end.x, y), BLUE, 1)


def draw_braille_dots(img, arr, grid):
    """ Just for debug purpose - if array element of corresponding braille dot is not zero, draw it """
    RED = (0, 0, 255)
    x_samples = ((grid.end.x - grid.start.x)/grid.cell_size.width) * float(BRAILLE_CELL_SIZE.width)
    y_samples = ((grid.end.y - grid.start.y)/grid.cell_size.height) * float(BRAILLE_CELL_SIZE.height)
    dot_field_size = Size(grid.cell_size.width/float(BRAILLE_CELL_SIZE.width),
                          grid.cell_size.height/float(BRAILLE_CELL_SIZE.height))

    for bx, x in enumerate(np.linspace(grid.start.x, grid.end.x, x_samples, endpoint=False)):
        for by, y in enumerate(np.linspace(grid.start.y, grid.end.y, y_samples, endpoint=False)):
            if (arr[by, bx] != 0).any():
                center = Point(int(x + dot_field_size.width//2), int(y + dot_field_size.height//2))
                cv2.circle(img, center, radius=2, color=RED, thickness=-1)


def draw_contour(img, contour):
    for c in contour:
        img[c.y, c.x] = (0, 255, 255)


def draw_normal_vec(img, arr, grid):
    GREEN = (0, 255, 0)
    FACTOR = 20
    x_samples = ((grid.end.x - grid.start.x)/grid.cell_size.width) * float(BRAILLE_CELL_SIZE.width)
    y_samples = ((grid.end.y - grid.start.y)/grid.cell_size.height) * float(BRAILLE_CELL_SIZE.height)
    dot_field_size = Size(grid.cell_size.width/float(BRAILLE_CELL_SIZE.width),
                          grid.cell_size.height/float(BRAILLE_CELL_SIZE.height))

    for bx, x in enumerate(np.linspace(grid.start.x, grid.end.x, x_samples, endpoint=False)):
        for by, y in enumerate(np.linspace(grid.start.y, grid.end.y, y_samples, endpoint=False)):
            if (arr[by, bx] != 0).any():
                start = Point(int(x + dot_field_size.width//2), int(y + dot_field_size.height//2))
                # Y with minus, because OpenCV use different coordinate system
                vec_end = Point(arr[by, bx][0], -arr[by, bx][1])
                end = Point(start.x + int(vec_end.x*FACTOR), start.y + int(vec_end.y*FACTOR))
                cv2.line(img, start, end, GREEN, 1)


def braille_array(img, grid):
    """ Extract braille data - dots that cover chars (any pixel in dot field is none zero) in all cell """
    height = ((grid.end.y - grid.start.y) // grid.cell_size.height) * BRAILLE_CELL_SIZE.height
    width = ((grid.end.x - grid.start.x) // grid.cell_size.width) * BRAILLE_CELL_SIZE.width
    braille_arr = np.zeros(shape=[height, width], dtype=img.dtype)

    for cx, x in enumerate(range(grid.start.x, grid.end.x, grid.cell_size.width)):
        for cy, y in enumerate(range(grid.start.y, grid.end.y, grid.cell_size.height)):
            cell = img[y:y+grid.cell_size.height, x:x+grid.cell_size.width]

            braille_cell = braille_in_cell(cell, grid)
            x1, x2 = cx*BRAILLE_CELL_SIZE.width, cx*BRAILLE_CELL_SIZE.width+BRAILLE_CELL_SIZE.width
            y1, y2 = cy*BRAILLE_CELL_SIZE.height, cy*BRAILLE_CELL_SIZE.height+BRAILLE_CELL_SIZE.height
            braille_arr[y1:y2, x1:x2] = braille_cell

    return braille_arr


def braille_in_cell(cell, grid):
    """ Extract braille data - dots that cover chars in cell """
    dot_field_size = Size(grid.cell_size.width//BRAILLE_CELL_SIZE.width, grid.cell_size.height//BRAILLE_CELL_SIZE.height)
    braille_cell = np.zeros([BRAILLE_CELL_SIZE.height, BRAILLE_CELL_SIZE.width], dtype=cell.dtype)

    for bx, x in enumerate(np.linspace(0, grid.cell_size.width, BRAILLE_CELL_SIZE.width, endpoint=False)):
        for by, y in enumerate(np.linspace(0, grid.cell_size.height, BRAILLE_CELL_SIZE.height, endpoint=False)):
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

    # First contour position is at the bottom of image
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


def smooth_contours(img):
    kernel_dil = np.ones((3, 3), np.uint8)
    kernel_ero = np.ones((2, 2), np.uint8)

    dilation_img = cv2.dilate(img, kernel_dil, iterations=1)
    erosion_img = cv2.erode(dilation_img, kernel_ero, iterations=1)

    # blur = cv2.GaussianBlur(erosion_img, (3, 3), 0)

    return erosion_img


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


def contour_points(img):
    cont_img = copy.copy(img)
    _, contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [Point(c[0, 0], c[0, 1]) for c in np.vstack(contours)]


def approx_surface_slope(contour, grid):
    height = ((grid.end.y - grid.start.y)//grid.cell_size.height) * BRAILLE_CELL_SIZE.height
    width = ((grid.end.x - grid.start.x)//grid.cell_size.width) * BRAILLE_CELL_SIZE.width
    norm_vec_arr = np.zeros(shape=[height, width, 2], dtype=np.float32)

    first_pt, last_pt = None, None
    for c in contour:
        if not first_pt:
            first_pt = c
            continue

        if in_dot_field(first_pt, c, grid):
            last_pt = c
        elif last_pt:
            # print 'jest'
            norm_vec = calculate_norm_vector(first_pt, last_pt)
            pos = array_pos(first_pt, grid)
            norm_vec_arr[pos.y, pos.x] = norm_vec

            first_pt = c
            last_pt = None
            # exit()
        else:
            # print 'ups'
            pass

    return norm_vec_arr


def in_dot_field(first_pt, test_pt, grid):
    width = grid.cell_size.width/float(BRAILLE_CELL_SIZE.width)
    height = grid.cell_size.height/float(BRAILLE_CELL_SIZE.height)
    x = grid.start.x + math.ceil((first_pt.x - grid.start.x)/width) * width
    y = grid.start.y + math.ceil((first_pt.y - grid.start.y)/height) * height

    if test_pt.x < int(x):
        x -= width
    if test_pt.y < int(y):
        y -= height

    tl_pt = Point(int(x), int(y))
    br_pt = Point(int(x + width), int(y + height))

    return tl_pt.x <= test_pt.x < br_pt.x and tl_pt.y <= test_pt.y < br_pt.y


def calculate_norm_vector(pt1, pt2):
    # calculation tangent line (ax + by + c = 0) to points
    # Y should be with minus, because OpenCV use different coordinate system
    if pt2.x - pt1.x == 0:
        a = 1.0 if pt2.y > pt1.y else -1.0
        b = 0.0
    else:
        a = (-pt2.y + pt1.y)/float(pt2.x - pt1.x)
        b = 1.0

    # normalized perpendicular vector to line (ax + by + c = 0) equal to v = [-a, b]
    mag = math.sqrt(a**2 + b**2)
    if pt2.x <= pt1.x:
        return np.array([-a/mag, b/mag])
    return np.array([a/mag, -b/mag])


def array_pos(pt, grid):
    width = grid.cell_size.width/float(BRAILLE_CELL_SIZE.width)
    height = grid.cell_size.height/float(BRAILLE_CELL_SIZE.height)
    x = (pt.x - grid.start.x)//width
    y = (pt.y - grid.start.y)//height
    return Point(int(x), int(y))


def export_braille_data(file_name, braille_arr):
    """ Export braille data to file """
    np.savetxt(file_name+'.braille', braille_arr, fmt='%02x')


def main():
    file_name = 'ascii_fig.png'
    term_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    # Processing should be on the image with a black background and white foreground
    _, gray_img= cv2.threshold(src=term_img, thresh=30, maxval=255, type=cv2.THRESH_BINARY)

    grid = grid_data(gray_img)
    erase_calibration_area(gray_img)

    cont_img = connect_nearby_contours(gray_img)
    cont_img = smooth_contours(cont_img)

    contour = contour_points(cont_img)
    norm_vec_arr = approx_surface_slope(contour, grid)

    braille_arr = braille_array(gray_img, grid)
    export_braille_data(file_name, braille_arr)

    debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    # draw_filled_cell(term_img, start_pt, cell_size)
    draw_braille_dots(debug_img, norm_vec_arr, grid)
    draw_normal_vec(debug_img, norm_vec_arr, grid)
    # draw_grid(debug_img, grid)
    draw_contour(debug_img, contour)

    cv2.imshow('debug_img', debug_img)
    cv2.imshow('term_img', term_img)
    cv2.imshow('cont_img', cont_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
