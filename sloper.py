#! /usr/bin/env python3

import collections as co
import itertools as it
import numpy as np
import math
import copy
import cv2


Point = co.namedtuple('Point', ['x', 'y'])
Size = co.namedtuple('Size', ['height', 'width'])
Grid = co.namedtuple('Grid', ['start', 'end', 'cell_size'])


CALIBRATION_AREA_SIZE = 40
FACTOR = 20
BUF_CELL = Size(height=4, width=2)
VECTOR_DIM = 2

BLACK_1D = 0
WHITE_1D = 255
BLUE_3D = (255, 0, 0)
RED_3D = (0, 0, 255)
GREEN_3D = (0, 255, 0)
YELLOW_3D = (0, 255, 255)


def main():
    file_name = 'ascii_fig.png'
    term_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    # Processing should be on the image with a black background and white
    # foreground
    _, gray_img = cv2.threshold(src=term_img, thresh=30, maxval=255, type=cv2.THRESH_BINARY)

    grid = grid_data(gray_img)
    erase_calibration_area(gray_img)

    cont_img = connect_nearby_contours(gray_img)
    cont_img = smooth_contours(cont_img)

    contour = contour_points(cont_img)
    normal_vec_arr = approximate_surface_slopes(contour, grid)

    braille_arr = braille_array(gray_img, grid)
    export_braille_data(file_name, braille_arr)
    export_normal_vec_arr(file_name, normal_vec_arr)

    debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    # draw_filled_cell(term_img, grid.start, grid)
    # draw_braille_dots(debug_img, normal_vec_arr, grid)
    draw_braille_normal_vec(debug_img, normal_vec_arr, grid)
    # draw_grid(debug_img, grid)
    draw_contour(debug_img, contour)

    cv2.imshow('Normal vectors', debug_img)
    cv2.imshow('Terminal (ASCII) image', term_img)
    cv2.imshow('Contours', cont_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def grid_data(img):
    """
    Calculate grid data.
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
    cell_size = Size(height, width)

    start_pt = Point(under_tl_pt.x, under_br_pt.y - height + 1)
    end_pt = Point(start_pt.x + ((img.shape[1] - start_pt.x) // cell_size.width) * cell_size.width,
                   start_pt.y + ((img.shape[0] - start_pt.y) // cell_size.height) * cell_size.height)

    grid = Grid(start_pt, end_pt, cell_size)
    print('[+] Cell top-left:', grid.start)
    print('[+] Cell bottom-right:', grid.end)
    print('[+] Cell size:', grid.cell_size)
    return grid


def underscore_pos(img):
    """
    Return underscore ('_') position (top-left point, bottom-right point).
    """
    under_tl_pt = None
    for x, y in it.product(range(CALIBRATION_AREA_SIZE), range(CALIBRATION_AREA_SIZE)):
        if img[y, x] != BLACK_1D:
            under_tl_pt = Point(x, y)
            break

    tmp = None
    for x in range(under_tl_pt.x, CALIBRATION_AREA_SIZE):
        if img[under_tl_pt.y, x] == BLACK_1D:
            break
        tmp = Point(x, under_tl_pt.y)

    under_br_pt = None
    for y in range(tmp.y, CALIBRATION_AREA_SIZE):
        if img[y, tmp.x] == BLACK_1D:
            break
        under_br_pt = Point(tmp.x, y)

    print('[+] Underscore top-left:', under_tl_pt)
    print('[+] Underscore bottom-right:', under_br_pt)
    return under_tl_pt, under_br_pt


def roof_pos(img, under_tl_pt, under_br_pt):
    """Return roof sign '^' position (the pick point)."""
    roof_pt = Point(0, CALIBRATION_AREA_SIZE)
    width = under_br_pt.x - under_tl_pt.x + 1

    for x in range(under_br_pt.x + 1, under_br_pt.x + width):
        for y in range(CALIBRATION_AREA_SIZE):
            if img[y, x] != BLACK_1D and y < roof_pt.y:
                roof_pt = Point(x, y)

    print('[+] Roof pos:', roof_pt)
    return roof_pt


def separator_height(img, under_tl_pt, under_br_pt):
    """
    Return separator height between underscore '_' and bottom roof sign '^'.
    """
    roof_pt = Point(0, CALIBRATION_AREA_SIZE)
    width = under_br_pt.x - under_tl_pt.x + 1

    for x in range(under_tl_pt.x, under_tl_pt.x + width):
        for y in range(under_br_pt.y + 1, CALIBRATION_AREA_SIZE):
            if img[y, x] != BLACK_1D and y < roof_pt.y:
                roof_pt = Point(x, y)

    height = roof_pt.y - under_br_pt.y

    print('[+] Separator height:', height)
    return height


def erase_calibration_area(img):
    """Erase calibration are from image (fill are with black)."""
    cv2.rectangle(img, (0, 0), (CALIBRATION_AREA_SIZE, CALIBRATION_AREA_SIZE), BLACK_1D, cv2.FILLED)


def draw_filled_cell(img, pt, grid):
    """For DEBUG
    Fill buf cell with color.
    """
    for x in range(pt.x, pt.x + grid.cell_size.width):
        for y in range(pt.y, pt.y + grid.cell_size.height):
            img[y, x] ^= 158


def draw_grid(img, grid):
    """For DEBUG
    Draw grid that separate cells.
    """
    for x in range(grid.start.x, grid.end.x + 1, grid.cell_size.width):
        cv2.line(img, (x, grid.start.y), (x, grid.end.y), BLUE_3D, 1)

    for y in range(grid.start.y, grid.end.y + 1, grid.cell_size.height):
        cv2.line(img, (grid.start.x, y), (grid.end.x, y), BLUE_3D, 1)


def draw_braille_dots(img, arr, grid):
    """For DEBUG
    Draw dot for each "not zero" element of array.
    """
    foreach_arr_elements(img, arr, grid, draw_dot)


def draw_braille_normal_vec(img, arr, grid):
    """
    Draw perpendicular vector to the surface, where surface is every "non zero"
    of array (array with normal vectors). Normal vectors are multiply by some
    factor, to be better visible.
    """
    foreach_arr_elements(img, arr, grid, draw_norm_vec)


def foreach_arr_elements(img, arr, grid, draw_func):
    """
    Call draw_func() for each "non zero" array element.
    """
    x_samples = ((grid.end.x - grid.start.x)/grid.cell_size.width) * BUF_CELL.width
    y_samples = ((grid.end.y - grid.start.y)/grid.cell_size.height) * BUF_CELL.height

    for bx, x in enumerate(np.linspace(grid.start.x, grid.end.x, x_samples, endpoint=False)):
        for by, y in enumerate(np.linspace(grid.start.y, grid.end.y, y_samples, endpoint=False)):
            if np.any(arr[by, bx]):
                draw_func(img, field_pt=Point(x, y), normal_vec=arr[by, bx], grid=grid)


def draw_dot(img, field_pt, normal_vec, grid):
    """
    Draw dot (braille dot) at given point.
    """
    dot_field_size = Size(grid.cell_size.height/BUF_CELL.height,
                          grid.cell_size.width/BUF_CELL.width)
    center = Point(int(field_pt.x + dot_field_size.width//2),
                   int(field_pt.y + dot_field_size.height//2))
    cv2.circle(img, center, radius=2, color=RED_3D, thickness=-1)


def draw_norm_vec(img, field_pt, normal_vec, grid):
    """
    Draw vector (normal_vec) at given point. Basically normal_vec will be
    multiplied by FACTOR to be better visible.
    """
    dot_field_size = Size(grid.cell_size.height/BUF_CELL.height,
                          grid.cell_size.width/BUF_CELL.width)

    start = Point(int(field_pt.x + dot_field_size.width//2),
                  int(field_pt.y + dot_field_size.height//2))
    # Y with minus, because OpenCV use different coordinates order
    vec_end = Point(normal_vec[1], -normal_vec[0])
    end = Point(start.x + int(vec_end.x*FACTOR), start.y + int(vec_end.y*FACTOR))
    cv2.line(img, start, end, GREEN_3D, 1)


def draw_contour(img, contour):
    """
    Connect all counters point with lines.
    """
    for c in contour:
        img[c.y, c.x] = YELLOW_3D


def braille_array(img, grid):
    """
    Extract braille data - dots that cover chars (any pixel in dot field is
    "non zero") in all cells.
    """
    height = ((grid.end.y - grid.start.y) // grid.cell_size.height) * BUF_CELL.height
    width = ((grid.end.x - grid.start.x) // grid.cell_size.width) * BUF_CELL.width
    braille_arr = np.zeros(shape=[height, width], dtype=img.dtype)

    for cx, x in enumerate(range(grid.start.x, grid.end.x, grid.cell_size.width)):
        for cy, y in enumerate(range(grid.start.y, grid.end.y, grid.cell_size.height)):
            cell = img[y:y+grid.cell_size.height, x:x+grid.cell_size.width]

            braille_cell = braille_in_cell(cell, grid)
            x1, x2 = cx*BUF_CELL.width, cx*BUF_CELL.width+BUF_CELL.width
            y1, y2 = cy*BUF_CELL.height, cy*BUF_CELL.height+BUF_CELL.height
            braille_arr[y1:y2, x1:x2] = braille_cell

    return braille_arr


def braille_in_cell(cell, grid):
    """Extract braille data - dots that cover chars from one cell."""
    dot_field_size = Size(grid.cell_size.height//BUF_CELL.height,
                          grid.cell_size.width//BUF_CELL.width)
    braille_cell = np.zeros([BUF_CELL.height, BUF_CELL.width], dtype=cell.dtype)

    for bx, x in enumerate(np.linspace(0, grid.cell_size.width, BUF_CELL.width, endpoint=False)):
        for by, y in enumerate(np.linspace(0, grid.cell_size.height, BUF_CELL.height, endpoint=False)):
            y1, y2 = int(y), int(y)+dot_field_size.height
            x1, x2 = int(x), int(x)+dot_field_size.width
            dot_field = cell[y1:y2, x1:x2]

            if dot_field.any():
                braille_cell[by, bx] = WHITE_1D
            else:
                braille_cell[by, bx] = BLACK_1D

    return braille_cell


def connect_nearby_contours(img):
    """
    Connect nearby contours (ASCII characters).

    References:
    https://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them
    http://answers.opencv.org/question/169492/accessing-all-points-of-a-contour/
    https://docs.opencv.org/3.4.1/dd/d49/tutorial_py_contour_features.html
    """
    gray_img = copy.copy(img)
    contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # First contour position is at the bottom of image
    last = contours.pop(0)
    chain = [last]
    while len(contours) > 0:
        cnt = find_nearest(last, contours)

        if cnt is None:
            raise(Exception('Error! Contours length: %d' % len(contours)))

        chain.append(cnt)
        for i in range(len(contours)):
            if np.all(cnt == contours[i]):
                contours.pop(i)
                break
        last = cnt

    cont_img = np.zeros_like(gray_img)

    approx = cv2.approxPolyDP(np.vstack(chain), epsilon=2, closed=True)
    cv2.drawContours(cont_img, [approx], -1, WHITE_1D, 1)

    return cont_img


def smooth_contours(img):
    kernel_dil = np.ones((3, 3), np.uint8)
    kernel_ero = np.ones((2, 2), np.uint8)

    dilation_img = cv2.dilate(img, kernel_dil, iterations=1)
    erosion_img = cv2.erode(dilation_img, kernel_ero, iterations=1)

    return erosion_img


def find_nearest(head_cnt, contours, min_dist=15):
    """Find nearest contour to current head contour."""
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
    contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [Point(c[0, 0], c[0, 1]) for c in np.vstack(contours)]


def approximate_surface_slopes(contour, grid):
    height = ((grid.end.y - grid.start.y)//grid.cell_size.height) * BUF_CELL.height
    width = ((grid.end.x - grid.start.x)//grid.cell_size.width) * BUF_CELL.width
    normal_vec_arr = np.zeros(shape=[height, width, VECTOR_DIM], dtype=np.float32)

    first_pt, last_pt = None, None
    normal_vec = np.array([0, 0])
    for c in contour:
        if not first_pt:
            first_pt = c
            continue

        idx, tl_pt, br_pt = dot_field(first_pt, grid)
        if in_boundaries(c, tl_pt, br_pt):
            last_pt = c
        elif last_pt:
            normal_vec = calc_normal_unit_vec(first_pt, last_pt)
            normal_vec_arr[idx.y, idx.x] = normal_vec
            first_pt = c
            last_pt = None
        else:
            normal_vec_arr[idx.y, idx.x] = normal_vec
            first_pt = c
            last_pt = None

    return normal_vec_arr


def in_boundaries(test_pt, tl_pt, br_pt):
    return tl_pt.x <= test_pt.x < br_pt.x and tl_pt.y <= test_pt.y < br_pt.y


def calc_normal_unit_vec(pt1, pt2):
    # Calculation tangent line (ax + by + c = 0) to points
    # Y should be with minus, because OpenCV use different coordinate system
    if pt2.x - pt1.x == 0:
        a = 1.0 if pt2.y > pt1.y else -1.0
        b = 0.0
    else:
        a = (-pt2.y + pt1.y)/float(pt2.x - pt1.x)
        b = 1.0

    # Normalized (unit) perpendicular vector to line (ax + by + c = 0)
    # equal to v = [-a, b].
    # Values as stored in numpy.array where by convention dimensions should
    # start from Y followed by X [Y, X]
    mag = math.sqrt(a**2 + b**2)
    if pt2.x <= pt1.x:
        return np.array([b/mag, -a/mag])
    return np.array([-b/mag, a/mag])


def dot_field(pt, grid):
    width = grid.cell_size.width / BUF_CELL.width
    height = grid.cell_size.height / BUF_CELL.height

    idx = Point(int((pt.x - grid.start.x)/width),
                int((pt.y - grid.start.y)/height))
    x = grid.start.x + idx.x * width
    y = grid.start.y + idx.y * height

    if int(x + width) <= pt.x:
        idx = Point(idx.x + 1, idx.y)
    if int(y + height) <= pt.y:
        idx = Point(idx.x, idx.y + 1)

    tl_pt = Point(int(grid.start.x + idx.x * width),
                  int(grid.start.y + idx.y * height))
    br_pt = Point(int(grid.start.x + (idx.x + 1) * width),
                  int(grid.start.y + (idx.y + 1) * height))

    return idx, tl_pt, br_pt


def export_braille_data(file_name, braille_arr):
    """Export braille data to file."""
    np.savetxt(file_name+'.braille', braille_arr, fmt='%02x')


def export_normal_vec_arr(file_name, arr):
    """Export braille data to file."""
    height, width, vec_dim = arr.shape
    np.savetxt(file_name+'.norm', arr.reshape([height, width*vec_dim]), fmt='%.04f')


if __name__ == '__main__':
    main()
