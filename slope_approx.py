#! /usr/bin/env python
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


def draw_chars_areas(out_img, in_img, start_pt, end_pt, cell_size):
    """ Mark areas (with dimensions of braille dot field) if any part of chars is in this area """
    for x in range(start_pt.x, end_pt.x, cell_size.width):
        for y in range(start_pt.y, end_pt.y, cell_size.height):
            cell = in_img[y:y+cell_size.height, x:x+cell_size.width]
            if cell.any():
                draw_braille_dots(out_img, in_img, Point(x, y), cell_size)


def draw_braille_dots(out_img, in_img, pt, cell_size):
    """ Just for debug purpose - draw braille dots in cell if any pixel in dot field is none zero
    (is part of character).
    """
    braille_cell = Size(2.0, 4.0)
    dot_field_size = Size(cell_size.width/braille_cell.width, cell_size.height/braille_cell.height)

    for x in np.linspace(pt.x, pt.x + cell_size.width, braille_cell.width, endpoint=False):
        for y in np.linspace(pt.y, pt.y + cell_size.height, braille_cell.height, endpoint=False):
            y1, y2 = int(y), int(y+dot_field_size.height)
            x1, x2 = int(x), int(x+dot_field_size.width)
            field = in_img[y1:y2, x1:x2]
            if field.any():
                center = Point(x1 + int(dot_field_size.width/2), y1 + int(dot_field_size.height/2))
                cv2.circle(out_img, center, 2, (0, 0, 255), -1)
                # out_img[y1:y2, x1:x2] = (255, 250, 0)


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


def export_contour_img(file_name, img, start_pt, end_pt):
    out_img = img[start_pt.y:end_pt.y, start_pt.x:end_pt.x]
    np.savetxt(file_name+'.contour', out_img, fmt='%02x')


def approximate_slope(img, start_pt, end_pt, cell_size):
    for x in range(start_pt.x, end_pt.x, cell_size.width):
        for y in range(start_pt.y, end_pt.y, cell_size.height):
            cell = img[y:y+cell_size.height, x:x+cell_size.width]
            if cell.any():
                # draw_braille_dots(in_img, out_img, Point(x, y), cell_size)
                pass


def aaa(img, pt, cell_size):
    braille_cell = Size(2.0, 4.0)
    dot_size = Size(cell_size.width/braille_cell.width, cell_size.height/braille_cell.height)

    for x in np.linspace(pt.x, pt.x + cell_size.width, braille_cell.width, endpoint=False):
        for y in np.linspace(pt.y, pt.y + cell_size.height, braille_cell.height, endpoint=False):
            y1, y2 = int(y), int(y+dot_size.height)
            x1, x2 = int(x), int(x+dot_size.width)
            sector = img[y1:y2, x1:x2]
            if sector.any():
                pass
                # out_img[y1:y2, x1:x2] = (255, 250, 0)

def main():
    file_name = 'ascii_fig.png'
    orig_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    # Image should have white characters and black background
    _, gray_img= cv2.threshold(src=orig_img, thresh=30, maxval=255, type=cv2.THRESH_BINARY)

    start_pt, end_pt, cell_size = calibration_data(gray_img)
    erase_calibration_area(gray_img)

    cont_img = connect_nearby_contours(gray_img)
    export_contour_img(file_name, cont_img, start_pt, end_pt)

    approximate_slope(cont_img, start_pt, end_pt, cell_size)

    debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    # draw_filled_cell(orig_img, start_pt, cell_size)
    draw_chars_areas(debug_img, gray_img, start_pt, end_pt, cell_size)
    # for y in range(start_pt.y + cell_size.height * 8):
        # print 'value ' +str(y) + ' ' + str(debug_img[y, 7])
    # print 'value 6:' +str(y) + ' ' + str(debug_img[126, 6])
    draw_net(debug_img, start_pt, end_pt, cell_size)
    # draw_braille_dots(gray_img, debug_img, start_pt, cell_size)
    cv2.imshow('debug_img', debug_img)


    cv2.imshow('orig_img', orig_img)
    cv2.imshow('cont_img', cont_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
