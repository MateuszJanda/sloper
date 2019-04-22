#! /usr/bin/env python3

import traceback
import argparse
from PIL import Image, ImageDraw, ImageFont
import collections as co
import itertools as it
import numpy as np
import math
import cv2


Point = co.namedtuple('Point', ['x', 'y'])
Size = co.namedtuple('Size', ['height', 'width'])
Grid = co.namedtuple('Grid', ['start', 'end', 'cell'])


VEC_FACTOR = 20
SCR_CELL_SIZE = Size(height=4, width=2)
NORM_VEC_DIM = 2

BLACK_1D = 0
WHITE_1D = 255
BLUE_3D = (255, 0, 0)
WHITE_3D = (255, 255, 255)
BLACK_3D = (0, 0, 0)
RED_3D = (0, 0, 255)
GREEN_3D = (0, 255, 0)
YELLOW_3D = (0, 255, 255)


def main():
    args = interpret_args()

    grid, surface_arr, contour = None, None, None
    terminal_img, gray_img, contours_img = None, None, None

    terminal_img, gray_img = get_input_img(args)

    try:
        grid = grid_data(gray_img, args.calib_area)
        erase_estimate_calibration_area(gray_img, args.calib_area)

        contours_img = connect_nearby_chars(gray_img, args.radius)
        contours_img = smooth_contours(grid, contours_img)
        contour = contour_points(contours_img)

        surface_arr = approximate_surface_slopes(contour, grid)
        export_surface_arr(args.out_file, surface_arr)
    except:
        print('[!] Exception')
        traceback.print_exc()
    finally:
        inspect(grid, args.calib_area, surface_arr, contour, terminal_img,
            gray_img, contours_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def interpret_args():
    parser = argparse.ArgumentParser(
        description='Sloper calculate surface slope of ASCII-art figures.\n'
                    'Last version you can find on github.com/MateuszJanda/sloper',
        usage='Please try to use -h, --help for more informations',
        epilog='Example:\n'
               '# Detect surface of ASCII figure saved in text file\n'
               '$ python sloper.py -a ascii_data/umbrella.txt\n'
               '$ python sloper.py -a ascii_data/cat-drilled.txt -r 17 -o ascii_data/cat-drilled.surf\n'
               '\n'
               '# Detect surface of ASCII figure saved in image\n'
               '$ python sloper.py -i ascii_data/umbrella.png',
        formatter_class=CustomFormatter)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--image', metavar='file', dest='img_file',
        help='ASCII figure in image (with proper markers)')
    group.add_argument('-a', '--ascii', metavar='file', dest='ascii_file',
        help='ASCII figure in text file (with proper markers)')


    parser.add_argument('-o', '--output-file', metavar='file', required=False,
        default='output.surf', dest='out_file',
        help='Output file for surface array')
    parser.add_argument('-f', '--truetype-font', metavar='file', required=False,
        default='UbuntuMono-R', dest='font',
        help='TryType font file')
    parser.add_argument('-s', '--font-size', metavar='size', required=False,
        default=17, dest='font_size',
        help='TryType font size')
    parser.add_argument('-c', '--calibration-size', metavar='size', required=False,
        dest='calib_area',
        help='Calibration area size')
    parser.add_argument('-r', '--radius', metavar='radius', required=False,
        default=15, dest='radius',
        help='Radius to nearest neighbor')
    parser.add_argument('-t', '--threshold', metavar='value', required=False,
        default=30, dest='threshold',
        help='Threshold value')

    args = parser.parse_args()

    args.font_size = int(args.font_size)
    args.radius = int(args.radius)
    args.threshold = int(args.threshold)
    estimate_calibration_area(args)

    print('[+] Font size:', args.font_size)
    print('[+] Radius:', args.radius)
    print('[+] Threshold:', args.threshold)

    return args


def estimate_calibration_area(args):
    CALIBRATION_AREA_SIZE = 60
    if args.img_file and not args.calib_area:
        args.calib_area = CALIBRATION_AREA_SIZE
    elif args.ascii_file and not args.calib_area:
        args.calib_area = (args.font_size + 2) * 3

    print('[+] Calibration area:', args.calib_area)


def get_input_img(args):
    if args.img_file:
        terminal_img = cv2.imread(args.img_file, cv2.IMREAD_COLOR)
    else:
        with open(args.ascii_file, 'r') as f:
            text = f.read()

        wh_size = estimate_ascii_img_size(text, args)
        pil_img = Image.new('RGB', color=BLACK_3D, size=wh_size)
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(args.font, size=args.font_size)

        EXTRA_MARGINS = (2, 2)
        draw.text(xy=EXTRA_MARGINS, text=text, font=font, fill=WHITE_3D)
        terminal_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    gray_img = cv2.cvtColor(terminal_img, cv2.COLOR_RGB2GRAY)
    _, gray_img = cv2.threshold(src=gray_img, thresh=args.threshold,
        maxval=255, type=cv2.THRESH_BINARY)

    return terminal_img, gray_img


def estimate_ascii_img_size(text, args):
    EXTRA = 2
    grid = grid_ascii_data(text)
    cell_size = Size(height=args.font_size+EXTRA, width=(args.font_size+EXTRA)//2)
    width = (grid.end.x + EXTRA) * cell_size.width
    height = (grid.end.y + EXTRA) * cell_size.height

    img_size = Size(height, width)

    print('[+] Estimated cell size:', cell_size)
    print('[+] Estimated image size:', img_size)

    return (img_size.width, img_size.height)


def grid_ascii_data(text):
    """
    Calculate grid data for text/ASCII.
    Return:
        top-left point where first cell (grid) start
        bottom-right point where last cell (grid) end
        cell size (width, height), always (1, 1)
    """
    text_width = 0
    for line in text.split('\n'):
        text_width = max(text_width, len(line))

    text_arr = []
    for line in text.split('\n'):
        l = [ch for ch in line]
        l += [ch for ch in (text_width - len(line)) * ' ']
        text_arr.append(l)

    text_arr = np.array(text_arr)
    print('[+] Text array size:', Size(*text_arr.shape))

    text_arr = remove_ascii_margins(text_arr)
    start_pt = marker_ascii_pos(text_arr)

    text_size = Size(*text_arr.shape)
    end_pt = Point(x=start_pt.x + text_size.width,
                   y=start_pt.y + text_size.height)
    grid = Grid(start=start_pt, end=end_pt, cell=Size(1, 1))
    print('[+] Text grid top-left pos:', grid.start)
    print('[+] Text grid bottom-right pos:', grid.end)

    print('[+] ASCII array (without margins) size:', text_size)
    for line in text_arr:
        print(''.join(line))

    return grid


def remove_ascii_margins(ascii_arr):
    """Remove margins from ASCII array."""
    del_rows = [r for r, margin in enumerate(np.all(ascii_arr==' ', axis=0)) if margin]
    ascii_arr = np.delete(ascii_arr, del_rows, axis=1)

    del_columns = [c for c, margin in enumerate(np.all(ascii_arr==' ', axis=1)) if margin]
    ascii_arr = np.delete(ascii_arr, del_columns, axis=0)

    return ascii_arr


def marker_ascii_pos(text_arr):
    """
    Find marker position in ASCII array. Markers must by placed in first
    3x3 sub array.
    """
    DIM = 3
    for x, y in it.product(range(DIM), range(DIM)):
        if y+1 > text_arr.shape[0] or x+1 > text_arr.shape[1]:
            continue
        if text_arr[y, x] == '_' and text_arr[y+1, x] == '^' and text_arr[y, x+1] == '^':
            return Point(x, y)

    return None


def grid_data(img, calib_area):
    """
    Calculate grid data.
    Return:
        top-left point where first cell (grid) start
        bottom-right point where last cell (grid) end
        cell size (width, height)
    """
    under_tl_pt, under_br_pt = underscore_pos(img, calib_area)
    roof_pt = roof_pos(img, calib_area, under_tl_pt, under_br_pt)
    sep_height = separator_height(img, calib_area, under_tl_pt, under_br_pt)

    width = under_br_pt.x - under_tl_pt.x + 1
    height = under_br_pt.y - roof_pt.y + sep_height
    cell_size = Size(height, width)

    start_pt = Point(under_tl_pt.x, under_br_pt.y - height + 1)
    end_pt = Point(start_pt.x + ((img.shape[1] - start_pt.x) // cell_size.width) * cell_size.width,
                   start_pt.y + ((img.shape[0] - start_pt.y) // cell_size.height) * cell_size.height)

    grid = Grid(start_pt, end_pt, cell_size)
    print('[+] Image grid top-left pos:', grid.start)
    print('[+] Image grid bottom-right pos:', grid.end)
    print('[+] Image cell size:', grid.cell)
    return grid


def underscore_pos(img, calib_area):
    """
    Return underscore ('_') position (top-left point, bottom-right point).
    """
    under_tl_pt = None
    for x, y in it.product(range(calib_area), range(calib_area)):
        if img[y, x] != BLACK_1D:
            under_tl_pt = Point(x, y)
            break

    tmp = None
    for x in range(under_tl_pt.x, calib_area):
        if img[under_tl_pt.y, x] == BLACK_1D:
            break
        tmp = Point(x, under_tl_pt.y)

    under_br_pt = None
    for y in range(tmp.y, calib_area):
        if img[y, tmp.x] == BLACK_1D:
            break
        under_br_pt = Point(tmp.x, y)

    print('[+] Underscore top-left:', under_tl_pt)
    print('[+] Underscore bottom-right:', under_br_pt)
    return under_tl_pt, under_br_pt


def roof_pos(img, calib_area, under_tl_pt, under_br_pt):
    """Return roof sign '^' position (the pick point)."""
    roof_pt = Point(0, calib_area)
    width = under_br_pt.x - under_tl_pt.x + 1

    for x in range(under_br_pt.x + 1, under_br_pt.x + width):
        for y in range(calib_area):
            if img[y, x] != BLACK_1D and y < roof_pt.y:
                roof_pt = Point(x, y)

    print('[+] Roof pos:', roof_pt)
    return roof_pt


def separator_height(img, calib_area, under_tl_pt, under_br_pt):
    """
    Return separator height between underscore '_' and bottom roof sign '^'.
    """
    roof_pt = Point(0, calib_area)
    width = under_br_pt.x - under_tl_pt.x + 1

    for x in range(under_tl_pt.x, under_tl_pt.x + width):
        for y in range(under_br_pt.y + 1, calib_area):
            if img[y, x] != BLACK_1D and y < roof_pt.y:
                roof_pt = Point(x, y)

    height = roof_pt.y - under_br_pt.y

    print('[+] Separator height:', height)
    return height


def erase_estimate_calibration_area(img, calib_area):
    """Erase calibration are from image (fill are with black)."""
    cv2.rectangle(img, (0, 0), (calib_area, calib_area), BLACK_1D, cv2.FILLED)


def connect_nearby_chars(img, radius=15):
    """
    Connect nearby contours (ASCII characters).

    References:
    https://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them
    http://answers.opencv.org/question/169492/accessing-all-points-of-a-contour/
    https://docs.opencv.org/3.4.1/dd/d49/tutorial_py_contour_features.html
    """
    gray_img = np.copy(img)
    contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # First contour position is at the bottom of image
    print('[+] Contours count:', len(contours))
    last = contours.pop(0)
    chain = [last]
    while len(contours) > 0:
        cnt = find_nearest_contour(last, contours, radius)

        if cnt is None:
            print('[i] Contours left:', len(contours))
            approx = cv2.approxPolyDP(np.vstack(chain), epsilon=2, closed=True)
            debug_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(debug_img, [approx], -1, GREEN_3D, 1)
            cv2.imshow('Debug', debug_img)
            raise Exception('Error! Contour not found. Try to change radius or font size.')

        chain.append(cnt)
        for i in range(len(contours)):
            if np.all(cnt == contours[i]):
                contours.pop(i)
                break
        last = cnt

    contours_img = np.zeros_like(gray_img)

    approx = cv2.approxPolyDP(np.vstack(chain), epsilon=2, closed=True)
    cv2.drawContours(contours_img, [approx], -1, WHITE_1D, 1)

    return contours_img


def smooth_contours(grid, img):
    """Apply dilation and erosion to smooth contours shape."""
    kernel_dil = np.ones((3, 3), np.uint8)
    kernel_ero = np.ones((2, 2), np.uint8)

    dilation_img = cv2.dilate(img, kernel_dil, iterations=1)
    erosion_img = cv2.erode(dilation_img, kernel_ero, iterations=1)

    # Trim image borders
    cv2.line(erosion_img, (grid.end.x, 0), (grid.end.x, grid.end.y), BLACK_1D, 1)
    cv2.line(erosion_img, (0, grid.end.y), (grid.end.x, grid.end.y), BLACK_1D, 1)

    return erosion_img


def find_nearest_contour(head_cnt, contours, radius=15):
    """Find nearest contour to current head contour."""
    best_cnt = None
    for cnt in contours:
        for head_pos, cnt_pos in it.product(head_cnt, cnt):
            dist = np.linalg.norm(head_pos-cnt_pos)

            if abs(dist) < radius:
                radius = abs(dist)
                best_cnt = cnt

    return best_cnt


def contour_points(img):
    """Get list of start/end point off all contours."""
    contours_img = np.copy(img)
    contours, _ = cv2.findContours(contours_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [Point(c[0, 0], c[0, 1]) for c in np.vstack(contours)]


def approximate_surface_slopes(contour, grid):
    """
    Go along ASCII image surface and calculate normal vector (perpendicular to
    surface) for each position where braille dot, could be placed.
    """
    height = ((grid.end.y - grid.start.y)//grid.cell.height) * SCR_CELL_SIZE.height
    width = ((grid.end.x - grid.start.x)//grid.cell.width) * SCR_CELL_SIZE.width
    surface_arr = np.zeros(shape=[height, width, NORM_VEC_DIM], dtype=np.float32)

    first_pt, last_pt = None, None
    border_pt = Point(0, 0)
    normal_vec = np.array([0, 0])
    for c in contour:
        if not first_pt:
            first_pt = c
            continue

        center_pt, tl_pt, br_pt = dot_field(first_pt, grid)
        if in_boundaries(c, tl_pt, br_pt):
            last_pt = c
        elif last_pt:
            normal_vec = normal_unit_vec(first_pt, last_pt)
            surface_arr[center_pt.y, center_pt.x] = normal_vec
            first_pt = c
            last_pt = None
            border_pt = border_point(center_pt, border_pt)
        else:
            surface_arr[center_pt.y, center_pt.x] = normal_vec
            first_pt = c
            last_pt = None
            border_pt = border_point(center_pt, border_pt)

    surface_arr = remove_margins(surface_arr, border_pt)
    print('[+] Surface array shape:', surface_arr.shape)
    print('[+] Surface array size:', Size(*surface_arr.shape[:2]))

    return surface_arr


def in_boundaries(test_pt, tl_pt, br_pt):
    """Check if point is in boundaries."""
    return tl_pt.x <= test_pt.x < br_pt.x and tl_pt.y <= test_pt.y < br_pt.y


def normal_unit_vec(pt1, pt2):
    """
    Calculate normal unit vector perpendicular to vector defined by two points.
    """
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
    # Values as stored in numpy array where by convention dimensions should
    # start from Y followed by X [Y, X]
    mag = math.sqrt(a**2 + b**2)
    if pt2.x <= pt1.x:
        return np.array([b/mag, -a/mag])
    return np.array([-b/mag, a/mag])


def dot_field(pt, grid):
    """
    Calculate center and boundary points of field where braille dot could be
    placed.
    """
    width = grid.cell.width / SCR_CELL_SIZE.width
    height = grid.cell.height / SCR_CELL_SIZE.height

    center_pt = Point(int((pt.x - grid.start.x)/width),
                int((pt.y - grid.start.y)/height))
    x = grid.start.x + center_pt.x * width
    y = grid.start.y + center_pt.y * height

    if int(x + width) <= pt.x:
        center_pt = Point(center_pt.x + 1, center_pt.y)
    if int(y + height) <= pt.y:
        center_pt = Point(center_pt.x, center_pt.y + 1)

    tl_pt = Point(int(grid.start.x + center_pt.x * width),
                  int(grid.start.y + center_pt.y * height))
    br_pt = Point(int(grid.start.x + (center_pt.x + 1) * width),
                  int(grid.start.y + (center_pt.y + 1) * height))

    return center_pt, tl_pt, br_pt


def border_point(current_pt, old_pt):
    """Choose most bottom-right point."""
    x = current_pt.x if current_pt.x > old_pt.x else old_pt.x
    y = current_pt.y if current_pt.y > old_pt.y else old_pt.y

    return Point(x, y)


def remove_margins(surface_arr, border_pt):
    """Remove surface margins."""
    if border_pt.y % SCR_CELL_SIZE.height:
        height = (border_pt.y // SCR_CELL_SIZE.height + 1) * SCR_CELL_SIZE.height
    else:
        height = (border_pt.y // SCR_CELL_SIZE.height) * SCR_CELL_SIZE.height

    if border_pt.x % SCR_CELL_SIZE.width:
        width = (border_pt.x // SCR_CELL_SIZE.width + 1) * SCR_CELL_SIZE.width
    else:
        width = (border_pt.x // SCR_CELL_SIZE.width) * SCR_CELL_SIZE.width

    del_rows = [r for r in range(height, surface_arr.shape[0])]
    surface_arr = np.delete(surface_arr, del_rows, axis=0)

    del_columns = [c for c in range(width, surface_arr.shape[1])]
    surface_arr = np.delete(surface_arr, del_columns, axis=1)

    return surface_arr


def export_surface_arr(file_name, arr):
    """Export braille data to file."""
    height, width, vec_dim = arr.shape
    np.savetxt(file_name, arr.reshape([height, width*vec_dim]), fmt='%.04f')


def inspect(grid, calib_area, surface_arr, contour, terminal_img, gray_img, contours_img):
    """Inspect calculated data."""
    grid_img = np.copy(terminal_img)

    if terminal_img is not None:
        draw_estimate_calibration_area(terminal_img, calib_area)

    # Draw grid and markers cells.
    if not (terminal_img is None or surface_arr is None or grid is None):
        draw_cell(grid_img, grid.start, grid)
        draw_grid(grid_img, grid)
        draw_surface_arr_shape(grid_img, grid, surface_arr)
        cv2.imshow('Grid and markers', grid_img)

    if contours_img is not None:
        cv2.imshow('Contours', contours_img)

    # Braille dots in place where normal vector was calculated
    if not (gray_img is None or surface_arr is None or grid is None):
        dots_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        draw_braille_dots(dots_img, surface_arr, grid)
        cv2.imshow('Braille dots', dots_img)

    # Normal vectors perpendicular to the surface
    if not (gray_img is None or surface_arr is None or grid is None or contour is None):
        surface_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        draw_normal_vecs(surface_img, surface_arr, grid)
        draw_contour(surface_img, contour)
        cv2.imshow('Normal vectors', surface_img)


def draw_estimate_calibration_area(img, calib_area):
    cv2.line(img, (0, calib_area), (calib_area, calib_area), YELLOW_3D, 1)
    cv2.line(img, (calib_area, 0), (calib_area, calib_area), YELLOW_3D, 1)
    cv2.imshow('ASCII image', img)


def draw_cell(img, pt, grid, xor_value=158):
    """
    Mark screen cell with some color (xor with original color value).
    """
    val = np.array([xor_value, xor_value, xor_value]).astype(img.dtype)
    for x in range(pt.x, pt.x + grid.cell.width):
        for y in range(pt.y, pt.y + grid.cell.height):
            img[y, x] ^= val


def draw_grid(img, grid):
    """
    Draw grid that separate cells.
    """
    for x in range(grid.start.x, grid.end.x + 1, grid.cell.width):
        cv2.line(img, (x, grid.start.y), (x, grid.end.y), BLUE_3D, 1)

    for y in range(grid.start.y, grid.end.y + 1, grid.cell.height):
        cv2.line(img, (grid.start.x, y), (grid.end.x, y), BLUE_3D, 1)


def draw_surface_arr_shape(img, grid, surface_arr):
    """
    Draw surface_arr shape on grid.
    """
    end_pt = Point(grid.start.x + grid.cell.width * surface_arr.shape[1] // SCR_CELL_SIZE.width,
                   grid.start.y + grid.cell.height * surface_arr.shape[0] // SCR_CELL_SIZE.height)

    cv2.line(img, (grid.start.x, grid.start.y), (grid.start.x, end_pt.y), RED_3D, 1)
    cv2.line(img, (grid.start.x, grid.start.y), (end_pt.x, grid.start.y), RED_3D, 1)
    cv2.line(img, (grid.start.x, end_pt.y), (end_pt.x, end_pt.y), RED_3D, 1)
    cv2.line(img, (end_pt.x, grid.start.y), (end_pt.x, end_pt.y), RED_3D, 1)


def draw_braille_dots(img, arr, grid):
    """
    Draw dot for each "not zero" element of array.
    """
    foreach_arr_elements(img, arr, grid, draw_dot)


def draw_normal_vecs(img, arr, grid):
    """
    Draw perpendicular vector to the surface, where surface is every "non zero"
    of array (array with normal vectors). Normal vectors are multiply by some
    VEC_FACTOR, to be better visible.
    """
    foreach_arr_elements(img, arr, grid, draw_normal_vec)


def foreach_arr_elements(img, arr, grid, draw_func):
    """
    Call draw_func() for each "non zero" array element.
    """
    end_x = grid.start.x + (arr.shape[1] / SCR_CELL_SIZE.width) * grid.cell.width
    end_y = grid.start.y + (arr.shape[0] / SCR_CELL_SIZE.height) * grid.cell.height
    samples_x = ((end_x - grid.start.x)/grid.cell.width) * SCR_CELL_SIZE.width
    samples_y = ((end_y - grid.start.y)/grid.cell.height) * SCR_CELL_SIZE.height

    for bx, x in enumerate(np.linspace(grid.start.x, end_x, samples_x, endpoint=False)):
        for by, y in enumerate(np.linspace(grid.start.y, end_y, samples_y, endpoint=False)):
            if np.any(arr[by, bx]):
                draw_func(img, field_pt=Point(x, y), normal_vec=arr[by, bx], grid=grid)


def draw_dot(img, field_pt, normal_vec, grid):
    """
    Draw dot (braille dot) at given point.
    """
    dot_field_size = Size(grid.cell.height/SCR_CELL_SIZE.height,
                          grid.cell.width/SCR_CELL_SIZE.width)
    center = Point(int(field_pt.x + dot_field_size.width//2),
                   int(field_pt.y + dot_field_size.height//2))
    cv2.circle(img, center, radius=2, color=RED_3D, thickness=-1)


def draw_normal_vec(img, field_pt, normal_vec, grid):
    """
    Draw vector (normal_vec) at given point. Basically normal_vec will be
    multiplied by VEC_FACTOR to be better visible.
    """
    dot_field_size = Size(grid.cell.height/SCR_CELL_SIZE.height,
                          grid.cell.width/SCR_CELL_SIZE.width)

    start = Point(int(field_pt.x + dot_field_size.width//2),
                  int(field_pt.y + dot_field_size.height//2))
    # Y with minus, because OpenCV use different coordinates order
    vec_end = Point(normal_vec[1], -normal_vec[0])
    end = Point(start.x + int(vec_end.x*VEC_FACTOR),
                start.y + int(vec_end.y*VEC_FACTOR))
    cv2.line(img, start, end, GREEN_3D, 1)


def draw_contour(img, contour):
    """
    Connect all contours point with lines.
    """
    for c in contour:
        img[c.y, c.x] = YELLOW_3D


if __name__ == '__main__':
    main()
