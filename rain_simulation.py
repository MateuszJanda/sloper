#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coordinates systems:
    ptpos      - position in Cartesian coordinate system
    buffpos    - position on screen (of one character). Y from top to bottom
    arrpos     - similar to ptpos, but Y from top to bottom
"""


import sys
import collections as co
import itertools as it
import copy
import numpy as np
import math
import curses
import locale
import time
import pdb


Size = co.namedtuple('Size', ['width', 'height'])


EMPTY_BRAILLE = u'\u2800'
BUF_CELL_SIZE = Size(2, 4)
VECTOR_DIM = 2

GRAVITY_ACC = 9.8  # [m/s^2]
COEFFICIENT_OF_RESTITUTION = 0.5



def main(scr):
    setup_curses(scr)
    screen = Screen(scr)
    terrain = Terrain()

    im = Importer()
    txt, arr = im.load('ascii_fig.txt', 'ascii_fig.png.norm')

    terrain.add_arr(arr)
    screen.add_ascii(txt)
    # screen.add_norm_arr(arr)

    bodies = [
        Body(ptpos=Vector(50, 80), mass=10, velocity=Vector(0, -40)),
        # Body(ptpos=Vector(95, 80), mass=1, velocity=Vector(0, -40)),
        # Body(ptpos=Vector(110, 80), mass=1, velocity=Vector(0, -40)),
        # Body(ptpos=Vector(20, 80), mass=1, velocity=Vector(0, -40)),
    ]

    for b in bodies:
        b.forces = Vector(0, 0)

    t = 0
    freq = 100
    dt = 1/freq

    while True:
        screen.restore_backup()

        step_simulation(dt, bodies, terrain)

        for b in bodies:
            # screen.draw_hailstone(b.ptpos)
            screen.draw_point(b.ptpos)
        screen.refresh()

        time.sleep(dt)
        t += dt

    curses.endwin()


def setup_stderr():
    """Redirect stderr to other terminal. Run tty command, to get terminal id."""
    sys.stderr = open('/dev/pts/3', 'w')


def eprint(*args, **kwargs):
    """Print on stderr"""
    print(*args, file=sys.stderr)


def eassert(condition):
    """Assert. Disable curses and run pdb"""
    if not condition:
        curses.endwin()
        sys.stderr = sys.stdout
        pdb.set_trace()


def setup_curses(scr):
    """Setup curses screen"""
    curses.start_color()
    curses.use_default_colors()
    curses.halfdelay(5)
    curses.noecho()
    curses.curs_set(False)
    scr.clear()


class Vector(np.ndarray):
    def __new__(cls, x, y):
        obj = np.asarray([x, y]).view(cls)
        return obj

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    def magnitude(self):
        """Calculate vector magnitude"""
        return np.linalg.norm(self)

    def __str__(self):
        """string representation of object"""
        return "Vector(x=" + str(self.x) + ", y=" + str(self.y) + ")"

    def __repr__(self):
        """string representation of object"""
        return "Vector(x=" + str(self.x) + ", y=" + str(self.y) + ")"


class Screen:
    def __init__(self, scr):
        self._scr = scr

        self._buf_size = Size(curses.COLS-1, curses.LINES)
        self._arr_size = Size(self._buf_size.width*BUF_CELL_SIZE.width,
            self._buf_size.height*BUF_CELL_SIZE.height)

        self._buf = self._get_empty_buf()
        self._buf_backup = copy.deepcopy(self._buf)

    def _get_empty_buf(self):
        return [list(EMPTY_BRAILLE * self._buf_size.width) for _ in range(self._buf_size.height)]


    def add_norm_arr(self, arr, shift=Vector(0, 0)):
        """
        Add static element to screen buffer. Every element in array will be
        represent as braille character. By default all arrays are drawn in
        bottom left corner.
        """
        height, width, _ = arr.shape
        for x, y in it.product(range(width), range(height)):
            if np.any(arr[y, x] != 0):
                pt = arrpos_to_ptpos(x, y, Size(width, height)) + shift
                self.draw_point(pt)

        self._save_in_backup_buf()

    def add_ascii(self, ascii_arr, shift=Vector(0, 0)):
        height, width = ascii_arr.shape
        for x, y in it.product(range(width), range(height)):
            if np.any(ascii_arr[y, x] != ' '):
                w = x
                h = self._buf_size.height - height + y
                self._buf[h][w] = ascii_arr[y, x]

        self._save_in_backup_buf()

    def add_border(self):
        """For debug, draw screen border in braille characters"""
        for x in range(self._arr_size.width):
            self.draw_point(Vector(x, 0))
            self.draw_point(Vector(x, self._arr_size.height-1))

        for y in range(self._arr_size.height):
            self.draw_point(Vector(0, y))
            self.draw_point(Vector(self._arr_size.width-1, y))

        self._save_in_backup_buf()

    def _save_in_backup_buf(self):
        """Backup screen buffer"""
        self._buf_backup = copy.deepcopy(self._buf)

    def draw_point(self, pt):
        # Out of the screen
        if not (0 <= pt.x < self._arr_size.width and 0 <= pt.y < self._arr_size.height):
            return

        bufpos = ptpos_to_bufpos(pt)
        uchar = ord(self._buf[bufpos.y][bufpos.x])
        self._buf[bufpos.y][bufpos.x] = chr(uchar | self._braille_char(pt))

    def _braille_char(self, pt):
        """Point as braille character in buffer cell"""
        bx = int(pt.x) % BUF_CELL_SIZE.width
        by = int(pt.y) % BUF_CELL_SIZE.height

        if bx == 0:
            if by == 0:
                return ord(EMPTY_BRAILLE) | 0x40
            else:
                return ord(EMPTY_BRAILLE) | (0x04 >> (by - 1))
        else:
            if by == 0:
                return ord(EMPTY_BRAILLE) | 0x80
            else:
                return ord(EMPTY_BRAILLE) | (0x20 >> (by -1))

    def restore_backup(self):
        """Restore static elements added to screen"""
        self._buf = copy.deepcopy(self._buf_backup)

    def refresh(self):
        for num, line in enumerate(self._buf):
            self._scr.addstr(num, 0, ''.join(line))
        self._scr.refresh()


class Body:
    def __init__(self, ptpos, mass, velocity):
        self.ptpos = ptpos
        self.mass = mass
        self.vel = velocity
        self.lock = False


class Terrain:
    def __init__(self):
        self._terrain_size = Size(curses.COLS-1*BUF_CELL_SIZE.width,
            curses.LINES*BUF_CELL_SIZE.height)
        self._terrain = np.zeros(shape=[self._terrain_size.height, self._terrain_size.width, VECTOR_DIM])

    def size(self):
        return self._terrain_size

    def add_arr(self, arr, shift=Vector(0, 0)):
        """By default all arrays are drawn in bottom left corner."""
        arr_size = Size(arr.shape[1], arr.shape[0])

        x1 = shift.x
        x2 = x1 + arr_size.width
        y1 = self._terrain_size.height - arr_size.height - shift.y
        y2 = self._terrain_size.height - shift.y
        self._terrain[y1:y2, x1:x2] = arr


class Importer:
    def load(self, ascii_file, norm_file):
        ascii_arr = self._import_ascii_arr(ascii_file)
        ascii_arr = self._remove_ascii_marker(ascii_arr)
        ascii_arr = self._remove_ascii_margin(ascii_arr)

        norm_arr = self._import_norm_arr(norm_file)
        norm_arr = self._remove_norm_margin(norm_arr)

        self._validate_arrays(ascii_arr, norm_arr)
        return ascii_arr, norm_arr

    def _import_ascii_arr(self, ascii_file):
        """Import ascii figure from file"""
        ascii_fig = []
        with open(ascii_file, 'r') as f:
            for line in f:
                arr = np.array([ch for ch in line if ch != '\n'])
                ascii_fig.append(arr)

        ascii_arr = self._reshape_ascii(ascii_fig)

        return ascii_arr

    def _reshape_ascii(self, ascii_fig):
        """Fill end of each line in ascii_fig with spaces, and convert it to np.array"""
        max_size = 0
        for line in ascii_fig:
            max_size = max(max_size, len(line))

        larr = []
        for line in ascii_fig:
            arr = np.append(line, [ch for ch in (max_size - line.shape[0]) * ' '])
            larr.append(arr)

        ascii_arr = np.array(larr)

        return ascii_arr

    def _remove_ascii_marker(self, ascii_arr):
        """Erase 3x3 marker at the left-top position from ascii"""
        ascii_arr[0:3, 0:3] = np.array([' ' for _ in range(9)]).reshape(3, 3)
        return ascii_arr

    def _remove_ascii_margin(self, ascii_arr):
        del_rows = [idx for idx, margin in enumerate(np.all(ascii_arr == ' ', axis=0)) if margin]
        ascii_arr = np.delete(ascii_arr, del_rows, axis=1)

        del_columns = [idx for idx, margin in enumerate(np.all(ascii_arr == ' ', axis=1)) if margin]
        ascii_arr = np.delete(ascii_arr, del_columns, axis=0)

        return ascii_arr

    def _remove_norm_margin(self, norm_arr):
        if norm_arr.shape[1] % BUF_CELL_SIZE.width or norm_arr.shape[0] % BUF_CELL_SIZE.height:
            raise Exception("Arrays with normal vector can't be transformed to buffer")

        falgs = self.transform_norm(norm_arr)

        # eprint(norm_arr.shape)
        # eprint(np.all(falgs == False, axis=0))
        # eprint(np.all(falgs == False, axis=1))

        # empty = np.array([0, 0])
        # norm_reduce = np.logical_and.reduce(norm_arr == empty, axis=-1)

        del_rows = [list(range(idx*BUF_CELL_SIZE.height, idx*BUF_CELL_SIZE.height+BUF_CELL_SIZE.height))
                    for idx, margin in enumerate(np.all(falgs == False, axis=1)) if margin]
        # eprint(del_rows)
        norm_arr = np.delete(norm_arr, del_rows, axis=0)

        # eprint(norm_arr.shape)
        # del_columns = [idx for idx, margin in enumerate(np.all(norm_reduce, axis=1)) if margin]
        del_columns = [list(range(idx*BUF_CELL_SIZE.width, idx*BUF_CELL_SIZE.width+BUF_CELL_SIZE.width))
                    for idx, margin in enumerate(np.all(falgs == False, axis=0)) if margin]
        # eprint(del_columns)
        norm_arr = np.delete(norm_arr, del_columns, axis=1)

        # eprint(norm_arr.shape)

        return norm_arr


    def _import_norm_arr(self, norm_file):
        """Import array with normal vector"""
        arr = np.loadtxt(norm_file)
        height, width = arr.shape
        norm_arr = arr.reshape(height, width//VECTOR_DIM, VECTOR_DIM)
        norm_arr_size = Size(norm_arr.shape[1], norm_arr.shape[0])

        return norm_arr


    def transform_norm(self, norm_arr):
        empty = np.array([0, 0])
        norm_reduce = np.logical_or.reduce(norm_arr != empty, axis=-1)

        # size = Size(norm_arr.shape[1]//BUF_CELL_SIZE.width, norm_arr.shape[0]//BUF_CELL_SIZE.height)
        # result = np.zeros(size.width*size.height)
        result = []


        for y in range(0, norm_reduce.shape[0], BUF_CELL_SIZE.height):
            for x in range(0, norm_reduce.shape[1], BUF_CELL_SIZE.width):
                result.append(int(np.any(norm_reduce[y:y+BUF_CELL_SIZE.height, x:x+BUF_CELL_SIZE.width])))

        size = Size(norm_reduce.shape[1]//BUF_CELL_SIZE.width, norm_reduce.shape[0]//BUF_CELL_SIZE.height)
        result = np.reshape(result, (size.height, size.width))

        eprint(result)

        return result

    def _validate_arrays(self, ascii_arr, norm_arr):
        ascii_arr_size = Size(ascii_arr.shape[1], ascii_arr.shape[0])
        norm_arr_size = Size(norm_arr.shape[1]//BUF_CELL_SIZE.width, norm_arr.shape[0]//BUF_CELL_SIZE.height)

        if ascii_arr_size != norm_arr_size:
        # if ascii_arr.shape != norm_arr.shape:
            # raise Exception('Arrays imported from file. Mismatch size', ascii_arr.shape, norm_arr.shape)
            raise Exception('Arrays imported from file. Mismatch size', ascii_arr_size, norm_arr_size)

        eprint('Validate ok')


def ptpos_to_bufpos(pt):
    x = int(pt.x/BUF_CELL_SIZE.width)
    y = curses.LINES - 1 - int(pt.y/BUF_CELL_SIZE.height)
    return Vector(x, y)


def arrpos_to_ptpos(x, y, arr_size):
    """Array position to Cartesian coordinate system"""
    y = arr_size.height - y
    return Vector(x, y)


def ptpos_to_arrpos(pt):
    y = (curses.LINES - 1) * BUF_CELL_SIZE.height - pt.y
    return Vector(int(pt.x), int(y))


def step_simulation(dt, bodies, terrain):
    integrate(dt, bodies)
    collisions = detect_collisions(bodies, terrain)
    resolve_collisions(dt, collisions)


def integrate(dt, bodies):
    for b in bodies:
        if b.lock:
            continue

        b.acc = Vector(0, -GRAVITY_ACC) + b.forces/b.mass
        b.vel = b.vel + b.acc * dt
        b.ptpos = b.ptpos + b.vel * dt

        # Don't calculate collision if body is not moving
        if math.isclose(b.vel.magnitude(), 0, abs_tol=0.01):
            b.lock = True


class Collision:
    def __init__(self, body1, body2, relative_vel, collision_normal):
        self.body1 = body1
        self.body2 = body2
        self.relative_vel = relative_vel
        self.collision_normal = collision_normal


def detect_collisions(bodies, terrain):
    collisions = []
    for body in bodies:
        if body.lock:
            continue
        collisions += border_collision(body, terrain.size())

    return collisions


def border_collision(body, terrain_size):
    """Check collisions with border"""
    if body.ptpos.x < 0:
        return [Collision(body1=body,
            body2=None,
            relative_vel=-body.vel,
            collision_normal=Vector(1, 0))]
    elif body.ptpos.x > terrain_size.width:
        return [Collision(body1=body,
            body2=None,
            relative_vel=-body.vel,
            collision_normal=Vector(-1, 0))]
    elif body.ptpos.y < 0:
        return [Collision(body1=body,
            body2=None,
            relative_vel=-body.vel,
            collision_normal=Vector(0, 1))]
    elif body.ptpos.y > terrain_size.height:
        return [Collision(body1=body,
            body2=None,
            relative_vel=-body.vel,
            collision_normal=Vector(0, -1))]

    return []


def resolve_collisions(dt, collisions):
    for c in collisions:
        # Collision with screen border
        if not c.body2:
            impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * np.dot(c.relative_vel, c.collision_normal)) / \
                    (1/c.body1.mass)

            c.body1.vel -= (c.collision_normal / c.body1.mass) * impulse
            c.body1.ptpos += c.body1.vel * dt


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    setup_stderr()
    curses.wrapper(main)
