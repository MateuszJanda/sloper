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
import math
import time
import curses
import locale
import pdb
import numpy as np


Size = co.namedtuple('Size', ['width', 'height'])


EMPTY_BRAILLE = u'\u2800'
BUF_CELL_SIZE = Size(2, 4)
VECTOR_DIM = 2

GRAVITY_ACC = 9.8  # [m/s^2]
COEFFICIENT_OF_RESTITUTION = 0.5
COEFFICIENT_OF_FRICTION = 0.9


def main(scr):
    setup_curses(scr)
    screen = Screen(scr)
    terrain = Terrain()

    im = Importer()
    ascii_arr, norm_arr = im.load('ascii_fig.txt', 'ascii_fig.png.norm')

    terrain.add_arr(norm_arr)
    screen.add_ascii(ascii_arr)
    # screen.add_norm_arr(norm_arr)


    # for y in range(terrain._terrain_size.height):
    #     if np.any(terrain._terrain[y, 50]):
    #         eprint('YYY', y)
    #         eprint(arrpos_to_ptpos(50, y, terrain._terrain_size))

    #         exit()

    assert(np.all(Vector(50, 38) == Vector(50, 38)))
    arrpos = ptpos_to_arrpos(Vector(50,38))
    assert(np.all(Vector(50, 38) == arrpos_to_ptpos(arrpos.x, arrpos.y)))
    # eprint('line', curses.LINES )

    bodies = [
        # Body(ptpos=Vector(30, 80), mass=10, velocity=Vector(0, -40))
        Body(ptpos=Vector(50, 80), mass=10, velocity=Vector(0, -40)),  # TODO: check interaction with normal vector
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

        TODO: shift should be buf_shift
        """
        height, width, _ = arr.shape
        for x, y in it.product(range(width), range(height)):
            if np.any(arr[y, x] != 0):
                ptpos = arrpos_to_ptpos(x, self._arr_size.height - height + y) + shift
                self.draw_point(ptpos)

        self._save_in_backup_buf()

    def add_ascii(self, ascii_arr, shift=Vector(0, 0)):
        height, width = ascii_arr.shape
        for x, y in it.product(range(width), range(height)):
            if np.any(ascii_arr[y, x] != ' '):
                buffpos = Vector(x, self._buf_size.height - height + y)
                self._buf[buffpos.y][buffpos.x] = ascii_arr[y, x]

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
                return ord(EMPTY_BRAILLE) | (0x20 >> (by - 1))

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
        self.prev_ptpos = ptpos
        self.mass = mass
        self.vel = velocity
        self.lock = False

    def is_moving(self):
        return not math.isclose(self.vel.magnitude(), 0, abs_tol=0.01)


class Terrain:
    def __init__(self):
        self._terrain_size = Size((curses.COLS-1)*BUF_CELL_SIZE.width, curses.LINES*BUF_CELL_SIZE.height)
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

    def get_normal_vec(self, pt):
        # pt = np.floor(pt)
        # eprint('checking', pt)
        arrpos = ptpos_to_arrpos(pt)

        # if int(pt.y) == 38:
        #     eprint('col', arrpos.y)
        #     time.sleep(100)

        # eprint('arrpos', arrpos)
        normal_vec = self._terrain[arrpos.y, arrpos.x]
        if np.any(normal_vec != 0):
            # eprint('at point', pt, 'norm vec', Vector(normal_vec[0], normal_vec[1]))
            return Vector(normal_vec[0], normal_vec[1])

        return None

    def in_border(self, pt):
        arrpos = ptpos_to_arrpos(pt)
        return 0 <= arrpos.x < self._terrain_size.width and \
               0 <= arrpos.y < self._terrain_size.height


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
        """Remove margin from ascii_arr (line and columns with spaces at the edges"""
        del_rows = [idx for idx, margin in enumerate(np.all(ascii_arr == ' ', axis=0)) if margin]
        ascii_arr = np.delete(ascii_arr, del_rows, axis=1)

        del_columns = [idx for idx, margin in enumerate(np.all(ascii_arr == ' ', axis=1)) if margin]
        ascii_arr = np.delete(ascii_arr, del_columns, axis=0)

        return ascii_arr

    def _import_norm_arr(self, norm_file):
        """Import array with normal vector"""
        arr = np.loadtxt(norm_file)
        height, width = arr.shape
        norm_arr = arr.reshape(height, width//VECTOR_DIM, VECTOR_DIM)

        return norm_arr

    def _remove_norm_margin(self, norm_arr):
        """Remove margin from array with normal vectors (line and columns with
        np.array([0, 0]) at the edges"""
        if norm_arr.shape[1] % BUF_CELL_SIZE.width or norm_arr.shape[0] % BUF_CELL_SIZE.height:
            raise Exception("Arrays with normal vector can't be transformed to buffer")

        ascii_markers = self._transform_norm(norm_arr)
        del_rows = [list(range(idx*BUF_CELL_SIZE.height, idx*BUF_CELL_SIZE.height+BUF_CELL_SIZE.height))
                    for idx, margin in enumerate(np.all(ascii_markers == False, axis=1)) if margin]
        norm_arr = np.delete(norm_arr, del_rows, axis=0)

        del_columns = [list(range(idx*BUF_CELL_SIZE.width, idx*BUF_CELL_SIZE.width+BUF_CELL_SIZE.width))
                       for idx, margin in enumerate(np.all(ascii_markers == False, axis=0)) if margin]
        norm_arr = np.delete(norm_arr, del_columns, axis=1)

        return norm_arr

    def _transform_norm(self, norm_arr):
        """Transform array with normal vectors (for braille characters), to
        dimmensions of ascii figure, and mark if in ascii cell there was any
        character"""
        EMPTY = np.array([0, 0])
        norm_reduce = np.logical_or.reduce(norm_arr != EMPTY, axis=-1)

        result = []
        for y in range(0, norm_reduce.shape[0], BUF_CELL_SIZE.height):
            for x in range(0, norm_reduce.shape[1], BUF_CELL_SIZE.width):
                result.append(int(np.any(norm_reduce[y:y+BUF_CELL_SIZE.height, x:x+BUF_CELL_SIZE.width])))

        size = Size(norm_reduce.shape[1]//BUF_CELL_SIZE.width, norm_reduce.shape[0]//BUF_CELL_SIZE.height)
        result = np.reshape(result, (size.height, size.width))

        eprint(result)
        return result

    def _validate_arrays(self, ascii_arr, norm_arr):
        """Validate if both arrays describe same thing"""
        ascii_arr_size = Size(ascii_arr.shape[1], ascii_arr.shape[0])
        norm_arr_size = Size(norm_arr.shape[1]//BUF_CELL_SIZE.width, norm_arr.shape[0]//BUF_CELL_SIZE.height)

        if ascii_arr_size != norm_arr_size:
            raise Exception('Imported arrays (ascii/norm) - mismatch size', ascii_arr_size, norm_arr_size)

        eprint('Validation OK')


def ptpos_to_bufpos(pt):
    x = int(pt.x/BUF_CELL_SIZE.width)
    y = curses.LINES - 1 - int(pt.y/BUF_CELL_SIZE.height)
    return Vector(x, y)


def arrpos_to_ptpos(x, y):
    """Array position to Cartesian coordinate system"""
    y = curses.LINES * BUF_CELL_SIZE.height - y
    return Vector(x, y)


def ptpos_to_arrpos(pt):
    y = curses.LINES * BUF_CELL_SIZE.height - pt.y
    return Vector(int(pt.x), int(y))


def step_simulation(dt, bodies, terrain):
    calc_forces(dt, bodies)
    integrate(dt, bodies)
    collisions = detect_collisions(bodies, terrain)
    resolve_collisions(dt, collisions)
    fix_penetration(bodies, terrain.size())


def calc_forces(dt, bodies):
    for body in bodies:
        body.forces = Vector(0, -GRAVITY_ACC) * body.mass

        if int(body.prev_ptpos.y) == 0 and int(body.ptpos.y) == 0 and int(body.prev_ptpos.x) != int(body.ptpos.x):
            body.forces *= COEFFICIENT_OF_FRICTION


def integrate(dt, bodies):
    for b in bodies:
        if b.lock:
            eprint('LOCK')
            continue

        # b.acc = Vector(0, -GRAVITY_ACC) + b.forces/b.mass
        b.acc = b.forces / b.mass
        b.vel = b.vel + b.acc * dt
        b.prev_ptpos = b.ptpos
        b.ptpos = b.ptpos + b.vel * dt

        # eprint('Integrate current', b.ptpos, 'prev', b.prev_ptpos,)
        # eprint(b.ptpos)
        # if int(b.ptpos.y) == 38:
        #     time.sleep(5000)
        #     eprint('col')

        # Don't calculate collision if body is not moving
        # if math.isclose(b.vel.magnitude(), 0, abs_tol=0.01):
        if not b.is_moving():
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
        collisions += obstacle_collisions(body, terrain)

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


def obstacle_collisions(body, terrain):
    norml_vec = obstacle_pos(terrain, body.prev_ptpos, body.ptpos)
    # eprint(body.prev_ptpos)
    # eprint(body.ptpos)
    # eprint('----')
    # exit()
    if np.any(norml_vec):
        return [Collision(body1=body,
                          body2=None,
                          relative_vel=-body.vel,
                          collision_normal=norml_vec)]
    return []


def obstacle_pos(terrain, pt1, pt2):
    """Bresenham's line algorithm
    https://pl.wikipedia.org/wiki/Algorytm_Bresenhama
    """
    # x, y = pt1.x, pt1.y
    # eprint('start')
    check_pt = np.floor(pt1)
    # eprint(check_pt)
    pt1 = np.floor(pt1)
    pt2 = np.floor(pt2)
    # return None

    # Drawing direction
    if pt1.x < pt2.x:
        xi = 1
        dx = pt2.x - pt1.x
    else:
        xi = -1
        dx = pt1.x - pt2.x

    if pt1.y < pt2.y:
        yi = 1
        dy = pt2.y - pt1.y
    else:
        yi = -1
        dy = pt1.y - pt2.y

    if not terrain.in_border(check_pt):
        return None

    # normal_vec = terrain.get_normal_vec(check_pt)
    # if np.any(normal_vec):
    #     return normal_vec

    # X axis
    if dx > dy:
        ai = (dy - dx) * 2
        bi = dy * 2
        d = bi - dx
        while check_pt.x != pt2.x:
            # coordinate test
            if d >= 0:
                check_pt.x += xi
                check_pt.y += yi
                d += ai
            else:
                d += bi
                check_pt.x += xi

            if not terrain.in_border(check_pt):
                return None

            normal_vec = terrain.get_normal_vec(check_pt)
            if np.any(normal_vec):
                return normal_vec
    else:
        ai = (dx - dy) * 2
        bi = dx * 2
        d = bi - dy
        while check_pt.y != pt2.y:
            # coordinate test
            if d >= 0:
                check_pt.x += xi
                check_pt.y += yi
                d += ai
            else:
                d += bi
                check_pt.y += yi

            if not terrain.in_border(check_pt):
                return None

            normal_vec = terrain.get_normal_vec(check_pt)
            if np.any(normal_vec):
                return normal_vec

    return None


def resolve_collisions(dt, collisions):
    for c in collisions:
        # Collision with screen border
        if not c.body2:
            impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * np.dot(c.relative_vel, c.collision_normal)) / \
                    (1/c.body1.mass)

            # eprint('pos 1', c.body1.ptpos)
            # eprint('prev 1', c.body1.prev_ptpos)
            c.body1.vel -= (c.collision_normal / c.body1.mass) * impulse
            c.body1.ptpos += c.body1.vel * dt

            # eprint('vel', c.body1.vel)
            # eprint('normal', c.collision_normal)
            # eprint('pos 2', c.body1.ptpos)
            # time.sleep(100)


def fix_penetration(bodies, terrain_size):
    for body in bodies:
        body.ptpos.x = max(0, min(body.ptpos.x, terrain_size.width))
        body.ptpos.y = max(0, min(body.ptpos.y, terrain_size.height))


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    setup_stderr()
    curses.wrapper(main)
