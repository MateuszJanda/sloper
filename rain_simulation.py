#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coordinates systems:
    point/pos/pt    - position in Cartesian coordinate system
    buffpos         - position on screen (of one character). Y from top to bottom
    arrpos          - similar to point, but Y from top to bottom
"""


import sys
import collections as co
import itertools as it
import copy
import numpy as np
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


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        """ string representation of object """
        return "<" + str(self.x) + ", " + str(self.y) + ">"

    def __imul__(self, scalar):
        """ Multiply by scalar"""
        self.x *= scalar
        self.y *= scalar
        return self

    def __mul__(self, scalar):
        """ Multiply Vector by scalar"""
        return Vector(self.x * scalar, self.y * scalar)

    def __ifloordiv__(self, scalar):
        """ Floor division by scalar """
        self.x //= scalar
        self.y //= scalar
        return self

    def __floordiv__(self, scalar):
        """ Floor division by scalar """
        return Vector(self.x // scalar, self.y // scalar)

    def __itruediv__(self, scalar):
        """ True division Vector by scalar """
        self.x /= scalar
        self.y /= scalar
        return self

    def __truediv__(self, scalar):
        """ True division Vector by scalar """
        return Vector(self.x / scalar, self.y / scalar)

    def __isub__(self, vec):
        """ Subtract Vector """
        self.x -= vec.x
        self.y -= vec.y
        return self

    def __sub__(self, vec):
        """ Subtract two Vectors """
        return Vector(self.x - vec.x, self.y - vec.y)

    def __iadd__(self, vec):
        """ Add Vectors """
        self.x += vec.x
        self.y += vec.y
        return self

    def __add__(self, vec):
        """ Add two Vectors """
        return Vector(self.x + vec.x, self.y + vec.y)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalized(self):
        mag = self.magnitude()
        return Vector(self.x / mag, self.y / mag)


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

    def draw_arr(self, arr, shift=Vector(0, 0)):
        """Draw array. Every element is represent as braille character"""
        height, width, _ = arr.shape
        for x, y in it.product(range(width), range(height)):
            if np.any(arr[y, x] != 0):
                pt = arrpos_to_ptpos(x, y, Size(width, height)) + shift
                self.draw_point(pt)

        self._buf_backup = copy.deepcopy(self._buf)

    def draw_borders(self):
        for x in range(self._arr_size.width):
            self.draw_point(Vector(x, 0))
            self.draw_point(Vector(x, self._arr_size.height-1))

        for y in range(self._arr_size.height):
            self.draw_point(Vector(0, y))
            self.draw_point(Vector(self._arr_size.width-1, y))

        self._buf_backup = copy.deepcopy(self._buf)

    def draw_hail(self, pt):
        self.draw_point(pt)

    def draw_point(self, pt):
        bufpos = ptpos_to_bufpos(pt)

        # Out of the screen
        if not (0 <= pt.x  < self._arr_size.width and 0 <= pt.y < self._arr_size.height):
            return

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
                return ord(EMPTY_BRAILLE) | (0x4 >> (by - 1))
        else:
            if by == 0:
                return ord(EMPTY_BRAILLE) | 0x80
            else:
                return ord(EMPTY_BRAILLE) | (0x20 >> (by -1))

    def restore_backup(self):
        self._buf = copy.deepcopy(self._buf_backup)

    def refresh(self):
        for num, line in enumerate(self._buf):
            self._scr.addstr(num, 0, u''.join(line).encode('utf-8'))
        self._scr.refresh()


class Body:
    def __init__(self, pos, mass, velocity):
        self.pos = pos
        self.mass = mass
        self.vel = velocity


def main(scr):
    setup_curses(scr)
    screen = Screen(scr)
    arr = setup_obstacles(screen)

    screen.draw_arr(arr)
    screen.draw_borders()

    bodies = [
        Body(pos=Vector(110, 80), mass=5, velocity=Vector(0, -40)),
        Body(pos=Vector(50, 80), mass=10, velocity=Vector(0, -40)),
        Body(pos=Vector(95, 80), mass=1, velocity=Vector(0, -40))
    ]

    for b in bodies:
        b.forces = Vector(0, 0)

    t = 0
    freq = 100
    dt = 1/freq

    while True:
        step_simulation(dt, bodies, arr)
        screen.restore_backup()

        for b in bodies:
            screen.draw_hail(b.pos)
        screen.refresh()

        time.sleep(dt)
        t += dt

    curses.endwin()


def setup_stderr():
    """Redirect stderr to other terminal. Run tty command, to get terminal id."""
    sys.stderr = open('/dev/pts/2', 'w')


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


def setup_obstacles(screen):
    file_name = 'ascii_fig.png.norm'
    norm_vec_arr = import_norm_arr(file_name)

    norm_arr_size = Size(norm_vec_arr.shape[1], norm_vec_arr.shape[0])
    arr_size = Size((curses.COLS - 1) * 4, curses.LINES * 8)
    arr = np.zeros(shape=[arr_size.height, arr_size.width, VECTOR_DIM], dtype=norm_vec_arr.dtype)

    x1 = 0
    x2 = x1 + norm_arr_size.width
    y1 = arr_size.height - norm_arr_size.height
    y2 = arr_size.height
    arr[y1:y2, x1:x2] = norm_vec_arr

    return arr


def import_norm_arr(file_name):
    arr = np.loadtxt(file_name)
    height, width = arr.shape
    return arr.reshape(height, width//VECTOR_DIM, VECTOR_DIM)


def ptpos_to_bufpos(pt):
    x = int(pt.x/2)
    y = curses.LINES - 1 - int(pt.y/4)
    return Vector(x, y)


def arrpos_to_ptpos(x, y, arr_size):
    """Array position to cartesian coordinate system"""
    y = arr_size.height - y
    return Vector(x, y)


def ptpos_to_arrpos(pt):
    y = (curses.LINES - 1) * 4 - pt.y
    return Vector(int(pt.x), int(y))


def step_simulation(dt, bodies, arr):
    integrate(dt, bodies)
    collisions = detect_collisions(bodies, arr)
    resolve_collisions(dt, collisions)


def integrate(dt, bodies):
    for b in bodies:
        b.acc = Vector(0, -GRAVITY_ACC) + b.forces/b.mass
        b.vel += b.acc * dt
        b.pos += b.vel * dt


class Collision:
    def __init__(self, body1, body2, relative_vel, collision_normal):
        self.body1 = body1
        self.body2 = body2
        self.relative_vel = relative_vel
        self.collision_normal = collision_normal


def detect_collisions(bodies, obs_arr):
    collisions = []
    for body in bodies:
        collisions += border_collision(body, obs_arr)

    return collisions


def border_collision(body, obs_arr):
    """Check colisions with border"""
    # TODO: dedicated Size should be used instead obs_arr.shape to compare
    if body.pos.x < 0:
        return [Collision(body1=body,
            body2=None,
            relative_vel=body.vel,
            collision_normal=Vector(1, 0))]
    elif body.pos.x > obs_arr.shape[1]:
        return [Collision(body1=body,
            body2=None,
            relative_vel=body.vel,
            collision_normal=Vector(-1, 0))]
    elif body.pos.y < 0:
        return [Collision(body1=body,
            body2=None,
            relative_vel=body.vel,
            collision_normal=Vector(0, 1))]
    elif body.pos.y > obs_arr.shape[0]:
        return [Collision(body1=body,
            body2=None,
            relative_vel=body.vel,
            collision_normal=Vector(0, -1))]

    return []


def resolve_collisions(dt, collisions):
    for c in collisions:
        # Collision with screen border
        if not c.body2:
            rv = np.array([c.relative_vel.y, c.relative_vel.x])
            cn = np.array([c.collision_normal.y, c.collision_normal.x])
            impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * np.dot(rv, cn)) / \
                    1/c.body1.mass

            c.body1.vel += (c.collision_normal / c.body1.mass) * impulse
            c.body1.pos += c.body1.vel * dt


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    setup_stderr()
    curses.wrapper(main)
