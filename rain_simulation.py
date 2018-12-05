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


EMPTY_BRAILLE = u'\u2800'
VECTOR_DIM = 2
GRAVITY_ACC = 9.8  # [m/s^2]
COEFFICIENT_OF_RESTITUTION = 0.5


Size = co.namedtuple('Size', ['width', 'height'])


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
        import pdb
        pdb.set_trace()


def main(scr):
    setup_curses(scr)

    bodies = [
        Body(pos=Vector(110, 80), mass=5, velocity=Vector(0, -40)),
        Body(pos=Vector(50, 80), mass=10, velocity=Vector(0, -40)),
        Body(pos=Vector(95, 80), mass=1, velocity=Vector(0, -40))
    ]

    for b in bodies:
        b.forces = Vector(0, 0)

    obstacles, obstacles_arr = config_scene()

    t = 0
    freq = 100
    dt = 1.0/freq

    while True:
        calcs(bodies, obstacles_arr, dt)
        scene = copy.deepcopy(obstacles)

        for b in bodies:
            draw_point(scene, b.pos)
            pass
        # draw_info(screen, '[%.2f]: %.4f %.4f' % (t, bodies[1].pos.x, bodies[1].pos.y))
        # display(scr, screen)
        display(scr, scene)

        time.sleep(dt)
        t += dt

    curses.endwin()


def setup_curses(scr):
    """Setup curses screen"""
    curses.start_color()
    curses.use_default_colors()
    curses.halfdelay(5)
    curses.noecho()
    curses.curs_set(False)
    scr.clear()


def config_scene():
    file_name = 'ascii_fig.png.norm'
    norm_vec_arr = import_norm_vector_arr(file_name)

    obstacles = empty_scene()
    # draw_arr_as_braille(obstacles, norm_vec_arr)

    norm_arr_size = Size(norm_vec_arr.shape[1], norm_vec_arr.shape[0])
    obstacle_arr_size = Size((curses.COLS - 1) * 4, curses.LINES * 8)
    obstacles_arr = np.zeros(shape=[obstacle_arr_size.height, obstacle_arr_size.width, VECTOR_DIM], dtype=norm_vec_arr.dtype)

    x1 = 0
    x2 = x1 + norm_arr_size.width
    y1 = obstacle_arr_size.height - norm_arr_size.height
    y2 = obstacle_arr_size.height
    obstacles_arr[y1:y2, x1:x2] = norm_vec_arr

    draw_arr_as_braille(obstacles, obstacles_arr)

    return obstacles, obstacles_arr


def import_norm_vector_arr(file_name):
    arr = np.loadtxt(file_name)
    height, width = arr.shape
    return arr.reshape(height, width//VECTOR_DIM, VECTOR_DIM)


def empty_scene():
    return [list(EMPTY_BRAILLE * (curses.COLS - 1)) for _ in range(curses.LINES)]


def draw_arr_as_braille(buff, arr, shift=Vector(0, 0)):
    height, width, _ = arr.shape
    for x, y in it.product(range(width), range(height)):
        if (arr[y, x] != 0).any():
            pt = arrpos_to_point(x, y, Size(width, height))
            pt = Vector(pt.x + shift.x, pt.y + shift.y)
            draw_point(buff, pt)


def draw_point(screen, pt):
    x, y = point_to_buffpos(pt)

    # Out of screen
    if pt.y < 0 or y < 0 or pt.x < 0 or x >= curses.COLS - 1:
        return

    uchar = ord(screen[y][x])
    screen[y][x] = chr(uchar | braille_representation(pt))


def point_to_buffpos(pt):
    x = int(pt.x/2)
    y = curses.LINES - 1 - int(pt.y/4)
    return x, y


def arrpos_to_point(x, y, arr_size):
    """Array position to cartesian coordinate system"""
    y = arr_size.height - y
    return Vector(x, y)


def point_to_arrpos(pt):
    y = (curses.LINES - 1) * 4 - pt.y
    return int(pt.x), int(y)


def braille_representation(pt):
    """Point from cartesian coordinate system to his braille representation"""
    bx = int(pt.x) % 2
    by = int(pt.y) % 4

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


def display(scr, screen):
    for num, line in enumerate(screen):
        scr.addstr(num, 0, u''.join(line).encode('utf-8'))
    scr.refresh()


def calcs(bodies, obstacles_arr, dt):
    for b in bodies:
        b.forces = Vector(0, -GRAVITY_ACC)
        b.acc = b.forces/b.mass
        b.vel += b.acc * dt
        b.pos += b.vel * dt

    collisions = detect_collisions(bodies, obstacles_arr)
    resolve_collisions(dt, collisions)


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
        # Collision with border
        if not c.body2:
            rv = np.array([c.relative_vel.y, c.relative_vel.x])
            cn = np.array([c.collision_normal.y, c.collision_normal.x])
            impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * np.dot(rv, cn)) / \
                    1/c.body1.mass

            c.body1.vel += (c.collision_normal / c.body1.mass) * impulse
            c.body1.pos += c.body1.vel * dt


class Body:
    def __init__(self, pos, mass, velocity):
        self.pos = pos
        self.mass = mass
        self.vel = velocity


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    setup_stderr()
    curses.wrapper(main)
