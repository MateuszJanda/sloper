#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import sys
import collections as co
import itertools as it
import numpy as np
import curses
import locale
import time


EMPTY_BRAILLE = u'\u2800'

G = 9.8             # [m/s^2]
VECTOR_DIM = 2


Size = co.namedtuple('Size', ['width', 'height'])

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        """ string representation of an object """
        return "<" + str(self.x) + ", " + str(self.y) + ">"


def magnitude(vec1, vec2):
    return math.sqrt((vec1.x - vec2.x)**2 + (vec1.y - vec2.y)**2)


def normalize(vec):
    mag = magnitude(Vector(0, 0), vec)
    return Vector(vec.x / mag, vec.y / mag)


def mul_s(vec, s):
    return Vector(vec.x * s, vec.y * s)


def div_s(vec, s):
    return Vector(vec.x / s, vec.y / s)


def sub(vec1, vec2):
    return Vector(vec1.x - vec2.x, vec1.y - vec2.y)


def add(vec1, vec2):
    return Vector(vec1.x + vec2.x, vec1.y + vec2.y)


def main():
    locale.setlocale(locale.LC_ALL, '')
    esetup()
    curses.wrapper(run)


def esetup():
    sys.stderr = open('/dev/pts/2', 'w')


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr)


def eassert(condition):
    if not condition:
        curses.endwin()
        sys.stderr = sys.stdout
        import pdb
        pdb.set_trace()


def run(scr):
    setup_curses(scr)

    file_name = 'ascii_fig.png.norm'
    norm_vec_arr = import_norm_vector_arr(file_name)

    bodies = [
        Body(pos=Vector(110, 80), mass=100, velocity=Vector(0, 0)),
        Body(pos=Vector(50, 80), mass=10, velocity=Vector(0, 0)),
        Body(pos=Vector(95, 80), mass=1, velocity=Vector(0, 0))
    ]

    t = 0
    freq = 100
    dt = 1.0/freq

    screen = empty_scene()
    obstacles = empty_scene()
    draw_arr_as_braille(obstacles, norm_vec_arr)


    while True:
        calcs(bodies, dt)

        for b in bodies:
            draw_point(screen, b.pos)
            pass
        # draw_info(screen, '[%.2f]: %.4f %.4f' % (t, bodies[1].pos.x, bodies[1].pos.y))
        # display(scr, screen)
        display(scr, screen)

        time.sleep(0.01)
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
    screen[y][x] = unichr(uchar | point_to_braille(pt))


def point_to_buffpos(pt):
    """Point coordinate to buffer pos"""
    x = int(pt.x/2)
    y = curses.LINES - 1 - int(pt.y/4)

    return x, y


def arrpos_to_point(x, y, arr_size):
    """Array position to cartesian coordinate system"""
    y = arr_size.height - y
    return Vector(x, y)


def point_to_braille(pt):
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


def calcs(bodies, dt):
    for b in bodies:
        b.forces = Vector(0, 0)

    # for b1, b2 in itertools.combinations(bodies, 2):
    #     calc_forces(b1, b2, dt)

    for b in bodies:
        b.acc = add(Vector(0, -G), div_s(b.forces, b.mass))
        b.vel = add(b.vel, mul_s(b.acc, dt))
        b.pos = add(b.pos, mul_s(b.vel, dt))

        # eprint(mul_s(b.acc, dt))
        # eprint(b.vel)


# def calc_forces(body1, body2, dt):
#     dist = magnitude(body1.pos, body2.pos)
#     if dist < 1:
#         exit()

#     dir1 = normalize(sub(body2.pos, body1.pos))
#     dir2 = normalize(sub(body1.pos, body2.pos))
#     grav_mag = (G * body1.mass * body2.mass) / (dist**2)
#     body1.forces = add(body1.forces, mul_s(dir1, grav_mag))
#     body2.forces = add(body2.forces, mul_s(dir2, grav_mag))


class Body:
    def __init__(self, pos, mass, velocity):
        self.pos = pos
        self.mass = mass
        self.vel = velocity


if __name__ == '__main__':
    main()
