#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import collections as co
import itertools as it
import copy
import numpy as np
import curses
import locale
import time


EMPTY_BRAILLE = u'\u2800'

GRAVITY_ACC = 9.8  # [m/s^2]
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


def esetup():
    """Redirect stderr to other terminal. Run tty command, to get terminal id."""
    sys.stderr = open('/dev/pts/4', 'w')


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr)


def eassert(condition):
    if not condition:
        curses.endwin()
        sys.stderr = sys.stdout
        import pdb
        pdb.set_trace()


def main(scr):
    setup_curses(scr)

    bodies = [
        Body(pos=Vector(110, 80), mass=100, velocity=Vector(0, 0)),
        Body(pos=Vector(50, 80), mass=10, velocity=Vector(0, 0)),
        Body(pos=Vector(95, 80), mass=1, velocity=Vector(0, 0))
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
    # for b1, b2 in itertools.combinations(bodies, 2):
    #     calc_forces(b1, b2, dt)

    for b in bodies:
        b.acc = add(Vector(0, -GRAVITY_ACC), div_s(b.forces, b.mass))
        b.vel = add(b.vel, mul_s(b.acc, dt))
        b.pos = add(b.pos, mul_s(b.vel, dt))

    for b in bodies:
        if check_collision(b, obstacles_arr):
            collision(b, obstacles_arr)


    # for b in bodies:
    #     if collision(b.pos, :


        # eprint(mul_s(b.acc, dt))
        # eprint(b.vel)


# def calc_forces(body1, body2, dt):
#     dist = magnitude(body1.pos, body2.pos)
#     if dist < 1:
#         exit()

#     dir1 = normalize(sub(body2.pos, body1.pos))
#     dir2 = normalize(sub(body1.pos, body2.pos))
#     grav_mag = (GRAVITY_ACC * body1.mass * body2.mass) / (dist**2)
#     body1.forces = add(body1.forces, mul_s(dir1, grav_mag))
#     body2.forces = add(body2.forces, mul_s(dir2, grav_mag))


def check_collision(body, obs_arr):
    x, y = point_to_arrpos(body.pos)
    if (obs_arr[y, x] != 0).any():
        eprint('kolizja')
        return True

    return False


def collision(body, obs_arr):
    return
    x, y = point_to_arrpos(body.pos)
    collision_norm = Vector(obs_arr[y, x][1], obs_arr[y, x][0])

    e = 0.5

    j = (-(1+e) * (dot(relativVel, collision_norm))) / \
        ((1/body.mass) + \
         dot(collision_norm, cross(cross(body.collisionPoint, collision_norm) / body.inertia, body.collisionPoint)))

    body.vel += j * collision_norm / body.mass
    # body1.angularVel += cross(body1.collisionPoint, (j * collision_norm)) / body1.inertia

    # body2.vel -= j * collision_norm / body2.mass
    # body2.angularVel -= cross(body2.collisionPoint, (j * collision_norm)) / body2.inertia



class Body:
    def __init__(self, pos, mass, velocity):
        self.pos = pos
        self.mass = mass
        self.vel = velocity


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    esetup()
    curses.wrapper(main)
