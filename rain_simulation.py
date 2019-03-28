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
COEFFICIENT_OF_RESTITUTION = 0.3
COEFFICIENT_OF_FRICTION = 0.9


def main(scr):
    setup_curses(scr)
    terrain = Terrain()
    screen = Screen(scr, terrain)

    im = Importer()
    ascii_arr, norm_arr = im.load('ascii_fig.txt', 'ascii_fig.png.norm')

    terrain.add_arr(norm_arr)
    screen.add_ascii(ascii_arr)
    # screen.add_norm_arr(norm_arr)

    for x in range(terrain._terrain.shape[1]):
        if terrain._terrain[-1, x].any():
            eprint('POS', x)

    assert(np.all(Vector(x=50, y=38) == Vector(x=50, y=38)))
    arrpos = ptpos_to_arrpos(Vector(x=50, y=38))
    assert(np.all(Vector(x=50, y=38) == arrpos_to_ptpos(arrpos.x, arrpos.y)))

    bodies = [
        # Body(ptpos=Vector(x=30, y=80), mass=10, velocity=Vector(x=0, y=-40))
        Body(ptpos=Vector(x=50, y=80), mass=10, velocity=Vector(x=0, y=-40)),  # TODO: check interaction with normal vector
        # Body(ptpos=Vector(x=95, y=80), mass=1, velocity=Vector(x=0, y=-40)),
        # Body(ptpos=Vector(x=110, y=80), mass=1, velocity=Vector(x=0, y=-40)),
        # Body(ptpos=Vector(x=23, y=80), mass=1, velocity=Vector(x=0, y=-40)),
    ]

    t = 0
    freq = 100
    dt = 1/freq

    while True:
        screen.restore_backup()

        step_simulation(dt, bodies, terrain)

        for body in bodies:
            # screen.draw_hailstone(body.ptpos)
            screen.draw_point(body.ptpos)
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
    # TODO: inna kolejność (y, x)
    def __new__(cls, y, x):
        obj = np.asarray([y, x]).view(cls)
        return obj

    @property
    def x(self):
        return self[1]

    @x.setter
    def x(self, value):
        self[1] = value

    @property
    def y(self):
        return self[0]

    @y.setter
    def y(self, value):
        self[0] = value

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
    def __init__(self, scr, terrain):
        self._scr = scr
        self._terrain = terrain

        self._buf_size = Size(curses.COLS-1, curses.LINES)
        self._arr_size = Size(self._buf_size.width*BUF_CELL_SIZE.width,
                              self._buf_size.height*BUF_CELL_SIZE.height)

        self._buf = self._get_empty_buf()
        self._buf_backup = copy.deepcopy(self._buf)

    def _get_empty_buf(self):
        return [list(EMPTY_BRAILLE * self._buf_size.width) for _ in range(self._buf_size.height)]

    def add_norm_arr(self, arr, shift=Vector(x=0, y=0)):
        """
        Add static element to screen buffer. Every element in array will be
        represent as braille character. By default all arrays are drawn in
        bottom left corner.

        TODO: shift should be buf_shift
        TODO: replace by redraw_terain
        """
        height, width, _ = arr.shape
        # TODO: moze np.argwhere ???
        for x, y in it.product(range(width), range(height)):
            if np.any(arr[y, x] != 0):
                ptpos = arrpos_to_ptpos(x, self._arr_size.height - height  + y) + shift
                self.draw_point(ptpos)

        self._save_in_backup_buf()

    def add_ascii(self, ascii_arr, shift=Vector(x=0, y=0)):
        height, width = ascii_arr.shape
        for x, y in it.product(range(width), range(height)):
            if np.any(ascii_arr[y, x] != ' '):
                buffpos = Vector(x=x, y=self._buf_size.height - height + y)
                self._buf[buffpos.y][buffpos.x] = ascii_arr[y, x]

        self._save_in_backup_buf()

    def add_border(self):
        """For debug, draw screen border in braille characters"""
        for x in range(self._arr_size.width):
            self.draw_point(Vector(x=x, y=0))
            self.draw_point(Vector(x=x, y=self._arr_size.height-1))

        for y in range(self._arr_size.height):
            self.draw_point(Vector(x=0, y=y))
            self.draw_point(Vector(x=self._arr_size.width-1, y=y))

        self._save_in_backup_buf()

    def _save_in_backup_buf(self):
        """Backup screen buffer"""
        self._buf_backup = copy.deepcopy(self._buf)

    def draw_point(self, pt):
        # Out of the screen
        if not (0 <= pt.x < self._arr_size.width and 0 <= pt.y < self._arr_size.height):
            eprint('ERR')
            return

        bufpos = ptpos_to_bufpos(pt)
        block = self._terrain.get_buff_size_block(bufpos)
        if ord(self._buf[bufpos.y][bufpos.x]) < ord(EMPTY_BRAILLE) and np.any(block):
            uchar = self._block_to_uchar(block)
            # uchar = ord(self._buf[bufpos.y][bufpos.x])
        else:
            uchar = ord(self._buf[bufpos.y][bufpos.x])

        self._buf[bufpos.y][bufpos.x] = chr(uchar | self._braille_char(pt))

    def _block_to_uchar(self, block):
        height, width = block.shape
        uchar = ord(EMPTY_BRAILLE)
        for x, y in it.product(range(width), range(height)):
            if block[y, x]:
                uchar |= self._braille_char(Vector(x=x, y=BUF_CELL_SIZE.height-y))

        return uchar

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
        # self.prev_prev_ptpos = ptpos
        self.prev_ptpos = ptpos
        # self.next_ptpos = None
        self.mass = mass
        self.vel = velocity
        self.lock = False

    def is_moving(self):
        return not math.isclose(self.vel.magnitude(), 0, abs_tol=0.01)


class Terrain:
    NO_VECTOR = np.array([0, 0])

    def __init__(self):
        self._terrain_size = Size((curses.COLS-1)*BUF_CELL_SIZE.width, curses.LINES*BUF_CELL_SIZE.height)
        self._terrain = np.zeros(shape=[self._terrain_size.height, self._terrain_size.width, VECTOR_DIM])

    def size(self):
        return self._terrain_size

    def add_arr(self, arr, shift=Vector(x=0, y=0)):
        """By default all arrays are drawn in bottom left corner."""
        arr_size = Size(arr.shape[1], arr.shape[0])

        x1 = shift.x
        x2 = x1 + arr_size.width
        y1 = self._terrain_size.height - arr_size.height - shift.y
        y2 = self._terrain_size.height - shift.y
        self._terrain[y1:y2, x1:x2] = arr

    def get_normal_vec(self, pt):
        arrpos = ptpos_to_arrpos(pt)

        # if (pt.x == 28 or pt.x == 29) and pt.y == 0:
            # eprint('PYK')

        normal_vec = self._terrain[arrpos.y, arrpos.x]
        return Vector(x=normal_vec[0], y=normal_vec[1])

    def get_buff_size_block(self, bufpos):
        pt = Vector(bufpos.x * BUF_CELL_SIZE.width, bufpos.y * BUF_CELL_SIZE.height)

        block = self._terrain[pt.y:pt.y+BUF_CELL_SIZE.height,
                              pt.x:pt.x+BUF_CELL_SIZE.width]


        block = np.logical_or.reduce(block != Terrain.NO_VECTOR, axis=-1)
        return block

    def in_border(self, pt):
        arrpos = ptpos_to_arrpos(pt)
        return 0 <= arrpos.x < self._terrain_size.width and \
               0 <= arrpos.y < self._terrain_size.height

    def _bounding_box(self, ptpos, prev_ptpos):
        arr_pos = ptpos_to_arrpos(ptpos)
        arr_prev_pos = ptpos_to_arrpos(prev_ptpos)

        x1, x2 = min(arr_pos.x, arr_prev_pos.x), max(arr_pos.x, arr_prev_pos.x)
        y1, y2 = min(arr_pos.y, arr_prev_pos.y), max(arr_pos.y, arr_prev_pos.y)

        return x1, x2, y1, y2

    def fff(self, ptpos, prev_ptpos):
        x1, x2, y1, y2 = self._bounding_box(ptpos, prev_ptpos)

        box_normal_vec = self._terrain[y1:y2, x1:x2]
        box_markers = np.logical_or.reduce(box_normal_vec != Terrain.NO_VECTOR, axis=-1)
        indices = np.argwhere(box_markers)

        result = []
        for arrpos in indices:
            obstacle_pos = arrpos_to_ptpos(Vector(x=x1, y=y1) + Vector(*arrpos))
            normal_vec = box_normal_vec[arrpos.y, arrpos.x]
            normal_vec = Vector(x=normal_vec[0], y=normal_vec[1])
            result.append(obstacle_pos, normal_vec)

        return result

class Importer:
    def load(self, ascii_file, norm_file):
        ascii_arr = self._import_ascii_arr(ascii_file)
        ascii_arr = self._remove_ascii_marker(ascii_arr)
        ascii_arr = self._remove_ascii_margin(ascii_arr)

        norm_arr = self._import_norm_arr(norm_file)
        norm_arr = self._remove_norm_margin(norm_arr)

        self._transform_norm(norm_arr)

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
    return Vector(x=x, y=y)


def arrpos_to_ptpos(x, y):
    """Array position to Cartesian coordinate system"""
    y = curses.LINES * BUF_CELL_SIZE.height - 1 - y
    return Vector(x=x, y=y)


def ptpos_to_arrpos(pt):
    y = curses.LINES * BUF_CELL_SIZE.height - 1 - pt.y
    return Vector(x=int(pt.x), y=int(y))


def step_simulation(dt, bodies, terrain):
    calc_forces(dt, bodies)
    integrate(dt, bodies)
    collisions = detect_collisions(bodies, terrain)
    resolve_collisions(dt, collisions)
    fix_penetration(bodies, terrain.size())


def calc_forces(dt, bodies):
    for body in bodies:
        body.forces = Vector(x=0, y=-GRAVITY_ACC) * body.mass

        # if int(body.prev_ptpos.y) == 0 and int(body.ptpos.y) == 0 and int(body.prev_ptpos.x) != int(body.ptpos.x):
        if int(body.prev_ptpos.y) == 0 and int(body.ptpos.y) == 0:
            body.forces *= COEFFICIENT_OF_FRICTION


def integrate(dt, bodies):
    for body in bodies:
        if body.lock:
            eprint('LOCK')
            continue

        body.prev_ptpos = copy.copy(body.ptpos)
        # if np.any(body.next_ptpos):
        #     body.ptpos = copy.copy(body.next_ptpos)
        #     body.next_ptpos = None
        # else:


        # body.acc = Vector(x=0, y=-GRAVITY_ACC) + body.forces/body.mass
        body.acc = body.forces / body.mass
        body.vel = body.vel + body.acc * dt
        # body.prev_prev_ptpos = body.prev_ptpos
        # body.prev_ptpos = body.ptpos
        body.ptpos = body.ptpos + body.vel * dt

        # Don't calculate collision if body is not moving
        # if math.isclose(body.vel.magnitude(), 0, abs_tol=0.01):
        if not body.is_moving():
            body.lock = True


class Collision:
    def __init__(self, body1, body2, relative_vel, normal_vec, dist=0):
        self.body1 = body1
        self.body2 = body2
        self.relative_vel = relative_vel
        self.dist = dist
        self.normal_vec = normal_vec


def detect_collisions(bodies, terrain):
    collisions = []
    for body in bodies:
        if body.lock:
            continue
        # c = obstacle_collisions(body, terrain)
        c = obstacle_collisions3(body, terrain)
        # if c:
        #     eprint('OBSTACLE')
        if not c:
            c = border_collision(body, terrain.size())
        collisions += c

    return collisions


def border_collision(body, terrain_size):
    """Check collisions with border"""
    if body.ptpos.x < 0:
        return [Collision(body1=body,
                          body2=None,
                          relative_vel=-body.vel,
                          normal_vec=Vector(x=1, y=0))]
    elif body.ptpos.x > terrain_size.width:
        return [Collision(body1=body,
                          body2=None,
                          relative_vel=-body.vel,
                          normal_vec=Vector(x=-1, y=0))]
    elif body.ptpos.y < 0:
        return [Collision(body1=body,
                          body2=None,
                          relative_vel=-body.vel,
                          normal_vec=Vector(x=0, y=1))]
    elif body.ptpos.y > terrain_size.height:
        return [Collision(body1=body,
                          body2=None,
                          relative_vel=-body.vel,
                          normal_vec=Vector(x=0, y=-1))]

    return []


def obstacle_collisions(body, terrain):
    normal_vec = obstacle_pos(terrain, body.prev_ptpos, body.ptpos)
    if np.any(normal_vec):
        return [Collision(body1=body,
                          body2=None,
                          relative_vel=-body.vel,
                          normal_vec=normal_vec)]
    return []


def obstacle_collisions2(body, terrain):
    for check_pt in path(body):
        if not terrain.in_border(check_pt):
            break

        normal_vec = terrain.get_normal_vec(check_pt)
        if np.any(normal_vec):
            return [Collision(body1=body,
                      body2=None,
                      relative_vel=-body.vel,
                      normal_vec=normal_vec)]

    return []


def obstacle_collisions3(body, terrain):
    result = []
    for ptpos, normal_vec in terrain.fff(body.ptpos, body.prev_ptpos):
        collision = Collision(body1=body,
                              body2=None,
                              relative_vel=-body.vel,
                              dist=body.ptpos - ptpos,
                              normal_vec=normal_vec)

        result.append(collision)

    return result


    # for check_pt in path(body):
    #     if not terrain.in_border(check_pt):
    #         break

    #     normal_vec = terrain.get_normal_vec(check_pt)
    #     if np.any(normal_vec):
    #         return [Collision(body1=body,
    #                   body2=None,
    #                   relative_vel=-body.vel,
    #                   normal_vec=normal_vec)]

    # return []


def path(body):
    """Bresenham's line algorithm
    https://pl.wikipedia.org/wiki/Algorytm_Bresenhama
    """
    check_pt = np.floor(body.prev_ptpos)
    prev_ptpos = np.floor(body.prev_ptpos)
    ptpos = np.floor(body.ptpos)

    if prev_ptpos.x < ptpos.x:
        xi = 1
        dx = ptpos.x - prev_ptpos.x
    else:
        xi = -1
        dx = prev_ptpos.x - ptpos.x

    if prev_ptpos.y < ptpos.y:
        yi = 1
        dy = ptpos.y - prev_ptpos.y
    else:
        yi = -1
        dy = prev_ptpos.y - ptpos.y

    yield check_pt

    # X axis
    if dx > dy:
        ai = (dy - dx) * 2
        bi = dy * 2
        d = bi - dx
        while check_pt.x != ptpos.x:
            # coordinate test
            if d >= 0:
                check_pt.x += xi
                check_pt.y += yi
                d += ai
            else:
                d += bi
                check_pt.x += xi

            yield check_pt
    else:
        ai = (dx - dy) * 2
        bi = dx * 2
        d = bi - dy
        while check_pt.y != ptpos.y:
            # coordinate test
            if d >= 0:
                check_pt.x += xi
                check_pt.y += yi
                d += ai
            else:
                d += bi
                check_pt.y += yi

            yield check_pt


def obstacle_pos(terrain, prev_ptpos, ptpos):
    """Bresenham's line algorithm
    https://pl.wikipedia.org/wiki/Algorytm_Bresenhama
    """
    check_pt = np.floor(prev_ptpos)
    prev_ptpos = np.floor(prev_ptpos)
    ptpos = np.floor(ptpos)

    if prev_ptpos.x < ptpos.x:
        xi = 1
        dx = ptpos.x - prev_ptpos.x
    else:
        xi = -1
        dx = prev_ptpos.x - ptpos.x

    if prev_ptpos.y < ptpos.y:
        yi = 1
        dy = ptpos.y - prev_ptpos.y
    else:
        yi = -1
        dy = prev_ptpos.y - ptpos.y

    if not terrain.in_border(check_pt):
        return None

    normal_vec = terrain.get_normal_vec(check_pt)
    # If body collide with different obstacle than in previous step
    if np.any(normal_vec) and not (check_pt.y != prev_ptpos.y or check_pt.x != prev_ptpos.x):
        eprint('--->  UPS', check_pt.x, prev_ptpos.x)
    if np.any(normal_vec) and (check_pt.y != prev_ptpos.y or check_pt.x != prev_ptpos.x):
        return normal_vec

    # X axis
    if dx > dy:
        ai = (dy - dx) * 2
        bi = dy * 2
        d = bi - dx
        while check_pt.x != ptpos.x:
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
        while check_pt.y != ptpos.y:
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
    if collisions:
        eprint('RESOLV')

    for i in range(3):
        for c in collisions:
            # Collision with screen border
            if not c.body2:

                remove = np.dot(c.relative_vel, c.normal_vec) + c.dist/dt

                if remove < 0:

                    # impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * np.dot(c.relative_vel, c.normal_vec)) / \
                    #         (1/c.body1.mass)

                    impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * remove) / \
                            (1/c.body1.mass)

                    # eprint('pos 1', c.body1.ptpos)
                    # eprint('prev 1', c.body1.prev_ptpos)
                    c.body1.vel -= (c.normal_vec / c.body1.mass) * impulse
                    c.body1.ptpos += c.body1.vel * dt
                    # c.body1.next_ptpos = c.body1.ptpos + c.body1.vel * dt

                    # eprint('vel', c.body1.vel)
                    # eprint('normal', c.normal_vec)
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
