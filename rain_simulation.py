#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Coordinates systems:
    pos         - position in Cartesian coordinate system
    buf_pos     - position on screen (of one character). Y from top to bottom
    arr_pos     - similar to ptpos, but Y from top to bottom
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


class Size(np.ndarray):
    def __new__(cls, height, width):
        obj = np.asarray([height, width]).view(cls)
        return obj

    @property
    def width(self):
        return self[1]

    @width.setter
    def width(self, value):
        self[1] = value

    @property
    def height(self):
        return self[0]

    @height.setter
    def height(self, value):
        self[0] = value

    def __str__(self):
        """string representation of object."""
        return "Size(width=" + str(self.width) + ", height=" + str(self.height) + ")"

    def __repr__(self):
        """string representation of object."""
        return "Size(width=" + str(self.width) + ", height=" + str(self.height) + ")"


REFRESH_RATE = 100
EMPTY_BRAILLE = u'\u2800'
BUF_CELL_SIZE = Size(4, 2)
VECTOR_DIM = 2

# Physical values
GRAVITY_ACC = 9.8  # [m/s^2]
COEFFICIENT_OF_RESTITUTION = 0.3
COEFFICIENT_OF_FRICTION = 0.9


def main(scr):
    test_converters()
    setup_curses(scr)
    terrain = Terrain()
    screen = Screen(scr, terrain)

    im = Importer()
    ascii_arr, norm_arr = im.load('ascii_fig.txt', 'ascii_fig.png.norm')

    terrain.add_arr(norm_arr)
    # screen.add_ascii(ascii_arr)
    screen.add_norm_arr(norm_arr)

    bodies = [
        # Body(ptpos=Vector(x=34, y=80), mass=10, velocity=Vector(x=0, y=-40))
        Body(ptpos=Vector(x=50, y=80), mass=10, velocity=Vector(x=0, y=-40)),
        # Body(ptpos=Vector(x=95, y=80), mass=1, velocity=Vector(x=0, y=-40)),
        # Body(ptpos=Vector(x=110, y=80), mass=1, velocity=Vector(x=0, y=-40)),
        # Body(ptpos=Vector(x=23, y=80), mass=1, velocity=Vector(x=0, y=-40)),
    ]

    t = 0
    dt = 1/REFRESH_RATE

    while True:
        screen.restore_buffer()

        step_simulation(dt, bodies, terrain)

        for body in bodies:
            # screen.draw_hailstone(body.ptpos)
            screen.draw_point(body.ptpos)
        screen.refresh()

        time.sleep(dt)
        t += dt

    curses.endwin()


def setup_stderr():
    """
    Redirect stderr to other terminal. Run tty command, to get terminal id.
    """
    sys.stderr = open('/dev/pts/3', 'w')


def eprint(*args, **kwargs):
    """Print on stderr"""
    print(*args, file=sys.stderr)


def eassert(condition):
    """Assert. Disable curses and run pdb."""
    if not condition:
        curses.endwin()
        sys.stderr = sys.stdout
        pdb.set_trace()


def setup_curses(scr):
    """Setup curses screen."""
    curses.start_color()
    curses.use_default_colors()
    curses.halfdelay(5)
    curses.noecho()
    curses.curs_set(False)
    scr.clear()


class Vector(np.ndarray):
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
        """Calculate vector magnitude."""
        return np.linalg.norm(self)

    def __str__(self):
        """string representation of object."""
        return "Vector(x=" + str(self.x) + ", y=" + str(self.y) + ")"

    def __repr__(self):
        """string representation of object."""
        return "Vector(x=" + str(self.x) + ", y=" + str(self.y) + ")"


class Screen:
    def __init__(self, scr, terrain):
        self._scr = scr
        self._terrain = terrain

        self._buf_size = Size(curses.LINES, curses.COLS-1)
        self._arr_size = self._buf_size*BUF_CELL_SIZE

        self._buf = self._create_empty_buf()
        self._buf_backup = copy.deepcopy(self._buf)

    def _create_empty_buf(self):
        """
        Create empty screen buffer filled with "empty braille" chracters.
        TODO: Repleace it by np.array (constrained type)
        """
        return [list(EMPTY_BRAILLE * self._buf_size.width) for _ in range(self._buf_size.height)]

    def add_norm_arr(self, arr, buf_shift=Vector(x=0, y=0)):
        """For DEBUG
        Add static element to screen buffer. Every element in array will be
        represent as braille character. By default all arrays are drawn in
        bottom left corner.

        TODO: replace by redraw_terain
        """
        arr_shift = Vector(x=buf_shift.x*BUF_CELL_SIZE.width, y=buf_shift.y*BUF_CELL_SIZE.height)
        height, width, _ = arr.shape
        # TODO: moze np.argwhere ???
        for x, y in it.product(range(width), range(height)):
            if np.any(arr[y, x] != 0):
                ptpos = arrpos_to_ptpos(x, self._arr_size.height - height  + y) + arr_shift
                self.draw_point(ptpos)

        self._save_in_backup_buf()

    def add_ascii(self, ascii_arr, buf_shift=Vector(x=0, y=0)):
        """
        Add static element to screen buffer. By default array will be drawn in
        bottom left corner.
        """
        height, width = ascii_arr.shape
        for x, y in it.product(range(width), range(height)):
            if np.any(ascii_arr[y, x] != ' '):
                bufpos = Vector(x=x, y=self._buf_size.height - height + y)
                self._buf[bufpos.y][bufpos.x] = ascii_arr[y, x]

        self._save_in_backup_buf()

    def add_border(self):
        """For DEBUG
        Draw screen border in braille characters.
        """
        for x in range(self._arr_size.width):
            self.draw_point(Vector(x=x, y=0))
            self.draw_point(Vector(x=x, y=self._arr_size.height-1))

        for y in range(self._arr_size.height):
            self.draw_point(Vector(x=0, y=y))
            self.draw_point(Vector(x=self._arr_size.width-1, y=y))

        self._save_in_backup_buf()

    def _save_in_backup_buf(self):
        """Backup screen buffer."""
        self._buf_backup = copy.deepcopy(self._buf)

    def draw_point(self, pt):
        """
        Draw (put in screen buffer) single point. If theres is any ASCII
        character in screen cell, function will replace this character to his
        braille representation and merge this single point.
        """
        # Don't draw point when they are out of the screen
        if not (0 <= pt.x < self._arr_size.width and 0 <= pt.y < self._arr_size.height):
            return

        bufpos = ptpos_to_bufpos(pt)
        cell_box = self._terrain.cut_bufcell_box(bufpos)
        if ord(self._buf[bufpos.y][bufpos.x]) < ord(EMPTY_BRAILLE) and np.any(cell_box):
            uchar = self._cell_box_to_uchar(cell_box)
        else:
            uchar = ord(self._buf[bufpos.y][bufpos.x])

        self._buf[bufpos.y][bufpos.x] = chr(uchar | self._pos_to_braille(pt))

    def _cell_box_to_uchar(self, cell_box):
        """
        Convert BUF_CELL_SIZE cell_box to his braille character representation.
        """
        height, width = cell_box.shape
        uchar = ord(EMPTY_BRAILLE)
        for x, y in it.product(range(width), range(height)):
            if cell_box[y, x]:
                uchar |= self._pos_to_braille(Vector(x=x, y=BUF_CELL_SIZE.height-y))

        return uchar

    def _pos_to_braille(self, pt):
        """Point position as braille character in BUF_CELL."""
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

    def restore_buffer(self):
        """Restore static elements added to screen."""
        self._buf = copy.deepcopy(self._buf_backup)

    def refresh(self):
        """Draw buffer content to screen."""
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
        self._terrain_size = Size(curses.LINES*BUF_CELL_SIZE.height,
                                  (curses.COLS-1)*BUF_CELL_SIZE.width)
        self._terrain = np.zeros(shape=(self._terrain_size.height,
                                        self._terrain_size.width, VECTOR_DIM))

    def size(self):
        return self._terrain_size

    def add_arr(self, arr, shift=Vector(x=0, y=0)):
        """By default all arrays are drawn in bottom left corner."""
        arr_size = Size(*arr.shape[:2])

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

    def cut_bufcell_box(self, bufpos):
        pt = Vector(x=bufpos.x * BUF_CELL_SIZE.width, y=bufpos.y * BUF_CELL_SIZE.height)

        cell_box = self._terrain[pt.y:pt.y+BUF_CELL_SIZE.height,
                                 pt.x:pt.x+BUF_CELL_SIZE.width]

        cell_box = np.logical_or.reduce(cell_box != Terrain.NO_VECTOR, axis=-1)
        return cell_box

    def in_border(self, pt):
        arrpos = ptpos_to_arrpos(pt)
        return 0 <= arrpos.x < self._terrain_size.width and \
               0 <= arrpos.y < self._terrain_size.height

    def _bounding_box(self, ptpos, prev_ptpos):
        arr_pos = ptpos_to_arrpos(ptpos)
        arr_prev_pos = ptpos_to_arrpos(prev_ptpos)

        # eprint('what ', arr_pos.y, arr_prev_pos.y)
        x1, x2 = min(arr_pos.x, arr_prev_pos.x), max(arr_pos.x, arr_prev_pos.x)
        y1, y2 = min(arr_pos.y, arr_prev_pos.y), max(arr_pos.y, arr_prev_pos.y)
        # eprint(x1, y1, x2, y2)
        # eprint('from arrr', arrpos_to_ptpos(x1, y1), arrpos_to_ptpos(x2, y2))
        return Vector(x=x1-1, y=y1-1), Vector(x=x2+2, y=y2+2)

    def _cut_normal_vec_box(self, arr_tl, arr_br):
        # Fit array corner coordinates to not go out-of-bounds
        tl = Vector(x=max(arr_tl.x, 0), y=max(arr_tl.y, 0))
        br = Vector(x=min(arr_br.x, self._terrain_size.width),
                    y=min(arr_br.y, self._terrain_size.height))
        # Cut normal vectors from terrain array
        box = self._terrain[tl.y:br.y, tl.x:br.x]
        arr_shift = Vector(x=0, y=0)

        # If bounding box is out of terrain bounds, we need to add border padding
        expected_shape = (arr_br.y - arr_tl.y, arr_br.x - arr_tl.x, VECTOR_DIM)
        if expected_shape == box.shape:
            return box, arr_shift

        # Pad borders. If bounding box top-left corner is out of bound,
        # we need also create shit for terrain top-left corner position. It
        # will be needed to calculate distance.
        if arr_tl.x < 0:
            box = np.hstack((np.full(shape=(box.shape[0], 1, VECTOR_DIM), fill_value=[1,0]), box))
            arr_shift = Vector(x=arr_shift.x+1, y=arr_shift.y)
        if arr_tl.y < 0:
            arr_shift = Vector(x=arr_shift.x, y=arr_shift.y+1)
            box = np.vstack((np.full(shape=(1, box.shape[1], VECTOR_DIM), fill_value=[0,-1]), box))

        if arr_br.x > self._terrain_size.width:
            box = np.hstack((box, np.full(shape=(box.shape[0], 1, VECTOR_DIM), fill_value=[-1,0])))
        if arr_br.y > self._terrain_size.height:
            box = np.vstack((box, np.full(shape=(1, box.shape[1], VECTOR_DIM), fill_value=[0,1])))

        # Fix corners position, normal vector should guide to center of screen
        if arr_tl.x < 0 and arr_tl.y < 0:
            box[0, 0] = Vector(x=0.7071, y=-0.7071)
        if arr_tl.x < 0 and arr_br.y > self._terrain_size.height:
            box[-1, 0] = Vector(x=0.7071, y=0.7071)

        if arr_br.x > self._terrain_size.width and arr_tl.y < 0:
            box[0, -1] = Vector(x=-0.7071, y=-0.7071)
        if arr_br.x > self._terrain_size.width and arr_br.y > self._terrain_size.height:
            box[0, -1] = Vector(x=-0.7071, y=0.7071)

        return box, arr_shift


    def obstacles(self, ptpos, prev_ptpos):
        arr_tl, arr_br = self._bounding_box(ptpos, prev_ptpos)

        # if np.floor(ptpos.y) == 45:
        #     eprint('CORNER prev=%s, c1=%s, c2=%s' % (prev_ptpos, arrpos_to_ptpos(corner1.x, corner1.y), arrpos_to_ptpos(corner2.x, corner2.y)))
        #     exit()

        box, arr_shift = self._cut_normal_vec_box(arr_tl, arr_br)

        # if np.floor(ptpos.y) == 45:
        #     eprint(box_normal_vec)
        #     eprint('CORNER prev=%s, c1=%s, c2=%s' % (prev_ptpos, corner1, corner2))
        #     exit()

        box_markers = np.logical_or.reduce(box != Terrain.NO_VECTOR, axis=-1)

        result = []
        for arrpos in np.argwhere(box_markers):
            arrpos = Vector(*arrpos)
            normal_vec = box[arrpos.y, arrpos.x]
            normal_vec = Vector(x=normal_vec[0], y=normal_vec[1])

            global_pos = Vector(*(arr_tl + arr_shift + arrpos))
            global_pos = arrpos_to_ptpos(x=global_pos.x, y=global_pos.y)
            result.append((global_pos, normal_vec))


        # if np.floor(ptpos.y) == 45:
        #     eprint('box result ', result)
        #     exit()


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

        size = norm_reduce.shape//BUF_CELL_SIZE
        result = np.reshape(result, size)

        eprint(result)
        return result

    def _validate_arrays(self, ascii_arr, norm_arr):
        """Validate if both arrays describe same thing"""
        ascii_arr_size = Size(*ascii_arr.shape)
        norm_arr_size = norm_arr.shape[:2]//BUF_CELL_SIZE

        if np.any(ascii_arr_size != norm_arr_size):
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
    y = curses.LINES * BUF_CELL_SIZE.height - 1 - int(pt.y)
    return Vector(x=int(pt.x), y=int(y))


def step_simulation(dt, bodies, terrain):
    calc_forces(dt, bodies)
    integrate(dt, bodies)
    collisions = detect_collisions(bodies, terrain)
    resolve_collisions(dt, collisions)
    # fix_penetration(bodies, terrain.size())


def calc_forces(dt, bodies):
    for body in bodies:
        body.forces = Vector(x=0, y=-GRAVITY_ACC) * body.mass

        # if int(body.prev_ptpos.y) == 0 and int(body.ptpos.y) == 0 and int(body.prev_ptpos.x) != int(body.ptpos.x):
        if int(body.prev_ptpos.y) == 0 and int(body.ptpos.y) == 0:
            body.forces *= COEFFICIENT_OF_FRICTION


def integrate(dt, bodies):
    for body in bodies:
        # if body.lock:
        #     eprint('LOCK')
        #     continue

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
        # eprint("NEW POS", body.ptpos)

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
        # if body.lock:
        #     continue
        # c = obstacle_collisions(body, terrain)
        c = obstacle_collisions3(body, terrain)
        # if c:
        #     eprint('OBSTACLE')

        # if not c:
        #     c = border_collision(body, terrain.size())

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

   # if int(body.ptpos.y) == 45:
   #      eprint('ALL collision', len(result))
   #      exit()

    for obstacle_ptpos, normal_vec in terrain.obstacles(body.ptpos, body.prev_ptpos):
        # eprint('pt to distance', body.ptpos, ptpos)
        # dist = Vector(*(ptpos - body.ptpos))
        # eprint('dist ', dist.magnitude())

        # dist = min(
        #     Vector(*(ptpos - body.ptpos)).magnitude(),
        #     Vector(*(ptpos + Vector(x=1, y=0) - body.ptpos)).magnitude(),
        #     Vector(*(ptpos + Vector(x=0, y=1) - body.ptpos)).magnitude(),
        #     Vector(*(ptpos + Vector(x=1, y=1) - body.ptpos)).magnitude(),
        #     )

        r = 0.5
        p1 = np.floor(body.ptpos) + Vector(x=r, y=r)
        # p1 = body.ptpos + Vector(x=r, y=r)
        p2 = np.floor(obstacle_ptpos) + Vector(x=r, y=r)
        dist = Vector(*(p1 - p2)).magnitude() - 2*r

        collision = Collision(body1=body,
                              body2=None,
                              relative_vel=-body.vel,
                              dist=dist,
                              normal_vec=normal_vec)

        collision.obs_pos = obstacle_ptpos

        result.append(collision)


    # if int(body.ptpos.y) == 45:
    #     eprint('ALL collision', len(result))
    #     exit()

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
    mark = False
    for i in range(3):
        for c in collisions:
            # Collision with screen border
            if not c.body2:

                relative_vel = -c.body1.vel

                # eprint(c.normal_vec)
                eprint('CCC relV=%s, norm=%s, dot=%s, len=%d, o_pos=%s' % (relative_vel, c.normal_vec, np.dot(relative_vel, c.normal_vec), len(collisions), c.obs_pos))
                # eprint('np.dot ', np.dot(relative_vel, c.normal_vec))
                remove = np.dot(-relative_vel, c.normal_vec) + c.dist/dt
                eprint('remove=%f, dist=%f, vvvel=%f' % (remove, c.dist, c.dist/dt))

                if remove < 0 :
                    mark = True

                    # impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * np.dot(c.relative_vel, c.normal_vec)) / \
                    #         (1/c.body1.mass)

                    impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * remove) / \
                            (1/c.body1.mass)

                    # impulse = (remove) / \
                            # (1/c.body1.mass)

                    # eprint('body BEFORE pos=%s, vel=%s' % (c.body1.ptpos, c.body1.vel))
                    # eprint('prev 1', c.body1.prev_ptpos)

                    # c.body1.vel -= (c.normal_vec / c.body1.mass) * impulse
                    # c.body1.ptpos += c.body1.vel * dt

                    c.body1.vel += (c.normal_vec / c.body1.mass) * impulse
                    c.body1.ptpos -= c.body1.vel * dt



                    # eprint('vel', c.body1.vel)
                    # eprint('normal', c.normal_vec)
                    eprint('body AFTER pos=%s, vel=%s' % (c.body1.ptpos, c.body1.vel))
                    # time.sleep(100)


        # if mark:
        #     exit()


def test_converters():
    """
    For DEBUG.
    CHeck if converters work properly.
    """
    assert(np.all(Vector(x=50, y=38) == Vector(x=50, y=38)))
    arrpos = ptpos_to_arrpos(Vector(x=50, y=38))
    assert(np.all(Vector(x=50, y=38) == arrpos_to_ptpos(arrpos.x, arrpos.y)))

    ptpos = Vector(x=34.0, y=46.25706000000003)
    arrpos = ptpos_to_arrpos(ptpos)
    assert np.all(Vector(x=34, y=46) == arrpos_to_ptpos(arrpos.x, arrpos.y)), arrpos_to_ptpos(arrpos.x, arrpos.y)


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    setup_stderr()
    curses.wrapper(main)
