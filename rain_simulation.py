#! /usr/bin/env python3

"""
Coordinates systems:
    pos         - position in Cartesian coordinate system
    buf_pos     - position on screen (of one character). Y from top to bottom
    arr_pos     - similar to pos, but Y from top to bottom
"""


import sys
import itertools as it
from collections import defaultdict
import math
import time
import random
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


DEBUG_MODE = True
REFRESH_RATE = 100
EMPTY_BRAILLE = u'\u2800'
BUF_CELL_SIZE = Size(4, 2)
NORM_VEC_DIM = 2
NUM_ITERATION = 3

# Physical values
GRAVITY_ACC = 9.8  # [m/s^2]
COEFFICIENT_OF_RESTITUTION = 0.5
COEFFICIENT_OF_FRICTION = 0.9


def main(scr):
    test_converters()
    setup_curses(scr)
    terrain = Terrain()
    screen = Screen(scr, terrain)

    im = Importer()
    ascii_arr, norm_arr = im.load('ascii_fig.txt', 'ascii_fig.png.norm')

    terrain.add_array(norm_arr)
    screen.add_ascii_array(ascii_arr)
    # screen.add_terrain_data()
    # terrain.add_array(norm_arr, buf_shift=Vector(x=40, y=0))
    # screen.add_common_array(norm_arr, buf_shift=Vector(x=40, y=0))
    # screen.add_ascii_array(ascii_arr, buf_shift=Vector(x=40, y=0))

    bodies = [
        # Body(name=1, pos=Vector(x=34, y=80.0), mass=1, velocity=Vector(x=0, y=-40.0)),
        # Body(name=2, pos=Vector(x=50, y=80.0), mass=1, velocity=Vector(x=0, y=-40.0)),
        # Body(name=3, pos=Vector(x=112, y=80.0), mass=1, velocity=Vector(x=0, y=-40.0)),
        # Body(name=4, pos=Vector(x=110.5, y=70.0), mass=1, velocity=Vector(x=0, y=-40.0)),
        # Body(name=5, pos=Vector(x=110, y=80.0), mass=1, velocity=Vector(x=0, y=-40.0)),
        # Body(name=6, pos=Vector(x=23, y=80.0), mass=1, velocity=Vector(x=0, y=-40.0)),
        # Body(name=7, pos=Vector(x=22, y=80.0), mass=1, velocity=Vector(x=0, y=-40.0)),
        # Body(name=8, pos=Vector(x=21, y=80.0), mass=1, velocity=Vector(x=0, y=-40.0)),
        # Body(name=9, pos=Vector(x=20, y=80.0), mass=1, velocity=Vector(x=0, y=-40.0)),
        Body(name=10, pos=Vector(x=110, y=1.0), mass=1, velocity=Vector(x=1, y=0.0)),
        Body(name=11, pos=Vector(x=116, y=1.0), mass=1, velocity=Vector(x=0, y=0.0)),
    ]

    t = 0
    dt = 1/REFRESH_RATE

    while True:
        screen.restore_buffer()

        step_simulation(dt, bodies, terrain)

        for body in bodies:
            screen.draw_point(body.pos)
        screen.refresh()

        # time.sleep(dt)
        t += dt

    curses.endwin()


def setup_stderr():
    """
    Redirect stderr to other terminal. Run tty command, to get terminal id.
    """
    if DEBUG_MODE:
        sys.stderr = open('/dev/pts/3', 'w')


def eprint(*args, **kwargs):
    """Print on stderr"""
    if DEBUG_MODE:
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

    def __str__(self):
        """string representation of object."""
        # return "Vector(x=" + str(self.x) + ", y=" + str(self.y) + ")"
        return "Vector(x=%.4f, y=%.4f)" % (self.x, self.y)

    def __repr__(self):
        """string representation of object."""
        # return "Vector(x=" + str(self.x) + ", y=" + str(self.y) + ")"
        return "Vector(x=%.4f, y=%.4f)" % (self.x, self.y)

    def magnitude(self):
        """Calculate vector magnitude."""
        return np.linalg.norm(self)

    def unit(self):
        """Calculate unit vector."""
        mag = self.magnitude()
        return self/mag if mag else self


class Screen:
    def __init__(self, scr, terrain):
        self._scr = scr
        self._terrain = terrain

        # Redundant
        self._buf_size = Size(curses.LINES, curses.COLS-1)
        self._screen_size = self._buf_size*BUF_CELL_SIZE

        self._buf = self._create_empty_buf()
        self._buf_backup = np.copy(self._buf)

    def _create_empty_buf(self):
        """
        Create empty screen buffer filled with "empty braille" characters.
        """
        return np.full(shape=self._buf_size, fill_value=EMPTY_BRAILLE)

    def add_ascii_array(self, ascii_arr, buf_shift=Vector(x=0, y=0)):
        """
        Add static element to screen buffer. By default array will be drawn in
        bottom left corner.
        """
        height, width = ascii_arr.shape
        for y, x in np.argwhere(ascii_arr != ' '):
            buf_pos = Vector(x=x, y=self._buf_size.height - height + y) + buf_shift
            self._buf[buf_pos.y][buf_pos.x] = ascii_arr[y, x]

        self._save_in_backup_buf()

    def add_common_array(self, arr, buf_shift=Vector(x=0, y=0)):
        """For DEBUG
        Add static element to screen buffer. Every element in array will be
        represent as braille character. By default all arrays are drawn in
        bottom left corner.
        """
        arr_shift = buf_shift * BUF_CELL_SIZE
        height, width, _ = arr.shape
        for x, y in it.product(range(width), range(height)):
            if np.any(arr[y, x] != 0):
                arr_pos = Vector(x=x, y=self._screen_size.height - height  + y) + arr_shift
                pos = arrpos_to_pos(arr_pos)
                self.draw_point(pos)

        self._save_in_backup_buf()

    def add_terrain_data(self):
        """For DEBUG
        Redraw terrain array.
        """
        height, width, _ = self._terrain._terrain.shape
        for x, y in it.product(range(width), range(height)):
            if np.any(self._terrain._terrain[y, x] != 0):
                arr_pos = Vector(x=x, y=y)
                pos = arrpos_to_pos(arr_pos)
                self.draw_point(pos)

        self._save_in_backup_buf()

    def add_border(self):
        """For DEBUG
        Draw screen border in braille characters.
        """
        for x in range(self._screen_size.width):
            self.draw_point(Vector(x=x, y=0))
            self.draw_point(Vector(x=x, y=self._screen_size.height-1))

        for y in range(self._screen_size.height):
            self.draw_point(Vector(x=0, y=y))
            self.draw_point(Vector(x=self._screen_size.width-1, y=y))

        self._save_in_backup_buf()

    def _save_in_backup_buf(self):
        """Backup screen buffer."""
        self._buf_backup = np.copy(self._buf)

    def draw_rect(self, tl_pos, br_pos):
        """For DEBUG
        Draw rectangle
        """
        for x in range(tl_pos.x, br_pos.x):
            for y in range(br_pos.y, tl_pos.y):
                self.draw_point(Vector(x=x, y=y))

    def draw_point(self, pos):
        """
        Draw (put in screen buffer) single point. If theres is any ASCII
        character in screen cell, function will replace this character to his
        braille representation and merge this single point.
        """
        # Don't draw point when they are out of the screen
        if not (0 <= pos.x < self._screen_size.width and 0 <= pos.y < self._screen_size.height):
            return

        buf_pos = pos_to_bufpos(pos)
        cell_box = self._terrain.cut_bufcell_box(buf_pos)
        if ord(self._buf[buf_pos.y][buf_pos.x]) < ord(EMPTY_BRAILLE) and np.any(cell_box):
            uchar = self._cell_box_to_uchar(cell_box)
        else:
            uchar = ord(self._buf[buf_pos.y][buf_pos.x])

        self._buf[buf_pos.y][buf_pos.x] = chr(uchar | self._pos_to_braille(pos))

    def _cell_box_to_uchar(self, cell_box):
        """
        Convert BUF_CELL_SIZE cell_box to his braille character representation.
        """
        height, width = cell_box.shape
        uchar = ord(EMPTY_BRAILLE)
        for y, x in np.argwhere(cell_box):
            uchar |= self._pos_to_braille(Vector(x=x, y=BUF_CELL_SIZE.height-1-y))

        return uchar

    def _pos_to_braille(self, pos):
        """Point position as braille character in BUF_CELL."""
        by, bx = (np.floor(pos) % BUF_CELL_SIZE).astype(int)

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
        self._buf = np.copy(self._buf_backup)

    def refresh(self):
        """Draw buffer content to screen."""
        for num, line in enumerate(self._buf):
            self._scr.addstr(num, 0, ''.join(line))
        self._scr.refresh()


class Body:
    RADIUS = 0.5

    def __init__(self, name, pos, mass, velocity):
        self.pos = pos
        self.prev_pos = pos
        self.mass = mass
        self.vel = velocity
        self._id = name

    def __hash__(self):
        return self._id

    def __str__(self):
        """string representation of object."""
        return "Body(%d)" % self._id

    def __repr__(self):
        """string representation of object."""
        return "Body(%d)" % self._id


class Neighborhood:
    def __init__(self, bodies):
        self._buf_size = Size(curses.LINES, curses.COLS-1)
        self._checked_pairs = {}
        self._create_bufpos_map(bodies)

    def _create_bufpos_map(self, bodies):
        self._map = defaultdict(list)
        for body in bodies:
            buf_pos = pos_to_bufpos(body.pos)
            self._map[self._bufpos_hash(buf_pos)].append(body)

    def neighbors(self, body):
        # eassert(False)
        # Body can't collide with itself, so mark pair as checked
        pair_key = self._body_pair_hash(body, body)
        self._checked_pairs[pair_key] = True

        result = []
        range_x, range_y = self._bounding_box(body)
        for x, y in it.product(range_x, range_y):
            bufpos_key = self._bufpos_hash(Vector(x=x, y=y))

            # Check all neighbors bodies from nearby buf cell
            for neigh_body in self._map[bufpos_key]:
                pair_key = self._body_pair_hash(body, neigh_body)

                # If body pairs was already checked do nothing. We can have
                # only one such collision
                if pair_key in self._checked_pairs:
                    continue

                self._checked_pairs[pair_key] = True
                result.append(neigh_body)

        return result

    def _body_pair_hash(self, body1, body2):
        if hash(body1) < hash(body2):
            return (hash(body1), hash(body2))
        return (hash(body2), hash(body1))

    def _bufpos_hash(self, pos):
        buf_pos = pos_to_bufpos(pos)
        return buf_pos.y * self._buf_size.width + buf_pos.x

    def _bounding_box(self, body):
        direction = body.prev_pos - body.pos
        buf_pos = pos_to_bufpos(body.prev_pos)
        buf_prev_pos = pos_to_bufpos(body.pos + direction)

        x1, x2 = min(buf_pos.x, buf_prev_pos.x), max(buf_pos.x, buf_prev_pos.x)
        y1, y2 = min(buf_pos.y, buf_prev_pos.y), max(buf_pos.y, buf_prev_pos.y)
        return range(x1, x2+1), range(y1, y2+1)


class Terrain:
    EMPTY = np.array([0, 0])

    def __init__(self):
        # Redundant
        self._terrain_size = Size(curses.LINES*BUF_CELL_SIZE.height,
                                  (curses.COLS-1)*BUF_CELL_SIZE.width)
        self._terrain = np.zeros(shape=(self._terrain_size.height,
                                        self._terrain_size.width, NORM_VEC_DIM))

    def add_array(self, arr, buf_shift=Vector(x=0, y=0)):
        """By default all arrays are drawn in bottom left corner."""
        arr_size = Size(*arr.shape[:2])
        arr_shift = bufpos_to_arrpos(buf_shift)

        x1 = arr_shift.x
        x2 = x1 + arr_size.width
        y1 = self._terrain_size.height - arr_size.height - arr_shift.y
        y2 = self._terrain_size.height - arr_shift.y
        self._terrain[y1:y2, x1:x2] = arr

    def cut_bufcell_box(self, buf_pos):
        arr_pos = bufpos_to_arrpos(buf_pos)

        cell_box = self._terrain[arr_pos.y:arr_pos.y+BUF_CELL_SIZE.height,
                                 arr_pos.x:arr_pos.x+BUF_CELL_SIZE.width]

        cell_box = np.logical_or.reduce(cell_box != Terrain.EMPTY, axis=-1)
        return cell_box

    def obstacles(self, pos, prev_pos):
        arr_tl, arr_br = self._bounding_box(pos, prev_pos)
        box = self._cut_normal_vec_box(arr_tl, arr_br)
        box_markers = np.logical_or.reduce(box != Terrain.EMPTY, axis=-1)

        result = []
        for y, x in np.argwhere(box_markers):
            box_obs_pos = Vector(x=x, y=y)
            normal_vec = box[box_obs_pos.y, box_obs_pos.x]

            global_pos = arrpos_to_pos(arr_tl + box_obs_pos)
            result.append((global_pos, normal_vec))

        return result

    def _bounding_box(self, pos, prev_pos):
        """
        Return top-left, bottom-right position of bounding box.
        """
        arr_pos = pos_to_arrpos(pos)
        arr_prev_pos = pos_to_arrpos(prev_pos)

        x1, x2 = min(arr_pos.x, arr_prev_pos.x), max(arr_pos.x, arr_prev_pos.x)
        y1, y2 = min(arr_pos.y, arr_prev_pos.y), max(arr_pos.y, arr_prev_pos.y)
        return Vector(x=x1-1, y=y1-1), Vector(x=x2+2, y=y2+2)

    def _cut_normal_vec_box(self, arr_tl, arr_br):
        # Fit array corner coordinates to not go out-of-bounds
        tl = Vector(x=max(arr_tl.x, 0), y=max(arr_tl.y, 0))
        br = Vector(x=min(arr_br.x, self._terrain_size.width),
                    y=min(arr_br.y, self._terrain_size.height))
        # Cut normal vectors from terrain array
        box = self._terrain[tl.y:br.y, tl.x:br.x]

        # If bounding box is out of terrain bounds, we need to add border padding
        expected_shape = (arr_br.y - arr_tl.y, arr_br.x - arr_tl.x, NORM_VEC_DIM)
        if expected_shape == box.shape:
            return box

        # Pad borders. If bounding box top-left corner is out of bound,
        # we need also create shit for terrain top-left corner position. It
        # will be needed to calculate distance.
        if arr_tl.x < 0:
            box = np.hstack((np.full(shape=(box.shape[0], 1, NORM_VEC_DIM), fill_value=Vector(x=1, y=0)), box))
        elif arr_br.x > self._terrain_size.width:
            box = np.hstack((box, np.full(shape=(box.shape[0], 1, NORM_VEC_DIM), fill_value=Vector(x=-1, y=0))))

        if arr_tl.y < 0:
            box = np.vstack((np.full(shape=(1, box.shape[1], NORM_VEC_DIM), fill_value=Vector(x=0, y=-1)), box))
        elif arr_br.y > self._terrain_size.height:
            box = np.vstack((box, np.full(shape=(1, box.shape[1], NORM_VEC_DIM), fill_value=Vector(x=0, y=1))))

        # Fix corners position, normal vector should guide to center of screen
        # value = ±√(1² + 1²) = ±0.7071
        if arr_tl.x < 0 and arr_tl.y < 0:
            box[0, 0] = Vector(x=0.7071, y=-0.7071)
        elif arr_tl.x < 0 and arr_br.y > self._terrain_size.height:
            box[-1, 0] = Vector(x=0.7071, y=0.7071)
        elif arr_br.x > self._terrain_size.width and arr_tl.y < 0:
            box[0, -1] = Vector(x=-0.7071, y=-0.7071)
        elif arr_br.x > self._terrain_size.width and arr_br.y > self._terrain_size.height:
            box[0, -1] = Vector(x=-0.7071, y=0.7071)

        return box


class Importer:
    def load(self, ascii_file, normal_vec_file):
        ascii_arr = self._import_ascii_arr(ascii_file)
        ascii_arr = self._remove_ascii_marker(ascii_arr)
        ascii_arr = self._remove_ascii_margin(ascii_arr)

        norm_arr = self._import_norm_arr(normal_vec_file)
        norm_arr = self._remove_norm_margin(norm_arr)

        self._reduce_norm(norm_arr)
        self._print_ascii_markers(norm_arr)

        self._validate_arrays(ascii_arr, norm_arr)
        return ascii_arr, norm_arr

    def _import_ascii_arr(self, ascii_file):
        """Import ASCII figure from file."""
        ascii_fig = []
        with open(ascii_file, 'r') as f:
            for line in f:
                arr = np.array([ch for ch in line if ch != '\n'])
                ascii_fig.append(arr)

        ascii_arr = self._reshape_ascii(ascii_fig)

        return ascii_arr

    def _reshape_ascii(self, ascii_fig):
        """
        Fill end of each line in ascii_fig with spaces, and convert it to np.array.
        """
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
        """Erase 3x3 marker at the left-top position from ASCII."""
        DIM = 3
        ascii_arr[0:DIM, 0:DIM] = np.array([' ' for _ in range(DIM*DIM)]).reshape(DIM, DIM)
        return ascii_arr

    def _remove_ascii_margin(self, ascii_arr):
        """
        Remove margin from ascii_arr (line and columns with spaces at the edges.
        """
        del_rows = [idx for idx, margin in enumerate(np.all(ascii_arr==' ', axis=0)) if margin]
        ascii_arr = np.delete(ascii_arr, del_rows, axis=1)

        del_columns = [idx for idx, margin in enumerate(np.all(ascii_arr==' ', axis=1)) if margin]
        ascii_arr = np.delete(ascii_arr, del_columns, axis=0)

        return ascii_arr

    def _import_norm_arr(self, normal_vec_file):
        """Import array with normal vector."""
        arr = np.loadtxt(normal_vec_file)
        height, width = arr.shape
        norm_arr = arr.reshape(height, width // NORM_VEC_DIM, NORM_VEC_DIM)

        return norm_arr

    def _remove_norm_margin(self, norm_arr):
        """
        Remove margin from array with normal vectors (line and columns with
        np.array([0, 0]) at the edges.
        """
        if norm_arr.shape[1] % BUF_CELL_SIZE.width or norm_arr.shape[0] % BUF_CELL_SIZE.height:
            raise Exception("Arrays with normal vector can't be transformed to buffer")

        ascii_markers = self._reduce_norm(norm_arr)
        del_rows = [list(range(idx*BUF_CELL_SIZE.height, idx*BUF_CELL_SIZE.height+BUF_CELL_SIZE.height))
                    for idx, margin in enumerate(np.all(ascii_markers==False, axis=1)) if margin]
        norm_arr = np.delete(norm_arr, del_rows, axis=0)

        del_columns = [list(range(idx*BUF_CELL_SIZE.width, idx*BUF_CELL_SIZE.width+BUF_CELL_SIZE.width))
                       for idx, margin in enumerate(np.all(ascii_markers==False, axis=0)) if margin]
        norm_arr = np.delete(norm_arr, del_columns, axis=1)

        return norm_arr

    def _reduce_norm(self, norm_arr):
        """
        Reduce array with normal vectors (for each braille characters), to
        "ASCII figure" size array, and mark if in "ASCII cell"/"BUF CELL" there
        was any character.
        """
        EMPTY_VEC = np.array([0, 0])
        marker_arr = np.logical_or.reduce(norm_arr!=EMPTY_VEC, axis=-1)

        result = []
        for y in range(0, marker_arr.shape[0], BUF_CELL_SIZE.height):
            for x in range(0, marker_arr.shape[1], BUF_CELL_SIZE.width):
                result.append(np.any(marker_arr[y:y+BUF_CELL_SIZE.height, x:x+BUF_CELL_SIZE.width]))

        size = marker_arr.shape // BUF_CELL_SIZE
        result = np.reshape(result, size)

        return result

    def _print_ascii_markers(self, norm_arr):
        """
        Print ASCII markers for cells in array with normal vectors.
        """
        ascii_markers = self._reduce_norm(norm_arr)
        eprint(ascii_markers.astype(int))

    def _validate_arrays(self, ascii_arr, norm_arr):
        """Validate if both arrays describe same thing."""
        ascii_arr_size = Size(*ascii_arr.shape)
        norm_arr_size = norm_arr.shape[:2] // BUF_CELL_SIZE

        if np.any(ascii_arr_size != norm_arr_size):
            raise Exception('Imported arrays (ascii/norm) - mismatch size', ascii_arr_size, norm_arr_size)

        eprint('Validation OK')


def pos_to_bufpos(pos):
    x = math.floor(pos.x/BUF_CELL_SIZE.width)
    y = curses.LINES - 1 - math.floor(pos.y/BUF_CELL_SIZE.height)
    return Vector(x=x, y=y)


def bufpos_to_arrpos(buf_pos):
    """
    Return top-left corner to buffer cell in array coordinates (Y from top to
    bottom).
    """
    return Vector(x=buf_pos.x*BUF_CELL_SIZE.width, y=buf_pos.y*BUF_CELL_SIZE.height)


def arrpos_to_pos(arr_pos):
    """Array position to Cartesian coordinate system"""
    return Vector(x=arr_pos.x, y=curses.LINES * BUF_CELL_SIZE.height - 1 - arr_pos.y)


def pos_to_arrpos(pos):
    """
    Point position (in Cartesian coordinate system) to array position (Y from
    top to bottom).
    """
    y = curses.LINES * BUF_CELL_SIZE.height - 1 - math.floor(pos.y)
    return Vector(x=math.floor(pos.x), y=y)


def test_converters():
    """
    For DEBUG.
    Check if converters work properly.
    """
    assert(np.all(Vector(x=50, y=38) == Vector(x=50, y=38)))

    arr_pos = pos_to_arrpos(Vector(x=50, y=38))
    assert(np.all(Vector(x=50, y=38) == arrpos_to_pos(arr_pos)))

    arr_pos = pos_to_arrpos(Vector(x=34.0, y=46.25706000000003))
    assert np.all(Vector(x=34, y=46) == arrpos_to_pos(arr_pos)), arrpos_to_pos(arr_pos)


def step_simulation(dt, bodies, terrain):
    calc_forces(dt, bodies)
    integrate(dt, bodies)
    collisions = detect_collisions(bodies, terrain)
    resolve_collisions(dt, collisions)


def calc_forces(dt, bodies):
    for body in bodies:
        body.forces = Vector(x=0.0, y=-GRAVITY_ACC) * body.mass

        # TODO: replace by better solution
        # if math.floor(body.prev_pos.y) == 0 and math.floor(body.pos.y) == 0:
        #     body.forces *= COEFFICIENT_OF_FRICTION


def integrate(dt, bodies):
    for body in bodies:
        body.prev_pos = Vector(*body.pos)
        body.acc = body.forces / body.mass
        body.vel += body.acc * dt
        body.pos += body.vel * dt


class Collision:
    def __init__(self, body1, body2, dist, normal_vec):
        self.body1 = body1
        self.body2 = body2
        self.dist = dist
        self.normal_vec = normal_vec


def detect_collisions(bodies, terrain):
    collisions = []
    neighb = Neighborhood(bodies)
    for body in bodies:
        collisions += obstacle_collisions(body, terrain)
        collisions += bodies_collisions2(body, neighb)
    # for body1, body2 in it.combinations(bodies, 2):
    #     collisions += bodies_collisions(body1, body2)

    return collisions


def obstacle_collisions(body, terrain):
    result = []

    for obstacle_pos, normal_vec in terrain.obstacles(body.pos, body.prev_pos):
        pos = np.floor(obstacle_pos) + Vector(x=Body.RADIUS, y=Body.RADIUS)
        dist = (pos - body.pos).magnitude() - 2*Body.RADIUS

        collision = Collision(body1=body,
                              body2=None,
                              dist=dist,
                              normal_vec=normal_vec)

        result.append(collision)

    return result


def bodies_collisions(body1, body2):
    result = []

    dist = body2.pos - body1.pos
    normal_vec = -dist.unit()
    real_dist = dist.magnitude() - 2*Body.RADIUS
    collision = Collision(body1=body1,
                          body2=body2,
                          dist=real_dist,
                          normal_vec=normal_vec)

    result.append(collision)

    return result


def bodies_collisions2(body, neighb):
    result = []

    for neigh_body in neighb.neighbors(body):
        dist = neigh_body.pos - body.pos
        normal_vec = -dist.unit()
        real_dist = dist.magnitude() - 2*Body.RADIUS
        collision = Collision(body1=body,
                              body2=neigh_body,
                              dist=real_dist,
                              normal_vec=normal_vec)

        result.append(collision)

    eprint(len(result))
    return result


def resolve_collisions(dt, collisions):
    """
    Speculative contacts solver.

    References:
    https://wildbunny.co.uk/blog/2011/03/25/speculative-contacts-an-continuous-collision-engine-approach-part-1/
    """
    for _, c in it.product(range(NUM_ITERATION), collisions):
        # Collision with screen borders
        if not c.body2:
            relative_vel = -c.body1.vel
            remove = np.dot(relative_vel, c.normal_vec) - c.dist/dt

            if remove < 0:
                continue

            impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * remove) / \
                    (1/c.body1.mass)
            c.body1.vel -= (c.normal_vec / c.body1.mass) * impulse
        else:
            relative_vel = c.body2.vel - c.body1.vel
            remove = np.dot(relative_vel, c.normal_vec) - c.dist/dt

            if remove < 0:
                continue

            impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * remove) / \
                    (1/c.body1.mass  + 1/c.body2.mass)
            c.body1.vel -= (c.normal_vec / c.body1.mass) * impulse
            c.body2.vel += (c.normal_vec / c.body2.mass) * impulse


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    setup_stderr()
    curses.wrapper(main)
