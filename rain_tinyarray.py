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
import tinyarray as ta



DEBUG_MODE = False
REFRESH_RATE = 100
EMPTY_BRAILLE = u'\u2800'
BUF_CELL_SIZE = ta.array([4, 2])
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
    # terrain.add_array(norm_arr, buf_shift=ta.array([0, 40]))
    # screen.add_common_array(norm_arr, buf_shift=ta.array([0, 40]))
    # screen.add_ascii_array(ascii_arr, buf_shift=ta.array([0, 40]))

    bodies = create_bodies(cout=50)

    t = 0
    dt = 1/REFRESH_RATE

    while t < 3:
        screen.restore_bg_buffer()
        step_simulation(dt, bodies, terrain)
        for body in bodies:
            screen.draw_point(body.pos)
        screen.refresh()

        # time.sleep(dt)
        t += dt

    curses.endwin()


def create_bodies(cout):
    # bodies = [
    #     Body(name=1, pos=ta.array([80.0, 34]), mass=1, velocity=ta.array([-40.0, 0])),
    #     Body(name=1, pos=ta.array([80.0, 50]), mass=1, velocity=ta.array([-40.0, 0])),
    #     Body(name=1, pos=ta.array([80.0, 112]), mass=1, velocity=ta.array([-40.0, 0])),
    #     Body(name=1, pos=ta.array([70.0, 110.5]), mass=1, velocity=ta.array([-40.0, 0])),
    #     Body(name=1, pos=ta.array([80.0, 110]), mass=1, velocity=ta.array([-40.0, 0])),
    #     Body(name=1, pos=ta.array([80.0, 23]), mass=1, velocity=ta.array([-40.0, 0])),
    #     Body(name=1, pos=ta.array([80.0, 22]), mass=1, velocity=ta.array([-40.0, 0])),
    #     Body(name=1, pos=ta.array([80.0, 21]), mass=1, velocity=ta.array([-40.0, 0])),
    #     Body(name=1, pos=ta.array([80.0, 20]), mass=1, velocity=ta.array([-40.0, 0])),
    #     Body(name=1, pos=ta.array([1.0, 110]), mass=1, velocity=ta.array([0.0, 1])),
    #     Body(name=1, pos=ta.array([1.0, 116]), mass=1, velocity=ta.array([0.0, 0])),
    # ]

    # for idx, body in enumerate(bodies):
    #     body._id = idx

    random.seed(3300)

    eprint()

    size = ta.array([curses.LINES*BUF_CELL_SIZE[0],
                    (curses.COLS-1)*BUF_CELL_SIZE[1]])

    bodies = []
    visited = {}

    c = 0
    while c < cout:
        x, y = random.randint(1, size[1]), size[0] - (random.randint(2, 20) * 1.0)

        if (x, y) in visited:
            continue

        visited[(x,y)] = True
        bodies.append(Body(name=c, pos=ta.array([y, x]), mass=1, velocity=ta.array([-40.0, 0])))
        c += 1

    return bodies


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


def magnitude(vec):
    return math.sqrt(vec[0]**2 + vec[1]**2)


def unit(vec):
    mag = magnitude(vec)
    return vec/mag if mag else vec


class Screen:
    def __init__(self, scr, terrain):
        self._scr = scr
        self._terrain = terrain

        # Redundant
        self._bg_buf_size = ta.array([curses.LINES, curses.COLS-1])
        self._screen_size = self._bg_buf_size*BUF_CELL_SIZE

        self._bg_buf = self._create_empty_background()
        self._bg_buf_backup = np.copy(self._bg_buf)

    def _create_empty_background(self):
        """
        Create empty screen buffer filled with "empty braille" characters.
        """
        return np.full(shape=self._bg_buf_size, fill_value=EMPTY_BRAILLE)

    def add_ascii_array(self, ascii_arr, buf_shift=ta.array([0, 0])):
        """
        Add static element to screen buffer. By default array will be drawn in
        bottom left corner.
        """
        height, width = ascii_arr.shape
        for y, x in np.argwhere(ascii_arr != ' '):
            buf_pos = ta.array([self._bg_buf_size[0] - height + y, x]) + buf_shift
            self._bg_buf[buf_pos[0]][buf_pos[1]] = ascii_arr[y, x]

        self._save_bg_backup()

    def add_common_array(self, arr, buf_shift=ta.array([0, 0])):
        """For DEBUG
        Add static element to screen buffer. Every element in array will be
        represent as braille character. By default all arrays are drawn in
        bottom left corner.
        """
        arr_shift = buf_shift * BUF_CELL_SIZE
        height, width, _ = arr.shape
        for x, y in it.product(range(width), range(height)):
            if np.any(arr[y, x] != 0):
                arr_pos = ta.array([self._screen_size[0] - height  + y, x]) + arr_shift
                pos = arrpos_to_pos(arr_pos)
                self.draw_point(pos)

        self._save_bg_backup()

    def add_terrain_data(self):
        """For DEBUG
        Redraw terrain array.
        """
        height, width, _ = self._terrain._terrain.shape
        for x, y in it.product(range(width), range(height)):
            if np.any(self._terrain._terrain[y, x] != 0):
                arr_pos = ta.array([y, x])
                pos = arrpos_to_pos(arr_pos)
                self.draw_point(pos)

        self._save_bg_backup()

    def add_border(self):
        """For DEBUG
        Draw screen border in braille characters.
        """
        for x in range(self._screen_size[1]):
            self.draw_point(ta.array([0, x]))
            self.draw_point(ta.array([self._screen_size[0]-1, x]))

        for y in range(self._screen_size[0]):
            self.draw_point(ta.array([y, 0]))
            self.draw_point(ta.array([y, self._screen_size[1]-1]))

        self._save_bg_backup()

    def _save_bg_backup(self):
        """Backup screen buffer."""
        self._bg_buf_backup = np.copy(self._bg_buf)

    def draw_rect(self, tl_pos, br_pos):
        """For DEBUG
        Draw rectangle
        """
        for x in range(tl_pos[1], br_pos[1]):
            for y in range(br_pos[0], tl_pos[0]):
                self.draw_point(ta.array([y, x]))

    def draw_point(self, pos):
        """
        Draw (put in screen buffer) single point. If theres is any ASCII
        character in screen cell, function will replace this character to his
        braille representation and merge this single point.
        """
        # Don't draw point when they are out of the screen
        if not (0 <= pos[1] < self._screen_size[1] and 0 <= pos[0] < self._screen_size[0]):
            return

        buf_pos = pos_to_bufpos(pos)
        cell_box = self._terrain.cut_bufcell_box(buf_pos)
        if ord(self._bg_buf[buf_pos[0]][buf_pos[1]]) < ord(EMPTY_BRAILLE) and np.any(cell_box):
            uchar = self._cell_box_to_uchar(cell_box)
        else:
            uchar = ord(self._bg_buf[buf_pos[0]][buf_pos[1]])

        self._bg_buf[buf_pos[0]][buf_pos[1]] = chr(uchar | self._pos_to_braille(pos))

    def _cell_box_to_uchar(self, cell_box):
        """
        Convert BUF_CELL_SIZE cell_box to his braille character representation.
        """
        height, width = cell_box.shape
        uchar = ord(EMPTY_BRAILLE)
        for y, x in np.argwhere(cell_box):
            uchar |= self._pos_to_braille(ta.array([BUF_CELL_SIZE[0]-1-y, x]))

        return uchar

    def _pos_to_braille(self, pos):
        """Point position as braille character in BUF_CELL."""
        by, bx = ta.floor(pos) % BUF_CELL_SIZE
        by = int(by)
        bx = int(bx)

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

    def restore_bg_buffer(self):
        """Restore static elements added to screen."""
        self._bg_buf = np.copy(self._bg_buf_backup)

    def refresh(self):
        """Draw buffer content to screen."""
        dtype = np.dtype('U' + str(self._bg_buf_size[1]))
        for num, line in enumerate(self._bg_buf):
            self._scr.addstr(num, 0, line.view(dtype)[0])
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
        self._bg_buf_size = ta.array([curses.LINES, curses.COLS-1])
        self._checked_pairs = {}
        self._create_bufpos_map(bodies)

    def _create_bufpos_map(self, bodies):
        self._map = defaultdict(list)
        for body in bodies:
            buf_pos = pos_to_bufpos(body.pos)
            self._map[self._bufpos_hash(buf_pos)].append(body)

    def neighbors(self, body):
        # Body can't collide with itself, so mark pair as checked
        pair_key = self._body_pair_hash(body, body)
        self._checked_pairs[pair_key] = True

        result = []
        range_x, range_y = self._bounding_box(body)
        for x, y in it.product(range_x, range_y):
            bufpos_key = self._bufpos_hash(ta.array([y, x]))

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
        return buf_pos[0] * self._bg_buf_size[1] + buf_pos[1]

    def _bounding_box(self, body):
        direction = unit((body.pos - body.prev_pos)) * 2 * Body.RADIUS
        # eprint('PPP', body.prev_pos, body.pos, body.pos + direction)
        buf_pos = pos_to_bufpos(body.prev_pos)
        buf_prev_pos = pos_to_bufpos(body.pos + direction)

        x1, x2 = min(buf_pos[1], buf_prev_pos[1]), max(buf_pos[1], buf_prev_pos[1])
        y1, y2 = min(buf_pos[0], buf_prev_pos[0]), max(buf_pos[0], buf_prev_pos[0])
        return range(x1, x2+1), range(y1, y2+1)


class Terrain:
    EMPTY = np.array([0, 0])

    def __init__(self):
        # Redundant
        self._terrain_size = ta.array([curses.LINES*BUF_CELL_SIZE[0],
                                      (curses.COLS-1)*BUF_CELL_SIZE[1]])
        self._terrain = np.zeros(shape=(self._terrain_size[0],
                                        self._terrain_size[1], NORM_VEC_DIM))
        self._red = np.logical_or.reduce(self._terrain!=Terrain.EMPTY, axis=-1)

    def add_array(self, arr, buf_shift=ta.array([0, 0])):
        """By default all arrays are drawn in bottom left corner."""
        arr_size = ta.array(arr.shape[:2])
        arr_shift = bufpos_to_arrpos(buf_shift)

        x1 = arr_shift[1]
        x2 = x1 + arr_size[1]
        y1 = self._terrain_size[0] - arr_size[0] - arr_shift[0]
        y2 = self._terrain_size[0] - arr_shift[0]
        self._terrain[y1:y2, x1:x2] = arr

        self._red = np.logical_or.reduce(self._terrain!=Terrain.EMPTY, axis=-1)

    def cut_bufcell_box(self, buf_pos):
        arr_pos = bufpos_to_arrpos(buf_pos)

        cell_box = self._red[arr_pos[0]:arr_pos[0]+BUF_CELL_SIZE[0],
                             arr_pos[1]:arr_pos[1]+BUF_CELL_SIZE[1]]
        return cell_box

    def obstacles(self, pos, prev_pos):
        arr_tl, arr_br = self._bounding_box(pos, prev_pos)
        box = self._cut_normal_vec_box(arr_tl, arr_br)
        box_markers = np.logical_or.reduce(box!=Terrain.EMPTY, axis=-1)

        result = []
        height, width = box_markers.shape
        for y, x in it.product(range(height), range(width)):
            if box_markers[y, x]:
                normal_vec = box[y, x]

                global_pos = arrpos_to_pos(arr_tl + (y, x))
                result.append((global_pos, normal_vec))

        return result

    def _bounding_box(self, pos, prev_pos):
        """
        Return top-left, bottom-right position of bounding box.
        """
        arr_pos = pos_to_arrpos(pos)
        arr_prev_pos = pos_to_arrpos(prev_pos)

        x1, x2 = min(arr_pos[1], arr_prev_pos[1]), max(arr_pos[1], arr_prev_pos[1])
        y1, y2 = min(arr_pos[0], arr_prev_pos[0]), max(arr_pos[0], arr_prev_pos[0])
        return ta.array([y1-1, x1-1]), ta.array([y2+2, x2+2])

    def _cut_normal_vec_box(self, arr_tl, arr_br):
        # Fit array corner coordinates to not go out-of-bounds
        tl = ta.array([max(arr_tl[0], 0), max(arr_tl[1], 0)])
        br = ta.array([min(arr_br[0], self._terrain_size[0]),
                       min(arr_br[1], self._terrain_size[1])])
        # Cut normal vectors from terrain array
        box = self._terrain[tl[0]:br[0], tl[1]:br[1]]

        # If bounding box is out of terrain bounds, we need to add border padding
        expected_shape = (arr_br[0] - arr_tl[0], arr_br[1] - arr_tl[1], NORM_VEC_DIM)
        if expected_shape == box.shape:
            return box

        # Pad borders. If bounding box top-left corner is out of bound,
        # we need also create shit for terrain top-left corner position. It
        # will be needed to calculate distance.
        if arr_tl[1] < 0:
            box = np.hstack((np.full(shape=(box.shape[0], 1, NORM_VEC_DIM), fill_value=ta.array([0, 1])), box))
        elif arr_br[1] > self._terrain_size[1]:
            box = np.hstack((box, np.full(shape=(box.shape[0], 1, NORM_VEC_DIM), fill_value=ta.array([0, -1]))))

        if arr_tl[0] < 0:
            box = np.vstack((np.full(shape=(1, box.shape[1], NORM_VEC_DIM), fill_value=ta.array([-1, 0])), box))
        elif arr_br[0] > self._terrain_size[0]:
            box = np.vstack((box, np.full(shape=(1, box.shape[1], NORM_VEC_DIM), fill_value=ta.array([1, 0]))))

        # Fix corners position, normal vector should guide to center of screen
        # value = ±√(1² + 1²) = ±0.7071
        if arr_tl[1] < 0 and arr_tl[0] < 0:
            box[0, 0] = ta.array([-0.7071, 0.7071])
        elif arr_tl[1] < 0 and arr_br[0] > self._terrain_size[0]:
            box[-1, 0] = ta.array([0.7071, 0.7071])
        elif arr_br[1] > self._terrain_size[1] and arr_tl[0] < 0:
            box[0, -1] = ta.array([-0.7071, -0.7071])
        elif arr_br[1] > self._terrain_size[1] and arr_br[0] > self._terrain_size[0]:
            box[0, -1] = ta.array([0.7071, -0.7071])

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
        if norm_arr.shape[1] % BUF_CELL_SIZE[1] or norm_arr.shape[0] % BUF_CELL_SIZE[0]:
            raise Exception("Arrays with normal vector can't be transformed to buffer")

        ascii_markers = self._reduce_norm(norm_arr)
        del_rows = [list(range(idx*BUF_CELL_SIZE[0], idx*BUF_CELL_SIZE[0]+BUF_CELL_SIZE[0]))
                    for idx, margin in enumerate(np.all(ascii_markers==False, axis=1)) if margin]
        norm_arr = np.delete(norm_arr, del_rows, axis=0)

        del_columns = [list(range(idx*BUF_CELL_SIZE[1], idx*BUF_CELL_SIZE[1]+BUF_CELL_SIZE[1]))
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
        for y in range(0, marker_arr.shape[0], BUF_CELL_SIZE[0]):
            for x in range(0, marker_arr.shape[1], BUF_CELL_SIZE[1]):
                result.append(np.any(marker_arr[y:y+BUF_CELL_SIZE[0], x:x+BUF_CELL_SIZE[1]]))

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
        ascii_arr_size = ta.array(ascii_arr.shape)
        norm_arr_size = norm_arr.shape[:2] // BUF_CELL_SIZE

        if np.any(ascii_arr_size != norm_arr_size):
            raise Exception('Imported arrays (ascii/norm) - mismatch size', ascii_arr_size, norm_arr_size)

        eprint('Validation OK')


def pos_to_bufpos(pos):
    x = math.floor(pos[1]/BUF_CELL_SIZE[1])
    y = curses.LINES - 1 - math.floor(pos[0]/BUF_CELL_SIZE[0])
    return ta.array([y, x])


def bufpos_to_arrpos(buf_pos):
    """
    Return top-left corner to buffer cell in array coordinates (Y from top to
    bottom).
    """
    return ta.array([buf_pos[0]*BUF_CELL_SIZE[0], buf_pos[1]*BUF_CELL_SIZE[1]])


def arrpos_to_pos(arr_pos):
    """Array position to Cartesian coordinate system"""
    return ta.array([curses.LINES * BUF_CELL_SIZE[0] - 1 - arr_pos[0], arr_pos[1]])


def pos_to_arrpos(pos):
    """
    Point position (in Cartesian coordinate system) to array position (Y from
    top to bottom).
    """
    y = curses.LINES * BUF_CELL_SIZE[0] - 1 - math.floor(pos[0])
    return ta.array([y, math.floor(pos[1])])


def test_converters():
    """
    For DEBUG.
    Check if converters work properly.
    """
    assert(np.all(ta.array([38, 50]) == ta.array([38, 50])))

    arr_pos = pos_to_arrpos(ta.array([38, 50]))
    assert(np.all(ta.array([38, 50]) == arrpos_to_pos(arr_pos)))

    arr_pos = pos_to_arrpos(ta.array([46.25706000000003, 34.0]))
    assert np.all(ta.array([46, 34]) == arrpos_to_pos(arr_pos)), arrpos_to_pos(arr_pos)


def step_simulation(dt, bodies, terrain):
    calc_forces(dt, bodies)
    integrate(dt, bodies)
    collisions = detect_collisions(bodies, terrain)
    resolve_collisions(dt, collisions)


def calc_forces(dt, bodies):
    for body in bodies:
        body.forces = ta.array([-GRAVITY_ACC, 0.0]) * body.mass

        # TODO: replace by better solution
        # if math.floor(body.prev_pos[0]) == 0 and math.floor(body.pos[0]) == 0:
        #     body.forces *= COEFFICIENT_OF_FRICTION


def integrate(dt, bodies):
    for body in bodies:
        body.prev_pos = ta.array(body.pos)
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
        # collisions += bodies_collisions(body1, body2)

    return collisions


def obstacle_collisions(body, terrain):
    result = []

    for obstacle_pos, normal_vec in terrain.obstacles(body.pos, body.prev_pos):
        pos = ta.floor(obstacle_pos) + ta.array([Body.RADIUS, Body.RADIUS])
        dist = magnitude((pos - body.pos)) - 2*Body.RADIUS

        collision = Collision(body1=body,
                              body2=None,
                              dist=dist,
                              normal_vec=normal_vec)

        result.append(collision)

    return result


def bodies_collisions(body1, body2):
    result = []

    dist = body2.pos - body1.pos
    normal_vec = -unit(dist)
    real_dist = magnitude(dist) - 2*Body.RADIUS
    collision = Collision(body1=body1,
                          body2=body2,
                          dist=real_dist,
                          normal_vec=normal_vec)
    # eprint(real_dist, body1._id, body2._id)
    result.append(collision)

    return result


def bodies_collisions2(body, neighb):
    result = []

    for neigh_body in neighb.neighbors(body):
        dist = body.pos - neigh_body.pos
        normal_vec = unit(dist)
        real_dist = magnitude(dist) - 2*Body.RADIUS
        # eprint(body.pos, neigh_body.pos)
        # eprint(real_dist, neigh_body._id, body._id)
        collision = Collision(body1=body,
                              body2=neigh_body,
                              dist=real_dist,
                              normal_vec=normal_vec)

        result.append(collision)

    # eprint(len(result))
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
            # eassert(False)
            relative_vel = c.body2.vel - c.body1.vel
            remove = np.dot(relative_vel, c.normal_vec) - c.dist/dt

            if remove < 0:
                continue

            # eprint('IMPACT')
            impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * remove) / \
                    (1/c.body1.mass  + 1/c.body2.mass)
            c.body1.vel -= (c.normal_vec / c.body1.mass) * impulse
            c.body2.vel += (c.normal_vec / c.body2.mass) * impulse


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    setup_stderr()
    curses.wrapper(main)
