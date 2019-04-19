#! /usr/bin/env python3


"""
Coordinates systems:
    pos         - position in Cartesian coordinate system. Y from bottom to top,
                  point (0, 0) is in bottom left corner of the screen, All
                  physical calculation are performed in this system.
    scr_pos     - position on screen (of one character). Y from top to bottom,
                  point (0, 0) is in top-left corner of the screen.
    arr_pos     - position in numpy or tinyarray array. Y from top to bottom,
                  point (0, 0) is in top-felt corner of the screen.
"""


import sys
import itertools as it
from collections import defaultdict
import math
import curses
import pdb
import numpy as np
import tinyarray as ta


# Applications constants
TELEMETRY_MODE = False
EMPTY_BRAILLE = u'\u2800'
SCR_CELL_SHAPE = ta.array([4, 2])
NORM_VEC_DIM = 2
NUM_ITERATION = 3

# Physical constants
GRAVITY_ACC = 9.8  # [m/s^2]
COEFFICIENT_OF_RESTITUTION = 0.5


def setup(scr, enable=False, terminal='/dev/pts/1'):
    """Main setup function."""
    setup_curses(scr)
    setup_telemetry(enable, terminal)


class Screen:
    def __init__(self, scr, terrain):
        self._curses_scr = scr
        self._terrain = terrain

        bg_shape = (curses.LINES, curses.COLS-1)

        # background with "empty braille" characters.
        self._bg_buf = np.full(shape=bg_shape, fill_value=EMPTY_BRAILLE)
        self._bg_buf_backup = np.copy(self._bg_buf)

        self._dot_scr_shape = self._bg_buf.shape*SCR_CELL_SHAPE

    def add_ascii_array(self, ascii_arr, scr_shift=(0, 0)):
        """
        Add static element to screen buffer. By default array will be drawn in
        bottom left corner.
        """
        ascii_arr, scr_shift = adjust_array(ascii_arr, scr_shift)

        height, width = ascii_arr.shape
        for y, x in np.argwhere(ascii_arr!=' '):
            scr_pos = ta.array([self._bg_buf.shape[0] - height + y, x]) + scr_shift
            self._bg_buf[scr_pos[0], scr_pos[1]] = ascii_arr[y, x]

        self._save_background_backup()

    def add_common_array(self, arr, scr_shift=(0, 0)):
        """For DEBUG
        Add static element to screen buffer. Every element in array will be
        represent as braille character. By default all arrays are drawn in
        bottom left corner.
        """
        arr_shift = scr_shift * SCR_CELL_SHAPE
        arr, arr_shift = adjust_array(arr, arr_shift)

        height, width, _ = arr.shape
        for x, y in it.product(range(width), range(height)):
            if np.any(arr[y, x]):
                arr_pos = ta.array([self._dot_scr_shape[0] - height  + y, x]) + arr_shift
                pos = arr_to_phy(arr_pos)
                self.draw_point(pos)

        self._save_background_backup()

    def add_terrain_data(self):
        """For DEBUG
        Redraw terrain array.
        """
        height, width, _ = self._terrain._normal_vecs.shape
        for x, y in it.product(range(width), range(height)):
            if self._terrain._normal_marks[y, x]:
                arr_pos = ta.array([y, x])
                pos = arr_to_phy(arr_pos)
                self.draw_point(pos)

        self._save_background_backup()

    def add_border(self):
        """For DEBUG
        Draw screen border in braille characters.
        """
        for x in range(self._dot_scr_shape[1]):
            self.draw_point(ta.array([0, x]))
            self.draw_point(ta.array([self._dot_scr_shape[0]-1, x]))

        for y in range(self._dot_scr_shape[0]):
            self.draw_point(ta.array([y, 0]))
            self.draw_point(ta.array([y, self._dot_scr_shape[1]-1]))

        self._save_background_backup()

    def _save_background_backup(self):
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
        if not (0 <= pos[1] < self._dot_scr_shape[1] and 0 <= pos[0] < self._dot_scr_shape[0]):
            return

        scr_pos = phy_to_scr(pos)
        cell_box = self._terrain.cut_scrcell_box(scr_pos)
        if ord(self._bg_buf[scr_pos[0], scr_pos[1]]) < ord(EMPTY_BRAILLE) and np.any(cell_box):
            uchar = self._cell_box_to_uchar(cell_box)
        else:
            uchar = ord(self._bg_buf[scr_pos[0], scr_pos[1]])

        self._bg_buf[scr_pos[0], scr_pos[1]] = chr(uchar | self._pos_to_braille(pos))

    def _cell_box_to_uchar(self, cell_box):
        """
        Convert SCR_CELL_SHAPE cell_box to his braille character representation.
        """
        height, width = cell_box.shape
        uchar = ord(EMPTY_BRAILLE)
        for y, x in np.argwhere(cell_box):
            uchar |= self._pos_to_braille(ta.array([SCR_CELL_SHAPE[0] - 1 - y, x]))

        return uchar

    def _pos_to_braille(self, pos):
        """Point position as braille character in screen cell."""
        by = math.floor(pos[0] % SCR_CELL_SHAPE[0])
        bx = math.floor(pos[1] % SCR_CELL_SHAPE[1])

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

    def restore(self):
        """Restore static elements added to screen."""
        self._bg_buf = np.copy(self._bg_buf_backup)

    def refresh(self):
        """Draw buffer content to screen."""
        dtype = np.dtype('U' + str(self._bg_buf.shape[1]))
        for num, line in enumerate(self._bg_buf):
            self._curses_scr.addstr(num, 0, line.view(dtype)[0])
        self._curses_scr.refresh()

    def progress(self, current_time, total_time):
        """Show simulation calculation progress."""
        prog = 'Progress {proc:03.2f}%: {ctime:0.2f}/{ttime:0.2f} [sec] '.format(
            proc=current_time/total_time * 100,
            ctime=current_time,
            ttime=total_time)
        self._curses_scr.addstr(0, 0, prog)
        self._curses_scr.refresh()


class Body:
    RADIUS = 0.5

    def __init__(self, idx, pos, mass, vel):
        self.pos = pos
        self.prev_pos = pos
        self.mass = mass
        self.vel = vel
        self._idx = idx

    def __hash__(self):
        """Return Body unique hash (integer)."""
        return self._idx

    def __str__(self):
        """string representation of object."""
        return "Body(%d)" % self._idx

    def __repr__(self):
        """string representation of object."""
        return "Body(%d)" % self._idx


class NearestNeighborLookup:
    def __init__(self, bodies):
        self._bg_shape = ta.array([curses.LINES, curses.COLS-1])
        self._checked_pairs = {}
        self._create_scr_pos_map(bodies)

    def _create_scr_pos_map(self, bodies):
        """Map store for each screen cell list of bodies that is contains."""
        self._map = defaultdict(list)
        for body in bodies:
            scr_pos = phy_to_scr(body.pos)
            self._map[self._scr_pos_hash(scr_pos)].append(body)

    def neighbors(self, body):
        """Return list of body neighbors."""
        # Body can't collide with itself, so mark pair as checked
        pair_key = self._body_pair_hash(body, body)
        self._checked_pairs[pair_key] = True

        result = []
        scr_range_x, scr_range_y = self._scr_bounding_box(body)
        for x, y in it.product(scr_range_x, scr_range_y):
            scr_pos_key = self._scr_pos_hash(ta.array([y, x]))

            # Check all neighbors bodies from nearby screen cell
            for neigh_body in self._map[scr_pos_key]:
                pair_key = self._body_pair_hash(body, neigh_body)

                # If body pairs was already checked do nothing. We can have
                # only one such collision
                if pair_key in self._checked_pairs:
                    continue

                self._checked_pairs[pair_key] = True
                result.append(neigh_body)

        return result

    def _body_pair_hash(self, body1, body2):
        """Return bodies hashes in sorted order."""
        return minmax(hash(body1), hash(body2))

    def _scr_pos_hash(self, scr_pos):
        """
        Return screen pos hash. scr_pos (array/vector) doesn't have hash value,
        so this method generate it.
        """
        return scr_pos[0] * self._bg_shape[1] + scr_pos[1]

    def _scr_bounding_box(self, body):
        """
        Return bounding rectangle (screen cells coordinated), where nearby
        bodies should be searched.
        """
        direction = unit((body.pos - body.prev_pos)) * 3 * Body.RADIUS
        scr_pos = phy_to_scr(body.prev_pos)
        scr_prev_pos = phy_to_scr(body.pos + direction)

        x1, x2 = minmax(scr_pos[1], scr_prev_pos[1])
        y1, y2 = minmax(scr_pos[0], scr_prev_pos[0])
        return range(x1, x2+1), range(y1, y2+1)


class Terrain:
    EMPTY = np.array([0, 0])

    def __init__(self):
        normal_vecs_shape = ta.array([curses.LINES*SCR_CELL_SHAPE[0],
                                      (curses.COLS-1)*SCR_CELL_SHAPE[1]])
        self._normal_vecs = np.zeros(shape=(normal_vecs_shape[0],
                                            normal_vecs_shape[1],
                                            NORM_VEC_DIM))
        self._normal_marks = np.logical_or.reduce(self._normal_vecs!=Terrain.EMPTY, axis=-1)

    def add_array(self, arr, scr_shift=ta.array([0, 0])):
        """
        By default all arrays are drawn in bottom left corner of the screen.
        """
        arr_shift = scr_to_arr(scr_shift)
        arr, arr_shift = adjust_array(arr, arr_shift)
        arr_shape = ta.array(arr.shape[:2])

        x1 = arr_shift[1]
        x2 = x1 + arr_shape[1]
        y1 = self._normal_vecs.shape[0] - arr_shape[0] - arr_shift[0]
        y2 = self._normal_vecs.shape[0] - arr_shift[0]
        self._normal_vecs[y1:y2, x1:x2] = arr

        self._normal_marks = np.logical_or.reduce(self._normal_vecs!=Terrain.EMPTY, axis=-1)

    def cut_scrcell_box(self, scr_pos):
        """
        Cut normal vectors sub array with dimension of one screen cell -
        shape=(4, 2).
        """
        arr_pos = scr_to_arr(scr_pos)

        cell_box = self._normal_marks[arr_pos[0]:arr_pos[0]+SCR_CELL_SHAPE[0],
                                      arr_pos[1]:arr_pos[1]+SCR_CELL_SHAPE[1]]
        return cell_box

    def obstacles(self, pos, prev_pos):
        """
        Return all obstacles (represented by normal vectors) in rectangle, where
        pos and prev_pos determine rectangle diagonal.
        """
        arr_tl, arr_br = self._array_bounding_box(pos, prev_pos)
        box = self._cut_normal_vec_box(arr_tl, arr_br)
        box_markers = np.logical_or.reduce(box!=Terrain.EMPTY, axis=-1)

        result = []
        height, width = box_markers.shape
        for y, x in it.product(range(height), range(width)):
            if box_markers[y, x]:
                normal_vec = box[y, x]

                global_pos = arr_to_phy(arr_tl + (y, x))
                result.append((global_pos, normal_vec))

        return result

    def _array_bounding_box(self, pos, prev_pos):
        """
        Return top-left, bottom-right position of bounding box. Function add
        extra columns and rows in each dimension.
        """
        arr_pos = phy_to_arr(pos)
        arr_prev_pos = phy_to_arr(prev_pos)

        x1, x2 = minmax(arr_pos[1], arr_prev_pos[1])
        y1, y2 = minmax(arr_pos[0], arr_prev_pos[0])

        return ta.array([y1-1, x1-1]), ta.array([y2+2, x2+2])

    def _cut_normal_vec_box(self, arr_tl, arr_br):
        """
        Cut sub array from normal vectors, where arr_tl is top-left position,
        and arr_br bottom-right position of bounding rectangle.
        """
        # Fit array corner coordinates to not go out-of-bounds
        tl = ta.array([max(arr_tl[0], 0), max(arr_tl[1], 0)])
        br = ta.array([min(arr_br[0], self._normal_vecs.shape[0]),
                       min(arr_br[1], self._normal_vecs.shape[1])])
        # Cut normal vectors from terrain array
        box = self._normal_vecs[tl[0]:br[0], tl[1]:br[1]]

        # If bounding box is out of terrain bounds, we need to add border
        # padding
        expected_shape = (arr_br[0] - arr_tl[0], arr_br[1] - arr_tl[1], NORM_VEC_DIM)
        if expected_shape == box.shape:
            return box

        # Pad borders. If bounding box top-left corner is out of bound,
        # we need also create shit for terrain top-left corner position. It
        # will be needed to calculate distance.
        if arr_tl[1] < 0:
            wall = np.full(shape=(box.shape[0], 1, NORM_VEC_DIM), fill_value=ta.array([0, 1]))
            box = np.concatenate((wall, box), axis=1)
        elif arr_br[1] > self._normal_vecs.shape[1]:
            wall = np.full(shape=(box.shape[0], 1, NORM_VEC_DIM), fill_value=ta.array([0, -1]))
            box = np.concatenate((box, wall), axis=1)

        if arr_tl[0] < 0:
            wall = np.full(shape=(1, box.shape[1], NORM_VEC_DIM), fill_value=ta.array([-1, 0]))
            box = np.concatenate((wall, box), axis=0)
        elif arr_br[0] > self._normal_vecs.shape[0]:
            wall = np.full(shape=(1, box.shape[1], NORM_VEC_DIM), fill_value=ta.array([1, 0]))
            box = np.concatenate((box, wall), axis=0)

        # Fix corners position, normal vector should guide to center of screen
        # value = ±√(1² + 1²) = ±0.7071
        if arr_tl[1] < 0 and arr_tl[0] < 0:
            box[0, 0] = ta.array([-0.7071, 0.7071])
        elif arr_tl[1] < 0 and arr_br[0] > self._normal_vecs.shape[0]:
            box[-1, 0] = ta.array([0.7071, 0.7071])
        elif arr_br[1] > self._normal_vecs.shape[1] and arr_tl[0] < 0:
            box[0, -1] = ta.array([-0.7071, -0.7071])
        elif arr_br[1] > self._normal_vecs.shape[1] and arr_br[0] > self._normal_vecs.shape[0]:
            box[0, -1] = ta.array([0.7071, -0.7071])

        return box


class Importer:
    def load(self, ascii_file, normal_vec_file):
        """Load arrays from files."""
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
        Fill end of each line in ascii_fig with spaces, and convert it to
        numpy array.
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
        numpy array ([0, 0]) at the edges.
        """
        if norm_arr.shape[1] % SCR_CELL_SHAPE[1] or norm_arr.shape[0] % SCR_CELL_SHAPE[0]:
            raise Exception("Arrays with normal vector can't be transformed \
                to screen size buffer")

        ascii_markers = self._reduce_norm(norm_arr)
        del_rows = [list(range(idx*SCR_CELL_SHAPE[0], idx*SCR_CELL_SHAPE[0]+SCR_CELL_SHAPE[0]))
                    for idx, margin in enumerate(np.all(ascii_markers==False, axis=1)) if margin]
        norm_arr = np.delete(norm_arr, del_rows, axis=0)

        del_columns = [list(range(idx*SCR_CELL_SHAPE[1], idx*SCR_CELL_SHAPE[1]+SCR_CELL_SHAPE[1]))
                       for idx, margin in enumerate(np.all(ascii_markers==False, axis=0)) if margin]
        norm_arr = np.delete(norm_arr, del_columns, axis=1)

        return norm_arr

    def _reduce_norm(self, norm_arr):
        """
        Reduce array with normal vectors (for each braille characters), to
        "ASCII figure" array size, and mark if in "ASCII/screen cell" there
        was any character.
        """
        EMPTY_VEC = np.array([0, 0])
        marker_arr = np.logical_or.reduce(norm_arr!=EMPTY_VEC, axis=-1)

        result = []
        for y in range(0, marker_arr.shape[0], SCR_CELL_SHAPE[0]):
            for x in range(0, marker_arr.shape[1], SCR_CELL_SHAPE[1]):
                result.append(np.any(marker_arr[y:y+SCR_CELL_SHAPE[0], x:x+SCR_CELL_SHAPE[1]]))

        shape = marker_arr.shape // SCR_CELL_SHAPE
        result = np.reshape(result, shape)

        return result

    def _print_ascii_markers(self, norm_arr):
        """
        Print ASCII markers for cells in array with normal vectors.
        """
        ascii_markers = self._reduce_norm(norm_arr)
        log(ascii_markers.astype(int))

    def _validate_arrays(self, ascii_arr, norm_arr):
        """Validate if both arrays describe same thing."""
        norm_arr_shape = norm_arr.shape[:2] // SCR_CELL_SHAPE

        if np.any(ascii_arr.shape != norm_arr_shape):
            raise Exception('Imported arrays (ascii/norm) - mismatch size',
                ascii_arr.shape, norm_arr_shape)

        log('Validation OK')

##
# Helper functions.
##

def setup_curses(scr):
    """Setup curses screen."""
    curses.start_color()
    curses.use_default_colors()
    curses.halfdelay(5)
    curses.noecho()
    curses.curs_set(False)
    scr.clear()


def setup_telemetry(enable=False, terminal='/dev/pts/1'):
    """
    Redirect stderr to other terminal. Run tty command, to get terminal id.

    $ tty
    /dev/pts/1
    """
    global TELEMETRY_MODE
    TELEMETRY_MODE = enable
    if TELEMETRY_MODE:
        sys.stderr = open(terminal, 'w')


def log(*args, **kwargs):
    """Print on stderr."""
    if TELEMETRY_MODE:
        print(*args, file=sys.stderr)


def assert_that(condition):
    """Assert condition, disable curses and run pdb."""
    if not condition:
        curses.endwin()
        sys.stderr = sys.stdout
        pdb.set_trace()


def adjust_array(arr, shift):
    """Adjust array and shift, when one of shift values is negative."""
    new_arr = np.copy(arr)
    y, x = shift

    if shift[0] < 0:
        new_arr = new_arr[:new_arr.shape[0]+shift[0], :]
        y = 0
    if shift[1] < 0:
        new_arr = new_arr[:, -shift[1]:]
        x = 0
    new_shift = ta.array([y ,x])

    return new_arr, new_shift


def magnitude(vec):
    """Calculate vector magnitude."""
    return math.sqrt(vec[0]**2 + vec[1]**2)


def unit(vec):
    """Calculate unit vector."""
    mag = magnitude(vec)
    return vec/mag if mag else vec


def minmax(a, b):
    """Sort two values: first is min, second is max."""
    return (a, b) if a < b else (b, a)


def phy_to_scr(pos):
    """
    Screen cell position for given pos (physical position used in
    calculations.
    """
    x = math.floor(pos[1]/SCR_CELL_SHAPE[1])
    y = curses.LINES - 1 - math.floor(pos[0]/SCR_CELL_SHAPE[0])
    return ta.array([y, x])


def scr_to_arr(scr_pos):
    """
    Return top-left corner of screen cell in array coordinates (Y from top to
    bottom).
    """
    return scr_pos*SCR_CELL_SHAPE


def arr_to_phy(arr_pos):
    """
    Array position to Cartesian coordinate system used for physic
    calculations.
    """
    return ta.array([curses.LINES * SCR_CELL_SHAPE[0] - 1 - arr_pos[0], arr_pos[1]])


def phy_to_arr(pos):
    """
    Position (in Cartesian coordinate system) used in physic calculation to
    array position (Y from top to bottom).
    """
    y = curses.LINES * SCR_CELL_SHAPE[0] - 1 - math.floor(pos[0])
    return ta.array([y, math.floor(pos[1])])


def test_converters():
    """
    For DEBUG.
    Check if converters work properly.
    """
    arr_pos = phy_to_arr(ta.array([38, 50]))
    assert(np.all(ta.array([38, 50]) == arr_to_phy(arr_pos)))

    arr_pos = phy_to_arr(ta.array([46.25706000000003, 34.0]))
    assert np.all(ta.array([46, 34]) == arr_to_phy(arr_pos)), arr_to_phy(arr_pos)

##
# Physic engine.
##

def step_simulation(dt, bodies, terrain):
    """One step in simulation."""
    calc_forces(dt, bodies)
    integrate(dt, bodies)
    collisions = detect_collisions(bodies, terrain)
    resolve_collisions(dt, collisions)


def calc_forces(dt, bodies):
    """Calculate forces (gravity) for all bodies."""
    for body in bodies:
        body.forces = ta.array([-GRAVITY_ACC, 0.0]) * body.mass


def integrate(dt, bodies):
    """Integration motion equations."""
    for body in bodies:
        body.prev_pos = ta.array(body.pos)
        body.acc = body.forces / body.mass
        body.vel += body.acc * dt
        body.pos += body.vel * dt


class Collision:
    """Collision data need to resolve collision."""
    def __init__(self, body1, body2, dist, normal_vec):
        self.body1 = body1
        self.body2 = body2
        self.dist = dist
        self.normal_vec = normal_vec


def detect_collisions(bodies, terrain):
    """
    Detect collisions for all bodies with other bodies and terrain obstacles.
    """
    collisions = []
    nnlookup = NearestNeighborLookup(bodies)
    for body in bodies:
        collisions += obstacle_collisions(body, terrain)
        collisions += bodies_collisions(body, nnlookup)

    return collisions


def obstacle_collisions(body, terrain):
    """Calculate collision with terrain obstacles."""
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


def bodies_collisions(body, nnlookup):
    """Calculate body collision with his neighbors."""
    result = []

    for neigh_body in nnlookup.neighbors(body):
        dist = body.pos - neigh_body.pos
        normal_vec = unit(dist)
        real_dist = magnitude(dist) - 2*Body.RADIUS
        collision = Collision(body1=body,
                              body2=neigh_body,
                              dist=real_dist,
                              normal_vec=normal_vec)

        result.append(collision)

    return result


def resolve_collisions(dt, collisions):
    """
    Speculative contacts solver.

    References:
    https://en.wikipedia.org/wiki/Collision_response#Impulse-based_reaction_model
    https://wildbunny.co.uk/blog/2011/03/25/speculative-contacts-an-continuous-collision-engine-approach-part-1/
    http://twvideo01.ubm-us.net/o1/vault/gdc2013/slides/824737Catto_Erin_PhysicsForGame.pdf
    https://github.com/mattleibow/jitterphysics/wiki/Speculative-Contacts
    https://github.com/pratik2709/Speculative-Contacts
    https://codepen.io/kenjiSpecial/pen/bNJQKQ
    """
    for _, c in it.product(range(NUM_ITERATION), collisions):
        # Body collide with screen borders
        if not c.body2:
            relative_vel = -c.body1.vel
            remove = np.dot(relative_vel, c.normal_vec) - c.dist/dt

            if remove < 0:
                continue

            impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * remove) / \
                    (1/c.body1.mass)
            c.body1.vel -= (c.normal_vec / c.body1.mass) * impulse
        # Collision between two bodies
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
    print('Library. Nothing to execute.')
