#! /usr/bin/env python3

"""
Coordinates systems:
    pos         - position in Cartesian coordinate system
    buf_pos     - position on screen (of one character). Y from top to bottom
    arr_pos     - similar to pos, but Y from top to bottom
"""


import sys
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
    screen.add_ascii_array(ascii_arr)
    terrain.add_arr(norm_arr, buf_shift=Vector(x=40, y=0))
    screen.add_common_array(norm_arr, buf_shift=Vector(x=40, y=0))

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

        # Redundant
        self._buf_size = Size(curses.LINES, curses.COLS-1)
        self._screen_size = self._buf_size*BUF_CELL_SIZE

        self._buf = self._create_empty_buf()
        self._buf_backup = copy.deepcopy(self._buf)

    def _create_empty_buf(self):
        """
        Create empty screen buffer filled with "empty braille" characters.
        TODO: Replace it by np.array (constrained type)
        """
        return [list(EMPTY_BRAILLE * self._buf_size.width) for _ in range(self._buf_size.height)]

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
                pos = arrpos_to_ptpos(arr_pos)
                self.draw_point(pos)

        self._save_in_backup_buf()

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
        self._buf_backup = copy.deepcopy(self._buf)

    def draw_point(self, pos):
        """
        Draw (put in screen buffer) single point. If theres is any ASCII
        character in screen cell, function will replace this character to his
        braille representation and merge this single point.
        """
        # Don't draw point when they are out of the screen
        if not (0 <= pos.x < self._screen_size.width and 0 <= pos.y < self._screen_size.height):
            return

        buf_pos = ptpos_to_bufpos(pos)
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
            uchar |= self._pos_to_braille(Vector(x=x, y=BUF_CELL_SIZE.height-y))

        return uchar

    def _pos_to_braille(self, pos):
        """Point position as braille character in BUF_CELL."""
        bx = int(pos.x) % BUF_CELL_SIZE.width
        by = int(pos.y) % BUF_CELL_SIZE.height

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
        self.prev_ptpos = ptpos
        self.mass = mass
        self.vel = velocity


class Terrain:
    EMPTY = np.array([0, 0])

    def __init__(self):
        # Redundant
        self._terrain_size = Size(curses.LINES*BUF_CELL_SIZE.height,
                                  (curses.COLS-1)*BUF_CELL_SIZE.width)
        self._terrain = np.zeros(shape=(self._terrain_size.height,
                                        self._terrain_size.width, VECTOR_DIM))

    def add_arr(self, arr, buf_shift=Vector(x=0, y=0)):
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

    def obstacles(self, ptpos, prev_ptpos):
        arr_tl, arr_br = self._bounding_box(ptpos, prev_ptpos)
        box, arr_shift = self._cut_normal_vec_box(arr_tl, arr_br)
        box_markers = np.logical_or.reduce(box != Terrain.EMPTY, axis=-1)

        result = []
        for y, x in np.argwhere(box_markers):
            local_obs_pos = Vector(x=x, y=y)
            normal_vec = box[local_obs_pos.y, local_obs_pos.x]

            global_pos = arrpos_to_ptpos(arr_tl + arr_shift + local_obs_pos)
            result.append((global_pos, normal_vec))

        return result

    def _bounding_box(self, ptpos, prev_ptpos):
        arr_pos = ptpos_to_arrpos(ptpos)
        arr_prev_pos = ptpos_to_arrpos(prev_ptpos)

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
        arr_shift = Vector(x=0, y=0)

        # If bounding box is out of terrain bounds, we need to add border padding
        expected_shape = (arr_br.y - arr_tl.y, arr_br.x - arr_tl.x, VECTOR_DIM)
        if expected_shape == box.shape:
            return box, arr_shift

        # Pad borders. If bounding box top-left corner is out of bound,
        # we need also create shit for terrain top-left corner position. It
        # will be needed to calculate distance.
        if arr_tl.x < 0:
            box = np.hstack((np.full(shape=(box.shape[0], 1, VECTOR_DIM), fill_value=Vector(x=1, y=0)), box))
            arr_shift = Vector(x=arr_shift.x+1, y=arr_shift.y)
        if arr_tl.y < 0:
            arr_shift = Vector(x=arr_shift.x, y=arr_shift.y+1)
            box = np.vstack((np.full(shape=(1, box.shape[1], VECTOR_DIM), fill_value=Vector(x=0, y=-1)), box))

        if arr_br.x > self._terrain_size.width:
            box = np.hstack((box, np.full(shape=(box.shape[0], 1, VECTOR_DIM), fill_value=Vector(x=-1, y=0))))
        if arr_br.y > self._terrain_size.height:
            box = np.vstack((box, np.full(shape=(1, box.shape[1], VECTOR_DIM), fill_value=Vector(x=0, y=1))))

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
        ascii_arr[0:3, 0:3] = np.array([' ' for _ in range(9)]).reshape(3, 3)
        return ascii_arr

    def _remove_ascii_margin(self, ascii_arr):
        """
        Remove margin from ascii_arr (line and columns with spaces at the edges.
        """
        del_rows = [idx for idx, margin in enumerate(np.all(ascii_arr == ' ', axis=0)) if margin]
        ascii_arr = np.delete(ascii_arr, del_rows, axis=1)

        del_columns = [idx for idx, margin in enumerate(np.all(ascii_arr == ' ', axis=1)) if margin]
        ascii_arr = np.delete(ascii_arr, del_columns, axis=0)

        return ascii_arr

    def _import_norm_arr(self, norm_file):
        """Import array with normal vector."""
        arr = np.loadtxt(norm_file)
        height, width = arr.shape
        norm_arr = arr.reshape(height, width//VECTOR_DIM, VECTOR_DIM)

        return norm_arr

    def _remove_norm_margin(self, norm_arr):
        """
        Remove margin from array with normal vectors (line and columns with
        np.array([0, 0]) at the edges.
        """
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
        """
        Transform array with normal vectors (for braille characters), to
        dimensions of ASCII figure, and mark if in ASCII cell there was any
        character.
        """
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
        """Validate if both arrays describe same thing."""
        ascii_arr_size = Size(*ascii_arr.shape)
        norm_arr_size = norm_arr.shape[:2]//BUF_CELL_SIZE

        if np.any(ascii_arr_size != norm_arr_size):
            raise Exception('Imported arrays (ascii/norm) - mismatch size', ascii_arr_size, norm_arr_size)

        eprint('Validation OK')


def ptpos_to_bufpos(pos):
    x = int(pos.x/BUF_CELL_SIZE.width)
    y = curses.LINES - 1 - int(pos.y/BUF_CELL_SIZE.height)
    return Vector(x=x, y=y)


def bufpos_to_arrpos(buf_pos):
    """
    Return top-left corner to buffer cell in array coordinates (Y from top to
    bottom).
    """
    return Vector(x=buf_pos.x*BUF_CELL_SIZE.width, y=buf_pos.y*BUF_CELL_SIZE.height)


def arrpos_to_ptpos(arr_pos):
    """Array position to Cartesian coordinate system"""
    return Vector(x=arr_pos.x, y=curses.LINES * BUF_CELL_SIZE.height - 1 - arr_pos.y)


def ptpos_to_arrpos(pos):
    """
    Point position (in Cartesian coordinate system) to array position (Y from
    top to bottom).
    """
    y = curses.LINES * BUF_CELL_SIZE.height - 1 - int(pos.y)
    return Vector(x=int(pos.x), y=int(y))


def test_converters():
    """
    For DEBUG.
    Check if converters work properly.
    """
    assert(np.all(Vector(x=50, y=38) == Vector(x=50, y=38)))

    arr_pos = ptpos_to_arrpos(Vector(x=50, y=38))
    assert(np.all(Vector(x=50, y=38) == arrpos_to_ptpos(arr_pos)))

    arr_pos = ptpos_to_arrpos(Vector(x=34.0, y=46.25706000000003))
    assert np.all(Vector(x=34, y=46) == arrpos_to_ptpos(arr_pos)), arrpos_to_ptpos(arr_pos)


def step_simulation(dt, bodies, terrain):
    calc_forces(dt, bodies)
    integrate(dt, bodies)
    collisions = detect_collisions(bodies, terrain)
    resolve_collisions(dt, collisions)


def calc_forces(dt, bodies):
    for body in bodies:
        body.forces = Vector(x=0, y=-GRAVITY_ACC) * body.mass

        if int(body.prev_ptpos.y) == 0 and int(body.ptpos.y) == 0:
            body.forces *= COEFFICIENT_OF_FRICTION


def integrate(dt, bodies):
    for body in bodies:
        body.prev_ptpos = copy.copy(body.ptpos)
        body.acc = body.forces / body.mass
        body.vel = body.vel + body.acc * dt
        body.ptpos = body.ptpos + body.vel * dt


class Collision:
    def __init__(self, body1, body2, dist, normal_vec):
        self.body1 = body1
        self.body2 = body2
        self.dist = dist
        self.normal_vec = normal_vec


def detect_collisions(bodies, terrain):
    collisions = []
    for body in bodies:
        c = obstacle_collisions(body, terrain)
        collisions += c

    return collisions


def obstacle_collisions(body, terrain):
    result = []

    for obstacle_ptpos, normal_vec in terrain.obstacles(body.ptpos, body.prev_ptpos):
        r = 0.5
        p1 = np.floor(body.ptpos) + Vector(x=r, y=r)
        # p1 = body.ptpos + Vector(x=r, y=r)
        p2 = np.floor(obstacle_ptpos) + Vector(x=r, y=r)
        dist = Vector(*(p1 - p2)).magnitude() - 2*r

        collision = Collision(body1=body,
                              body2=None,
                              dist=dist,
                              normal_vec=normal_vec)

        collision.obs_pos = obstacle_ptpos
        result.append(collision)

    return result


def resolve_collisions(dt, collisions):
    for i in range(3):
        for c in collisions:
            # Collision with screen border
            if not c.body2:

                relative_vel = -c.body1.vel

                eprint('CCC relV=%s, norm=%s, dot=%s, len=%d, o_pos=%s' % (relative_vel, c.normal_vec, np.dot(relative_vel, c.normal_vec), len(collisions), c.obs_pos))
                remove = np.dot(-relative_vel, c.normal_vec) + c.dist/dt
                eprint('remove=%f, dist=%f, vvvel=%f' % (remove, c.dist, c.dist/dt))

                if remove < 0 :
                    mark = True

                    # impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * np.dot(c.relative_vel, c.normal_vec)) / \
                    #         (1/c.body1.mass)
                    # impulse = (remove) / \
                            # (1/c.body1.mass)

                    impulse = (-(1+COEFFICIENT_OF_RESTITUTION) * remove) / \
                            (1/c.body1.mass)

                    c.body1.vel += (c.normal_vec / c.body1.mass) * impulse
                    c.body1.ptpos -= c.body1.vel * dt

                    eprint('body AFTER pos=%s, vel=%s' % (c.body1.ptpos, c.body1.vel))


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    setup_stderr()
    curses.wrapper(main)
