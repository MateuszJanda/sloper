#! /usr/bin/env python3

import ascii_engine as ae
import random
import curses
import locale
import tinyarray as ta

REFRESH_RATE = 100


def main(scr):
    ae.setup_curses(scr)

    screen, terrain = create_scene(scr)
    bodies = create_bodies(count=50)

    t = 0
    dt = 1/REFRESH_RATE

    while True:
        screen.restore()
        ae.step_simulation(dt, bodies, terrain)
        for body in bodies:
            screen.draw_point(body.pos)
        screen.refresh()

        # time.sleep(dt)
        t += dt

    curses.endwin()


def create_scene(scr):
    """Create scene with obstacles. Return screen and terrain objects."""
    terrain = ae.Terrain()
    screen = ae.Screen(scr, terrain)

    im = ae.Importer()
    ascii_arr, norm_arr = im.load('ascii_fig.txt', 'ascii_fig.png.norm')

    terrain.add_array(norm_arr)
    screen.add_ascii_array(ascii_arr)
    # screen.add_terrain_data()

    return screen, terrain


def create_bodies(count):
    """Create bodies."""
    random.seed(3300)
    height, width = curses.LINES*ae.BUF_CELL_SIZE[0], (curses.COLS-1)*ae.BUF_CELL_SIZE[1]

    bodies = []
    visited = {}

    idx = 0
    while idx < count:
        y, x = height - (random.randint(2, 20) * 1.0), random.randint(1, width)

        if (y, x) in visited:
            continue

        visited[(y, x)] = True
        bodies.append(ae.Body(idx=idx,
                           pos=ta.array([y, x]),
                           mass=1,
                           vel=ta.array([-40.0, 0])))
        idx += 1

    # bodies = [
    #     # Body(idx=1, pos=ta.array([80.0, 32]), mass=1, vel=ta.array([-40.0, 0])),
    #     # Body(idx=1, pos=ta.array([80.0, 34]), mass=1, vel=ta.array([-40.0, 0])),
    #     # Body(idx=1, pos=ta.array([80.0, 50]), mass=1, vel=ta.array([-40.0, 0])),
    #     # Body(idx=1, pos=ta.array([80.0, 112]), mass=1, vel=ta.array([-40.0, 0])),
    #     # Body(idx=1, pos=ta.array([70.0, 110.5]), mass=1, vel=ta.array([-40.0, 0])),
    #     # Body(idx=1, pos=ta.array([80.0, 110]), mass=1, vel=ta.array([-40.0, 0])),
    #     # Body(idx=1, pos=ta.array([80.0, 23]), mass=1, vel=ta.array([-40.0, 0])),
    #     # Body(idx=1, pos=ta.array([80.0, 22]), mass=1, vel=ta.array([-40.0, 0])),
    #     # Body(idx=1, pos=ta.array([80.0, 21]), mass=1, vel=ta.array([-40.0, 0])),
    #     # Body(idx=1, pos=ta.array([80.0, 20]), mass=1, vel=ta.array([-40.0, 0])),
    #     # Body(idx=1, pos=ta.array([1.0, 110]), mass=1, vel=ta.array([0.0, 1])),
    #     # Body(idx=1, pos=ta.array([1.0, 116]), mass=1, vel=ta.array([0.0, 0])),
    # ]

    # for idx, body in enumerate(bodies):
    #     body._idx = idx

    return bodies


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    ae.setup_stderr()
    curses.wrapper(main)