#! /usr/bin/env python3

import locale
import curses
import random
import time
import ascii_engine as ae


REFRESH_RATE = 30  # FPS
ANIMATION_TIME = 3  # [sec]


def main(scr):
    ae.setup_curses(scr)
    ae.setup_telemetry(enable=False, terminal='/dev/pts/3')

    screen, terrain = create_scene(scr)
    bodies = create_bodies(count=200)

    dt = 1/REFRESH_RATE

    t = 0
    animation = []
    while t < ANIMATION_TIME:
        ae.step_simulation(dt, bodies, terrain)
        animation.append([body.pos for body in bodies])
        screen.progress(t, ANIMATION_TIME)
        t += dt

    # Play animation in loop
    while True:
        t = 0

        for step in animation:
            tic = time.time()

            screen.restore()
            for body_pos in step:
                screen.draw_point(body_pos)
            screen.refresh()

            delay = max(0, dt - (time.time() - tic))
            ae.telemetry_log(delay)
            time.sleep(delay)
            t += dt


def create_scene(scr):
    """Create scene with obstacles. Return screen and terrain objects."""
    terrain = ae.Terrain()
    screen = ae.Screen(scr, terrain)

    im = ae.Importer()
    ascii_arr, norm_arr = im.load('ascii_fig.txt', 'ascii_fig.png.norm')

    terrain.add_array(norm_arr)
    screen.add_ascii_array(ascii_arr)

    return screen, terrain


def create_bodies(count):
    """Create bodies."""
    random.seed(3300)
    height = curses.LINES * ae.SCR_CELL_SHAPE[0]
    width = (curses.COLS-1) * ae.SCR_CELL_SHAPE[1]

    bodies = []
    visited = {}

    idx = 0
    while idx < count:
        y = height - random.randint(2, 20)
        x = random.randint(1, width)

        if (y, x) in visited:
            continue

        visited[(y, x)] = True
        bodies.append(ae.Body(idx=idx,
                              pos=(y, x),
                              mass=1,
                              vel=(-40, 0)))
        idx += 1

    return bodies


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    curses.wrapper(main)
