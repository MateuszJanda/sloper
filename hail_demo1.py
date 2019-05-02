#! /usr/bin/env python3

"""
Author: Mateusz Janda <mateusz janda at gmail com>
Site: github.com/MateuszJanda
"""

import locale
import curses
import random
import time
import ascii_engine as ae


REFRESH_RATE = 30  # FPS
ANIMATION_TIME = 10  # [sec]


def main(scr):
    ae.setup(scr, debug=False, terminal='/dev/pts/3')

    screen, terrain = create_scene(scr)
    bodies = create_bodies(count=2000)

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
            time.sleep(delay)
            t += dt


def create_scene(scr):
    """Create scene with obstacles. Return screen and terrain objects."""
    terrain = ae.Terrain()
    screen = ae.Screen(scr, terrain)

    im = ae.Importer()

    ascii_arr, norm_arr = im.load('ascii_data/umbrella.txt', 'ascii_data/umbrella-drilled.surf')
    add_obstacle(screen, terrain, ascii_arr, norm_arr, scr_shift=(0, 25))
    add_obstacle(screen, terrain, ascii_arr, norm_arr, scr_shift=(-2, 64))

    ascii_arr, norm_arr = im.load('ascii_data/rect.txt', 'ascii_data/rect.surf')
    add_obstacle(screen, terrain, ascii_arr, norm_arr, scr_shift=(-7, 5))
    add_obstacle(screen, terrain, ascii_arr, norm_arr, scr_shift=(-9, 105))
    add_obstacle(screen, terrain, ascii_arr, norm_arr, scr_shift=(1, -1))
    add_obstacle(screen, terrain, ascii_arr, norm_arr, scr_shift=(3, 75))

    return screen, terrain


def add_obstacle(screen, terrain, ascii_arr, norm_arr, scr_shift):
    terrain.add_surface_array(norm_arr, scr_shift)
    screen.add_ascii_array(ascii_arr, scr_shift)


def create_bodies(count):
    """Create bodies."""
    random.seed(3300)
    height = curses.LINES * ae.SCR_CELL_SHAPE[0]
    width = (curses.COLS-1) * ae.SCR_CELL_SHAPE[1]

    bodies = []
    visited = {}

    idx = 0
    while idx < count:
        y = height - random.randint(2, 25)
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
