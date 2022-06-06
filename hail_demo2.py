#! /usr/bin/env python3

# Author: Mateusz Janda <mateusz janda at gmail com>
# Site: github.com/MateuszJanda/sloper
# Ad maiorem Dei gloriam

import locale
import curses
import random
import time
import ascii_engine as ae


REFRESH_RATE = 30  # FPS


def main(scr):
    ae.setup(scr, debug_terminal=None)
    random.seed(3300)

    screen, terrain = create_scene(scr)

    bodies, visited = create_bodies(bodies=[], visited=[], count=40)

    dt = 1/REFRESH_RATE
    t = 0
    while True:
        tic = time.time()
        ae.step_simulation(dt, bodies, terrain)

        screen.restore()
        for body in bodies:
            screen.draw_point(body.pos)
        screen.refresh()

        if t < 4:
            bodies, visited = create_bodies(bodies, visited, count=2)
        if t > 4:
            bodies, visited = replace_bodies(bodies, visited, count=2)

        delay = max(0, dt - (time.time() - tic))
        time.sleep(delay)
        t += dt


def create_scene(scr):
    """Create scene with obstacles. Return screen and terrain objects."""
    terrain = ae.Terrain()
    screen = ae.Screen(scr, terrain)

    im = ae.Importer()

    ascii_arr, norm_arr = im.load('ascii_data/cat.txt', 'ascii_data/cat-drilled.surf')
    add_obstacle(screen, terrain, ascii_arr, norm_arr, scr_shift=(0, 25))

    return screen, terrain


def add_obstacle(screen, terrain, ascii_arr, norm_arr, scr_shift):
    terrain.add_surface_array(norm_arr, scr_shift)
    screen.add_ascii_array(ascii_arr, scr_shift)


def replace_bodies(bodies, visited, count):
    bodies = bodies[count:]
    visited = visited[count:]
    return create_bodies(bodies, visited, count)


def create_bodies(bodies, visited, count):
    """Create bodies."""
    height = curses.LINES * ae.SCR_CELL_SHAPE[0]
    width = (curses.COLS-1) * ae.SCR_CELL_SHAPE[1]

    idx = 0
    while idx < count:
        y = height - random.randint(2, 20)
        x = random.randint(1, width)

        if (y, x) in visited:
            continue

        visited.append((y, x))
        bodies.append(ae.Body(idx=idx,
                              pos=(y, x),
                              mass=1,
                              vel=(-40, 12)))
        idx += 1

    return bodies, visited


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, '')
    curses.wrapper(main)
