#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import sys
import collections as co
import numpy as np
import curses
import locale
import pdb


locale.setlocale(locale.LC_ALL, '')
G = 1.0
VECTOR_DIM = 2


Vector = co.namedtuple('Vector', ['x', 'y'])


def main():
    # esetup()
    print('asdf')
    # sys.stdout = open('/dev/pts/5', 'w')
    sys.stderr = open('/dev/pts/6', 'w')
    # print(type(sys.stdout))
    print('jkl')

    curses.setupterm(fd=sys.stdout.fileno())

    curses.wrapper(run)


def esetup():
    print('asdf')
    # sys.stdout = open('/dev/pts/5', 'w')
    sys.stderr = open('/dev/pts/6', 'w')
    print('jkl')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr)


def eassert(condition):
    if not condition:
        curses.endwin()
        sys.stderr = sys.stdout
        pdb.set_trace()


def run(scr):
    print(dir(scr))
    setup(scr)
    print('xxx')

    file_name = 'ascii_fig.png.norm'
    norm_vec_arr = import_norm_vector_arr(file_name)
    eprint('erro1')

    scr.addstr(0, 0, 'Hello world')
    scr.refresh()

    eprint('erro2')

    while scr.getch() == -1:
        continue

    n = 1337
    # eassert(n < 0)

    # out.write('asdf')

    # t = 0
    # freq = 100
    # dt = 1.0/freq
    # screen_buf = clear_buf()

    # while True:
    #     calcs(bodies, dt)

    #     for b in bodies:
    #         draw_pt(screen_buf, b.pos)
    #     draw_info(screen_buf,  '[%.2f]: %.4f %.4f' % (t, bodies[1].pos.x, bodies[1].pos.y))
    #     display(scr, screen_buf)

    #     time.sleep(0.01)
    #     t += dt

    curses.endwin()


def setup(scr):
    """Setup curses screen"""

    curses.start_color()
    curses.use_default_colors()
    curses.halfdelay(5)
    curses.noecho()
    curses.curs_set(False)
    scr.clear()


def import_norm_vector_arr(file_name):
    arr = np.loadtxt(file_name)
    height, width = arr.shape
    return arr.reshape(height, width//VECTOR_DIM, VECTOR_DIM)


if __name__ == '__main__':
    main()
