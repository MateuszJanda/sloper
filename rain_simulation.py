#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import sys
import collections as co
import numpy as np
import curses
import locale

locale.setlocale(locale.LC_ALL, '')
G = 1.0
VECTOR_DIM = 2


Vector = co.namedtuple('Vector', ['x', 'y'])


class StdOutWrapper:
    text = ""
    def write(self,txt):
        self.text += str(txt) + '\n'
    def get_text(self):
        return self.text



def main():
    out = StdOutWrapper()
    sys.stdout = out
    sys.stderr = out

    scr = setup()

    # bodies = [
    #     Body(pos=Vector(110, 80), mass=10000, velocity=Vector(0, 0)),
    #     Body(pos=Vector(50, 100), mass=10, velocity=Vector(12, 3)),
    #     Body(pos=Vector(95, 80), mass=1, velocity=Vector(9, 21))
    # ]

    # print 'asdf'

    file_name = 'ascii_fig.png.norm'
    norm_vec_arr = import_norm_vector_arr(file_name)
    out.write(norm_vec_arr.shape)
    # out.write('asdf')
    # out.write(norm_vec_arr)

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

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    sys.stdout.write(out.get_text())


def setup():
    scr = curses.initscr()
    curses.start_color()
    curses.use_default_colors()
    curses.halfdelay(5)
    curses.noecho()
    curses.curs_set(False)
    scr.clear()
    return scr


def import_norm_vector_arr(file_name):
    arr = np.loadtxt(file_name)
    height, width = arr.shape
    return arr.reshape(height, width//VECTOR_DIM, VECTOR_DIM)


if __name__ == '__main__':
    main()
