#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import sys
import os
import code
import collections as co
import numpy as np
import curses
import locale

locale.setlocale(locale.LC_ALL, '')
G = 1.0
VECTOR_DIM = 2


Vector = co.namedtuple('Vector', ['x', 'y'])


# class StdOutWrapper:
#     text = ""
#     def write(self,txt):
#         self.text += str(txt) + '\n'
#     def get_text(self):
#         return self.text


def eprint(*args, **kwargs):
    # print(*args, file=sys.stderr, **kwargs)
    print(*args, file=sys.stderr)

def eassert(condition, local):
    if not condition:
        # c = code.InteractiveConsole(locals=locals(), filename='/dev/pts/1')
        # c.interact()
        curses.endwin()
        # code.interact(local=locals())
        # sys.stdout = sys.stderr
        # code.interact(local=locals())
        # code.interact(local=dict(globals(), **locals()))
        import pdb; pdb.set_trace()
        # f1 = open('/dev/pts/1', 'r')
        # import pdb; pdb.Pdb(stdin=f1, stdout=sys.stderr).set_trace()
        # import pdb; pdb.Pdb(stdout=sys.stderr).set_trace()



def main():
    # with open('/dev/pts/1', 'rb') as inf, open('/dev/pts/1', 'wb') as outf:
    #     os.dup2(inf.fileno(), 0)
    #     os.dup2(outf.fileno(), 1)
    #     os.dup2(outf.fileno(), 2)
        curses.wrapper(run)


def run(scr):
    # out = StdOutWrapper()
    # sys.stdout = out
    # sys.stderr = out

    setup(scr)

    # bodies = [
    #     Body(pos=Vector(110, 80), mass=10000, velocity=Vector(0, 0)),
    #     Body(pos=Vector(50, 100), mass=10, velocity=Vector(12, 3)),
    #     Body(pos=Vector(95, 80), mass=1, velocity=Vector(9, 21))
    # ]

    # print 'asdf'

    file_name = 'ascii_fig.png.norm'
    norm_vec_arr = import_norm_vector_arr(file_name)
    # out.write(norm_vec_arr.shape)
    # out.write('asdf')
    # out.write(norm_vec_arr)
    # eprint(norm_vec_arr)
    eprint('info on error')

    # scr.addstr(0, 0, u''.join('Hello world').encode('utf-8'))
    scr.addstr(0, 0, 'Hello world')
    scr.refresh()

    # while scr.getch() == -1:
        #
    eprint('you entered something')

    n = 1337
    eassert(n < 0, locals())

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

    # sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__
    # sys.stdout.write(out.get_text())


def setup(scr):
    """
    Setup curses screen
    """
    # scr = curses.initscr()
    curses.start_color()
    curses.use_default_colors()
    curses.halfdelay(5)
    curses.noecho()
    curses.curs_set(False)
    scr.clear()
    # return scr


def import_norm_vector_arr(file_name):
    arr = np.loadtxt(file_name)
    height, width = arr.shape
    return arr.reshape(height, width//VECTOR_DIM, VECTOR_DIM)


if __name__ == '__main__':
    main()
