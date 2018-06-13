import sys
import pdb
# import rpdb

def main():
    print('first')
    n = 1337

    f0 = open('/tmp/aaa', 'r')
    f1 = open('/tmp/aaa', 'w')
    # pdb.Pdb(stdin=f0, stdout=f1).set_trace()
    # sys.stdin = open('/tmp/aaa', 'r')
    # sys.stdout = open('/tmp/aaa', 'w')
    # pdb.set_trace()
    import rpdb2; rpdb2.start_embedded_debugger('asdf')
    # import rpdb; rpdb.set_trace()
    # debugger = rpdb.Rpdb(port=12345)
    # debugger.set_trace()

    print('end')


if __name__ == '__main__':
    main()