import numpy as np

def mag(v):
    return np.linalg.norm(v)

def res(vel1, vel2, m1=2, m2=3, n=np.array([1,0]), e=1):
    vr = np.dot(vel2 - vel1, n)
    print("vr*n:", vr)
    j = (-(1+e) * vr)/(1/m1 + 1/m2)
    vel1p = vel1 - (j/m1)*n
    vel2p = vel2 + (j/m2)*n
    print('real e:', e)
    print('j:', j)
    print("v1  v2: ", vel1, vel2)
    print("v1' v2':", vel1p, vel2p)
    #print("ll", vel2p - velP)
    #print('mm', vel2 - vel1)
    e1 = mag(vel2p - vel1p)/mag(vel2 - vel1)
    print('Calc e1:', e1)
    #e2 = (vel2p - vel1p)/(vel2 - vel1)
    e3 = (vel2p[0] - vel1p[0])/(vel2[0] - vel1[0])
    print('Calc e3:', e3)
    vpr = np.dot(vel2p - vel1p, n)
    print('check:', vpr, -e*vr)

v1 = np.array([1.53142525 ,-30.50398945])
v2 = np.array([-2.31223138, -26.98750722])
n = np.array([-0.9999, -0.0143])
res(v1, v2, m1=1, m2=1, e=0.1, n=n)

v1 = np.array([-2, 0])
v2 = np.array([-3, 0])
n = np.array([1, 0])
# res(v1, v2, m1=1, m2=1, e=0.5, n=n)
# res(v1, v2, m1=10, m2=2, e=0.5, n=n)

v1 = np.array([-2, 0])
v2 = np.array([3, 0])
n = np.array([1, 0])
# res(v1, v2, m1=1, m2=1, e=0.5, n=n)
# res(v1, v2, m1=10, m2=2, e=0.5, n=n)