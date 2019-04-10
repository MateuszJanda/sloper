import numpy as np

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

    def normal(self):
        """Normal vector - perpendicular normalized vector."""
        # return Vector(x=-self.y, y=self.x) / self.magnitude()
        mag = self.magnitude()
        if mag:
            return self/mag
        return self


    def __str__(self):
        """string representation of object."""
        # return "Vector(x=" + str(self.x) + ", y=" + str(self.y) + ")"
        return "Vector(x=%.4f, y=%.4f)" % (self.x, self.y)

    def __repr__(self):
        """string representation of object."""
        # return "Vector(x=" + str(self.x) + ", y=" + str(self.y) + ")"
        return "Vector(x=%.4f, y=%.4f)" % (self.x, self.y)


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


v1 = Vector(x=1.0000, y=0.0858)
v2 = Vector(x=-1.0000, y=0.1344)
n = Vector(x=1.0000, y=0.0000)
res(v1, v2, m1=1, m2=1, e=0.5, n=n)
