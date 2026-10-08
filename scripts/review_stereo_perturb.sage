# Adversarial perturbation tests of stereographic.sage helpers (copies, not the original).
import numpy as np

def unit(v):
    v = np.array(v, dtype=float); return v / np.linalg.norm(v)

def ortho_frame(n):
    n = unit(n)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, n)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = unit(helper - np.dot(helper, n) * n)
    v = np.cross(n, u)
    return n, u, v

def line_sphere_intersection(n1, c1, n2, c2):
    n1 = unit(n1); n2 = unit(n2)
    d = np.cross(n1, n2)
    assert np.linalg.norm(d) > 1e-9, "circle planes are parallel"
    G = np.array([[n1 @ n1, n1 @ n2], [n2 @ n1, n2 @ n2]])
    rhs = np.array([c1, c2])
    a, b = np.linalg.solve(G, rhs)
    p0 = a * n1 + b * n2
    A = d @ d; B = 2.0 * (p0 @ d); C = p0 @ p0 - 1.0
    disc = B * B - 4 * A * C
    assert disc > 1e-9, "circles do not intersect"
    sq = np.sqrt(disc)
    return p0 + (-B + sq)/(2*A) * d, p0 + (-B - sq)/(2*A) * d

# Test 1: nearly-parallel normals -> G near singular. Chosen nR,nB differ only in x-sign.
# What if someone makes the two planes nearly identical normals?
print("=== Test 1: near-parallel normals ===")
try:
    n1 = unit([0.01, 0.0, -1.0]); n2 = unit([-0.01, 0.0, -1.0])
    r = line_sphere_intersection(n1, 0.4, n2, 0.4)
    print("ok, cond(G) tiny-angle case returned", r[0])
except Exception as e:
    print("raised:", repr(e))

# Test 2: tweak plane constant so circles DON'T intersect (c too large) -> disc<0
print("=== Test 2: c too large, no intersection ===")
nR = unit([0.49,0,-0.87]); nB = unit([-0.49,0,-0.87])
try:
    r = line_sphere_intersection(nR, 0.99, nB, 0.99)
    print("ok returned (unexpected):", r[0])
except AssertionError as e:
    print("AssertionError (loud, good):", e)
except Exception as e:
    print("other:", repr(e))

# Test 3: ortho_frame with n=+z axis (|n[0]|=0 <0.9 -> helper x). cross degenerate?
print("=== Test 3: ortho_frame on axis normals ===")
for n in [[0,0,1],[1,0,0],[0,1,0],[0,0,-1]]:
    nn,u,v = ortho_frame(n)
    print(n, "-> |u|=%.3f |v|=%.3f u.n=%.2e" % (np.linalg.norm(u), np.linalg.norm(v), u@nn))

# Test 4: what if a used sphere point has z very close to 1 (project blows up)?
print("=== Test 4: project near north pole ===")
P = np.array([0.0,0.0,1.0])
for z in [0.5, 0.9, 0.999, 1.0-1e-10, 1.0]:
    Q = np.array([np.sqrt(max(0,1-z*z)),0,z])
    denom = 1.0 - Q[2]
    if denom > 1e-9:
        t = 2.0/denom
        print("z=%.12g -> |proj xy|=%.3g" % (z, np.hypot(*(P+t*(Q-P))[:2])))
    else:
        print("z=%.12g -> denom<=1e-9, assert would fire" % z)

# Test 5: disc guaranteed >0 for the SHIPPED constants? print actual disc
print("=== Test 5: shipped-constants disc ===")
n1=unit([0.49,0,-0.87]); n2=unit([-0.49,0,-0.87]); c1=c2=0.4
d=np.cross(n1,n2); G=np.array([[n1@n1,n1@n2],[n2@n1,n2@n2]])
a,b=np.linalg.solve(G,[c1,c2]); p0=a*n1+b*n2
A=d@d;B=2*(p0@d);C=p0@p0-1; print("disc=",B*B-4*A*C)
