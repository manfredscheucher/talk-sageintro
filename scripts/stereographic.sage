# Durable figure for the talk: stereographic projection of two small circles.
#
# Keyword: "Stereographic projection"
# Reference: https://en.wikipedia.org/wiki/Stereographic_projection
#
# Scene (analogous to a GeoGebra construction):
#   - Unit sphere at origin, radius 1, drawn transparent.
#   - Tangent plane z = -1 at the south pole, drawn transparent.
#   - Projection center P = north pole (0,0,1).
#   - Two small circles on the sphere (sphere cap boundaries, i.e. sphere n plane),
#     neither through P, intersecting each other in two points A, B.
#   - One extra point C on the red circle, one extra point D on the blue circle.
#   - Rays from P through A,B,C,D hit z=-1 at A',B',C',D'.
#   - The red projected circle through A',B',C'; the blue through A',B',D'.
#
# Math is derived and then verified numerically (asserts at the end). No hand-waving.
#
# Run:  sage scripts/stereographic.sage      (renders /tmp/stereographic.png headless)

import numpy as np

# ---------------------------------------------------------------------------
# Geometry helpers (plain numpy; Sage symbolic not needed for the derivation)
# ---------------------------------------------------------------------------

def unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)

def ortho_frame(n):
    """Return orthonormal (n, u, v) with u, v spanning the plane normal to n."""
    n = unit(n)
    # pick a helper not parallel to n
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, n)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = unit(helper - np.dot(helper, n) * n)
    v = np.cross(n, u)
    return n, u, v

def small_circle_point(n, c, theta):
    """Point on the small circle {x : |x|=1, n.x = c} at parameter theta.

    Parametrisation: x = c*n + sqrt(1-c^2) * (cos th * u + sin th * v).
    With (n,u,v) orthonormal and |c|<1 this lands exactly on the unit sphere:
    |x|^2 = c^2 + (1-c^2) = 1, and n.x = c. (angular radius = arccos c.)
    """
    n, u, v = ortho_frame(n)
    rho = np.sqrt(1.0 - c * c)
    return c * np.array(n) + rho * (np.cos(theta) * u + np.sin(theta) * v)

P = np.array([0.0, 0.0, 1.0])  # projection center = north pole

def project_to_plane(Q):
    """Stereographic projection of sphere point Q from P onto z = -1.

    Line P + t (Q - P) meets z = -1 when 1 + t (Qz - 1) = -1  =>  t = 2/(1 - Qz).
    Requires Qz < 1 (Q != P), guaranteed since our circles avoid P.
    """
    Q = np.array(Q, dtype=float)
    denom = 1.0 - Q[2]
    assert denom > 1e-9, "point coincides with projection center P"
    t = 2.0 / denom
    return P + t * (Q - P)

def line_sphere_intersection(n1, c1, n2, c2):
    """Two points where the circle-planes' common line meets the unit sphere.

    The two planes n1.x = c1, n2.x = c2 intersect in a line L = p0 + s*dir.
    dir = n1 x n2.  A particular point p0 solves both plane equations; take the
    least-norm solution of the 2x3 system via the normal equations restricted to
    span(n1,n2):  p0 = a*n1 + b*n2 with [n1.n1 n1.n2; n2.n1 n2.n2][a;b] = [c1;c2].
    Then substitute into |x|^2 = 1 -> quadratic in s, solve for the two roots.
    """
    n1 = unit(n1); n2 = unit(n2)
    d = np.cross(n1, n2)
    assert np.linalg.norm(d) > 1e-9, "circle planes are parallel"
    G = np.array([[n1 @ n1, n1 @ n2], [n2 @ n1, n2 @ n2]])
    rhs = np.array([c1, c2])
    a, b = np.linalg.solve(G, rhs)
    p0 = a * n1 + b * n2
    # |p0 + s d|^2 = 1  ->  (d.d) s^2 + 2 (p0.d) s + (p0.p0 - 1) = 0
    A = d @ d
    B = 2.0 * (p0 @ d)
    C = p0 @ p0 - 1.0
    disc = B * B - 4 * A * C
    assert disc > 1e-9, "circles do not intersect (no two real points)"
    sq = np.sqrt(disc)
    s1 = (-B + sq) / (2 * A)
    s2 = (-B - sq) / (2 * A)
    return p0 + s1 * d, p0 + s2 * d

def planar_circle_through_3(p1, p2, p3):
    """Center (in the z=-1 plane) and radius of the circle through three
    coplanar points (all with z = -1). Solve in (x,y) via the linear system from
    |(x,y)-(cx,cy)|^2 = R^2 differences (perpendicular-bisector equations)."""
    (x1, y1), (x2, y2), (x3, y3) = p1[:2], p2[:2], p3[:2]
    # (x-cx)^2+(y-cy)^2 = R^2 ; subtract pairs to kill the quadratic/R^2 terms:
    #   2(x2-x1)cx + 2(y2-y1)cy = (x2^2-x1^2)+(y2^2-y1^2)
    #   2(x3-x1)cx + 2(y3-y1)cy = (x3^2-x1^2)+(y3^2-y1^2)
    A = np.array([[2 * (x2 - x1), 2 * (y2 - y1)],
                  [2 * (x3 - x1), 2 * (y3 - y1)]])
    rhs = np.array([(x2**2 - x1**2) + (y2**2 - y1**2),
                    (x3**2 - x1**2) + (y3**2 - y1**2)])
    cx, cy = np.linalg.solve(A, rhs)
    R = np.hypot(x1 - cx, y1 - cy)
    return np.array([cx, cy, -1.0]), R

# ---------------------------------------------------------------------------
# Choose the two small circles (planes chosen so they overlap and avoid P)
# ---------------------------------------------------------------------------

# Red circle plane and blue circle plane.
#
# Design (see stereographic_probe.py for the parameter search): both normals
# point mostly DOWNWARD (negative z) with a positive offset c=0.4, so the two
# small circles sit on the lower-front of the sphere, well away from the north
# pole P=(0,0,1). They are mirror images across the x=0 plane, so they cross in
# two well-separated points A=(0,+0.89,-0.46), B=(0,-0.89,-0.46). The angular
# radius is large (rho=sqrt(1-c^2)=0.92) so each circle is visibly round, and no
# used point climbs above z=+0.1 -> every projection stays within radius ~2.2.
nR = unit([0.49, 0.0, -0.87]); cR = 0.4     # red:  nR . x = cR
nB = unit([-0.49, 0.0, -0.87]); cB = 0.4    # blue: nB . x = cB

# Neither circle may pass through P=(0,0,1): that happens iff nR.P == cR.
assert abs(nR @ P - cR) > 1e-3, "red circle passes through P"
assert abs(nB @ P - cB) > 1e-3, "blue circle passes through P"

# Intersection points A, B of the two sphere-circles.
A3, B3 = line_sphere_intersection(nR, cR, nB, cB)

# Extra points: C on red, D on blue. theta=0 is the circle's apex pointing away
# from the A-B axis, i.e. the "front" point farthest from both A and B -> C and D
# are nicely spread (C at x=+0.99, D at x=-0.99, both at z=+0.1), symmetric and
# clearly distinct from A,B.
C3 = small_circle_point(nR, cR, 0.0)        # front apex of red circle
D3 = small_circle_point(nB, cB, np.pi)      # front apex of blue circle

# Project all four onto the plane z=-1.
Ap = project_to_plane(A3)
Bp = project_to_plane(B3)
Cp = project_to_plane(C3)
Dp = project_to_plane(D3)

# Projected circles (red through A',B',C'; blue through A',B',D').
redCenter, redR = planar_circle_through_3(Ap, Bp, Cp)
blueCenter, blueR = planar_circle_through_3(Ap, Bp, Dp)

# ---------------------------------------------------------------------------
# Build the Sage 3D figure
# ---------------------------------------------------------------------------

RED = (0.85, 0.12, 0.12)
BLUE = (0.12, 0.30, 0.85)
DARK = (0.08, 0.08, 0.08)   # points / labels

fig = Graphics()

# Transparent unit sphere: light, smooth, low opacity so it reads as "sphere"
# but never hides the circles/points inside and in front of it.
fig += sphere((0, 0, 0), 1, color=(1.0, 0.90, 0.15), opacity=0.30, mesh=False, aspect_ratio=1)

# Tangent plane z = -1: a subtle shaded square big enough to contain every
# primed point (max projected radius ~2.2 -> half-width 3.0 is comfortable).
S = 3.0
fig += polygon3d([(-S, -S, -1), (S, -S, -1), (S, S, -1), (-S, S, -1)],
                 color=(0.65, 0.78, 0.95), opacity=0.28)

th = var('th')

def sphere_circle_plot(n, c, col):
    # Build a Sage-symbolic parametrisation in th from the precomputed frame,
    # so parametric_plot3d gets pure symbolic expressions (robust, no lambdas).
    nn, u, v = ortho_frame(n)
    rho = float(np.sqrt(1.0 - c * c))
    cn = [float(c) * float(nn[i]) for i in range(3)]
    expr = tuple(cn[i] + rho * (cos(th) * float(u[i]) + sin(th) * float(v[i]))
                 for i in range(3))
    return parametric_plot3d(expr, (th, 0, 2 * pi), color=col, thickness=3)

fig += sphere_circle_plot(nR, cR, RED)    # red circle on sphere
fig += sphere_circle_plot(nB, cB, BLUE)   # blue circle on sphere

def planar_circle_plot(center, R, col):
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    R = float(R)
    expr = (cx + R * cos(th), cy + R * sin(th), cz)
    return parametric_plot3d(expr, (th, 0, 2 * pi), color=col, thickness=3)

fig += planar_circle_plot(redCenter, redR, RED)    # projected red circle
fig += planar_circle_plot(blueCenter, blueR, BLUE)  # projected blue circle

# Thin dashed gray rays from P through each sphere point to its projected hit
# point. Thin + light so the construction lines never dominate the picture.
for Q, Qp in [(A3, Ap), (B3, Bp), (C3, Cp), (D3, Dp)]:
    fig += line3d([tuple(P), tuple(Qp)], color=(0.55, 0.55, 0.55),
                  thickness=4)

# Points and labels. The label sits a little above/beside the dot (offset along
# a direction that keeps it off the dot) in a dark readable color.
def pt(coord, col, name, off=(0.12, 0.0, 0.14)):
    g = point3d(tuple(coord), size=26, color=col)
    g += text3d(name, tuple(np.array(coord, float) + np.array(off)),
                color=DARK, fontsize=26)
    return g

# A, B are shared by both circles -> neutral dark. C red, D blue. P black.
fig += pt(P,  DARK, "P",  off=(0.10, 0.0, 0.16))
fig += pt(A3, DARK, "A",  off=(0.0, 0.16, 0.12))
fig += pt(B3, DARK, "B",  off=(0.0, -0.20, 0.12))
fig += pt(C3, RED,  "C",  off=(0.16, 0.0, 0.12))
fig += pt(D3, BLUE, "D",  off=(-0.22, 0.0, 0.12))
fig += pt(Ap, DARK, "A'", off=(0.12, 0.18, 0.0))
fig += pt(Bp, DARK, "B'", off=(0.12, -0.24, 0.0))
fig += pt(Cp, RED,  "C'", off=(0.20, 0.0, 0.0))
fig += pt(Dp, BLUE, "D'", off=(-0.28, 0.0, 0.0))

# ---------------------------------------------------------------------------
# Render headless. Try PNG (Tachyon raytracer); fall back to HTML.
# ---------------------------------------------------------------------------

# Orient the scene with an explicit camera and crop tight. For a static (Tachyon)
# PNG the reliable kwargs are frame / figsize; `viewpoint` only steers the
# interactive Three.js viewer, so instead we rotate the whole figure into a
# slightly elevated 3/4 view (sphere up top, plane z=-1 below, rays connecting)
# and let Tachyon's default camera look down the resulting y-axis.
scene = fig.rotateZ(-0.9).rotateX(-1.15)   # yaw then tilt down a little

# Try the nice settings first; if any kwarg is rejected, fall back to a minimal
# PNG, and only then to HTML -- so a PNG is produced no matter what.
rendered = None
for attempt, kwargs in [("PNG", dict(frame=False, figsize=8)),
                        ("PNG", dict(figsize=8)),
                        ("PNG", dict())]:
    try:
        scene.save('/tmp/stereographic.png', **kwargs)
        rendered = attempt
        break
    except Exception as e:
        print("PNG save failed (%r):" % kwargs, e)
if rendered is None:
    scene.save('/tmp/stereographic.html')
    rendered = 'HTML'
print("RENDERED:", rendered)

# ---------------------------------------------------------------------------
# Print coordinates and verify everything.
# ---------------------------------------------------------------------------

def fmt(v):
    return "(" + ", ".join("%+.4f" % x for x in v) + ")"

print("\nCoordinates:")
print("  P  =", fmt(P))
print("  A  =", fmt(A3))
print("  B  =", fmt(B3))
print("  C  =", fmt(C3))
print("  D  =", fmt(D3))
print("  A' =", fmt(Ap))
print("  B' =", fmt(Bp))
print("  C' =", fmt(Cp))
print("  D' =", fmt(Dp))
print("\nProjected circles (in z=-1 plane):")
print("  red  center=%s R=%.4f" % (fmt(redCenter), redR))
print("  blue center=%s R=%.4f" % (fmt(blueCenter), blueR))

TOL = 1e-6

# A, B lie on the unit sphere.
for name, X in [("A", A3), ("B", B3)]:
    assert abs(np.linalg.norm(X) - 1.0) < TOL, "%s not on unit sphere" % name

# A, B lie on BOTH sphere-circles (both plane equations).
for name, X in [("A", A3), ("B", B3)]:
    assert abs(nR @ X - cR) < TOL, "%s not on red circle" % name
    assert abs(nB @ X - cB) < TOL, "%s not on blue circle" % name

# C on red circle & sphere, D on blue circle & sphere.
assert abs(np.linalg.norm(C3) - 1.0) < TOL and abs(nR @ C3 - cR) < TOL, "C bad"
assert abs(np.linalg.norm(D3) - 1.0) < TOL and abs(nB @ D3 - cB) < TOL, "D bad"

# Every primed point has z = -1.
for name, X in [("A'", Ap), ("B'", Bp), ("C'", Cp), ("D'", Dp)]:
    assert abs(X[2] - (-1.0)) < TOL, "%s not on plane z=-1" % name

# Projected points actually lie on their projected circles.
for name, X in [("A'", Ap), ("B'", Bp), ("C'", Cp)]:
    assert abs(np.hypot(X[0] - redCenter[0], X[1] - redCenter[1]) - redR) < TOL, \
        "%s not on red projected circle" % name
for name, X in [("A'", Ap), ("B'", Bp), ("D'", Dp)]:
    assert abs(np.hypot(X[0] - blueCenter[0], X[1] - blueCenter[1]) - blueR) < TOL, \
        "%s not on blue projected circle" % name

print("\nAll assertions passed.")
