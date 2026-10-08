#!/usr/bin/env python3
"""Pure-numpy verification of the stereographic.sage math (no Sage needed).
Mirrors the geometry + the final assertions of stereographic.sage so the
construction can be checked even when Sage's renderer is unavailable.
Durable regression check (kept on purpose)."""
import numpy as np


def unit(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


def ortho_frame(n):
    n = unit(n)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, n)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = unit(helper - np.dot(helper, n) * n)
    v = np.cross(n, u)
    return n, u, v


def small_circle_point(n, c, theta):
    n, u, v = ortho_frame(n)
    rho = np.sqrt(1.0 - c * c)
    return c * np.array(n) + rho * (np.cos(theta) * u + np.sin(theta) * v)


P = np.array([0.0, 0.0, 1.0])


def project_to_plane(Q):
    Q = np.array(Q, float)
    denom = 1.0 - Q[2]
    assert denom > 1e-9
    return P + (2.0 / denom) * (Q - P)


def line_sphere_intersection(n1, c1, n2, c2):
    n1 = unit(n1); n2 = unit(n2)
    d = np.cross(n1, n2)
    G = np.array([[n1 @ n1, n1 @ n2], [n2 @ n1, n2 @ n2]])
    a, b = np.linalg.solve(G, [c1, c2])
    p0 = a * n1 + b * n2
    A = d @ d; B = 2.0 * (p0 @ d); C = p0 @ p0 - 1.0
    disc = B * B - 4 * A * C
    assert disc > 1e-9
    sq = np.sqrt(disc)
    return p0 + (-B + sq) / (2 * A) * d, p0 + (-B - sq) / (2 * A) * d


def planar_circle_through_3(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1[:2], p2[:2], p3[:2]
    A = np.array([[2 * (x2 - x1), 2 * (y2 - y1)],
                  [2 * (x3 - x1), 2 * (y3 - y1)]])
    rhs = np.array([(x2**2 - x1**2) + (y2**2 - y1**2),
                    (x3**2 - x1**2) + (y3**2 - y1**2)])
    cx, cy = np.linalg.solve(A, rhs)
    return np.array([cx, cy, -1.0]), np.hypot(x1 - cx, y1 - cy)


nR = unit([0.49, 0.0, -0.87]); cR = 0.4
nB = unit([-0.49, 0.0, -0.87]); cB = 0.4

assert abs(nR @ P - cR) > 1e-3
assert abs(nB @ P - cB) > 1e-3

A3, B3 = line_sphere_intersection(nR, cR, nB, cB)
C3 = small_circle_point(nR, cR, 0.0)
D3 = small_circle_point(nB, cB, np.pi)
Ap = project_to_plane(A3); Bp = project_to_plane(B3)
Cp = project_to_plane(C3); Dp = project_to_plane(D3)
redCenter, redR = planar_circle_through_3(Ap, Bp, Cp)
blueCenter, blueR = planar_circle_through_3(Ap, Bp, Dp)


def fmt(v):
    return "(" + ", ".join("%+.4f" % x for x in v) + ")"


for name, X in [("P", P), ("A", A3), ("B", B3), ("C", C3), ("D", D3),
                ("A'", Ap), ("B'", Bp), ("C'", Cp), ("D'", Dp)]:
    print("  %-3s %s" % (name, fmt(X)))
print("red  center=%s R=%.4f" % (fmt(redCenter), redR))
print("blue center=%s R=%.4f" % (fmt(blueCenter), blueR))
print("max projected radius = %.3f"
      % max(np.hypot(*q[:2]) for q in [Ap, Bp, Cp, Dp]))
print("sphere-z of used points: A=%.2f B=%.2f C=%.2f D=%.2f"
      % (A3[2], B3[2], C3[2], D3[2]))

TOL = 1e-6
for name, X in [("A", A3), ("B", B3)]:
    assert abs(np.linalg.norm(X) - 1.0) < TOL
    assert abs(nR @ X - cR) < TOL and abs(nB @ X - cB) < TOL
assert abs(np.linalg.norm(C3) - 1.0) < TOL and abs(nR @ C3 - cR) < TOL
assert abs(np.linalg.norm(D3) - 1.0) < TOL and abs(nB @ D3 - cB) < TOL
for X in [Ap, Bp, Cp, Dp]:
    assert abs(X[2] + 1.0) < TOL
for X in [Ap, Bp, Cp]:
    assert abs(np.hypot(X[0] - redCenter[0], X[1] - redCenter[1]) - redR) < TOL
for X in [Ap, Bp, Dp]:
    assert abs(np.hypot(X[0] - blueCenter[0], X[1] - blueCenter[1]) - blueR) < TOL
print("All assertions passed (pure-numpy mirror).")
