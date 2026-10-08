#!/usr/bin/env python3
"""Probe helper for stereographic.sage: search circle-plane parameters (n1,c1,n2,c2)
that give a balanced figure. Pure numpy, no Sage. Prints, for each candidate:
the two intersection points A,B, the max sphere-z over the four used points
(want it well below +1 so projections stay small) and the max projected radius
(want <= ~3). Durable companion to stereographic.sage (kept on purpose)."""
import numpy as np


def unit(v):
    v = np.array(v, float)
    return v / np.linalg.norm(v)


def frame(n):
    n = unit(n)
    h = np.array([1., 0, 0])
    if abs(h @ n) > 0.9:
        h = np.array([0., 1, 0])
    u = unit(h - (h @ n) * n)
    v = np.cross(n, u)
    return n, u, v


def line_sphere(n1, c1, n2, c2):
    n1 = unit(n1)
    n2 = unit(n2)
    d = np.cross(n1, n2)
    G = np.array([[n1 @ n1, n1 @ n2], [n2 @ n1, n2 @ n2]])
    a, b = np.linalg.solve(G, [c1, c2])
    p0 = a * n1 + b * n2
    A = d @ d
    B = 2 * (p0 @ d)
    C = p0 @ p0 - 1
    disc = B * B - 4 * A * C
    if disc <= 0:
        return None
    sq = np.sqrt(disc)
    return p0 + (-B + sq) / (2 * A) * d, p0 + (-B - sq) / (2 * A) * d


def circ_pt(n, c, th):
    n, u, v = frame(n)
    rho = np.sqrt(1 - c * c)
    return c * n + rho * (np.cos(th) * u + np.sin(th) * v)


P = np.array([0., 0, 1])


def proj(Q):
    Q = np.array(Q, float)
    t = 2 / (1 - Q[2])
    return P + t * (Q - P)


def report(nR, cR, nB, cB, cTh, dTh):
    nR, nB = unit(nR), unit(nB)
    res = line_sphere(nR, cR, nB, cB)
    if res is None:
        print("no intersection", nR, cR, nB, cB)
        return
    A, B = res
    C = circ_pt(nR, cR, cTh)
    D = circ_pt(nB, cB, dTh)
    pts = {'A': A, 'B': B, 'C': C, 'D': D}
    zmax = max(p[2] for p in pts.values())
    projs = {k: proj(v) for k, v in pts.items()}
    rad = max(np.hypot(p[0], p[1]) for p in projs.values())
    ab_sep = np.linalg.norm(A - B)
    print("nR", np.round(nR, 2), "cR", cR, "nB", np.round(nB, 2), "cB", cB)
    print("   A", np.round(A, 2), "B", np.round(B, 2))
    print("   |A-B|", round(ab_sep, 2), "zmax", round(zmax, 2),
          "maxprojrad", round(rad, 2))
    for k in 'ABCD':
        print("     %s'" % k, np.round(projs[k], 2))


if __name__ == "__main__":
    # Circles tilted so normals point UP (+z): the circle planes n.x=c with c>0
    # then sit in the LOWER hemisphere, away from the north pole P=(0,0,1).
    # Idea: tilt normals DOWNWARD (negative z) with positive c. Then the circle
    # planes n.x=c sit on the lower-front of the sphere, and their crossing line
    # dips to NEGATIVE z -> A,B land low, projecting to a small radius.
    cands = [
        (unit([0.6, 0, -0.5]), 0.3, unit([-0.6, 0, -0.5]), 0.3, 0.0, np.pi),
        (unit([0.7, 0, -0.7]), 0.25, unit([-0.7, 0, -0.7]), 0.25, 0.0, np.pi),
        (unit([0.8, 0, -0.6]), 0.3, unit([-0.8, 0, -0.6]), 0.3, 0.0, np.pi),
        (unit([0.7, 0, -0.5]), 0.35, unit([-0.7, 0, -0.5]), 0.35, 0.0, np.pi),
        (unit([0.7, 0.2, -0.6]), 0.3, unit([-0.7, 0.2, -0.6]), 0.3, 0.0, np.pi),
    ]
    for c in cands:
        report(*c)
        print()

    print("=== balanced hand picks (moderate tilt, spread circles) ===")
    for c in [
        (unit([0.3, 0, -1.0]), 0.5, unit([-0.3, 0, -1.0]), 0.5, 0.0, np.pi),
        (unit([0.45, 0, -1.0]), 0.45, unit([-0.45, 0, -1.0]), 0.45, 0.0, np.pi),
        (unit([0.5, 0, -0.9]), 0.4, unit([-0.5, 0, -0.9]), 0.4, 0.0, np.pi),
        # slight asymmetry in c so the two circles differ a touch (prettier)
        (unit([0.4, 0, -1.0]), 0.5, unit([-0.4, 0, -1.0]), 0.42, 0.0, np.pi),
    ]:
        report(*c)
        print()

    print("=== CHOSEN candidate: C/D theta sweep ===")
    nR = unit([0.49, 0, -0.87]); cR = 0.4
    nB = unit([-0.49, 0, -0.87]); cB = 0.4
    A, B = line_sphere(nR, cR, nB, cB)
    print("A", np.round(A, 3), "B", np.round(B, 3))
    for label, n, c in [("C(red)", nR, cR), ("D(blue)", nB, cB)]:
        for th in np.linspace(0, 2 * np.pi, 9)[:-1]:
            Q = circ_pt(n, c, th)
            Qp = proj(Q)
            print("  %s th=%.2f  Q=%s z=%+.2f  Q'=%s r=%.2f"
                  % (label, th, np.round(Q, 2), Q[2], np.round(Qp[:2], 2),
                     np.hypot(*Qp[:2])))

    print("=== grid search (symmetric, minimize maxprojrad, keep roundness) ===")
    best = None
    for az in np.linspace(0.3, 1.0, 8):       # normal x-component
        for nz in np.linspace(-1.0, -0.2, 9):  # normal z-component (down)
            for c in np.linspace(0.15, 0.5, 8):
                nR = unit([az, 0, nz])
                nB = unit([-az, 0, nz])
                res = line_sphere(nR, c, nB, c)
                if res is None:
                    continue
                A, B = res
                if np.linalg.norm(A - B) < 1.0:   # want well-separated A,B
                    continue
                rho = np.sqrt(1 - c * c)
                if rho < 0.65:                     # want visibly round circles
                    continue
                C = circ_pt(nR, c, 0.0)
                D = circ_pt(nB, c, np.pi)
                pts = [A, B, C, D]
                if max(p[2] for p in pts) > 0.9:   # avoid near-P points
                    continue
                rad = max(np.hypot(*proj(p)[:2]) for p in pts)
                if best is None or rad < best[0]:
                    best = (rad, az, nz, c, np.linalg.norm(A - B), rho)
    if best:
        rad, az, nz, c, sep, rho = best
        print("best: az=%.3f nz=%.3f c=%.3f  maxprojrad=%.2f |A-B|=%.2f rho=%.2f"
              % (az, nz, c, rad, sep, rho))
        report(unit([az, 0, nz]), c, unit([-az, 0, nz]), c, 0.0, np.pi)
