# Adversarial review repro: simulate notebook top-to-bottom for the g/g2/pos
# interaction (dual-graph -> tutte-code -> iterated-Tutte -> euclidean-tsp).
s = ":_gBca`GH`GI`HIdGeHfIbKOcJMaLNbMcNaOfPSdQTeRUdSeTfU_WYZ_VY[_XZ["
g = Graph(s)
print("g vertices:", g.order())

def tutte_layout(G, outer_face, weights):
    V = G.vertices()
    pos = dict()
    l = len(outer_face)
    a0 = pi/l + pi/2
    for i in range(l):
        ai = a0 + pi*2*i/l
        pos[outer_face[i]] = (cos(ai), sin(ai))
    n = len(V)
    M = zero_matrix(RR, n, n)
    b = zero_matrix(RR, n, 2)
    for i in range(n):
        v = V[i]
        if v in pos:
            M[i, i] = 1
            b[i, 0] = pos[v][0]
            b[i, 1] = pos[v][1]
        else:
            nv = G.neighbors(v)
            ss = 0
            for u in nv:
                j = V.index(u)
                wu = weights[u, v]
                ss += wu
                M[i, j] = -wu
            M[i, i] = ss
    sol = M.pseudoinverse()*b
    return {V[i]: sol[i] for i in range(n)}

outer_face = [v for v, _ in g.faces()[0]]
weights = {(u, v): 1 for u, v in g.edges(labels=False) for u, v in [(u, v), (v, u)]}
pos = tutte_layout(g, outer_face, weights)
print("tutte-code cell: pos covers g:", set(pos.keys()) == set(g.vertices()))

# iterated-Tutte cell defines g2 and REASSIGNS pos to g2's layout.
g2 = Graph(":~?@I`_GD`WkI`WWK_O_Ic?}J_?eOd_[KcWmK`o}CEGKT`gKHDWGLEXUSDpe[FQAOhOQGA`_c`qIQeqY??OaPFqEGAQQPEHyA_oOmehWXea?fcagk_QMSDrUDFQ]NCq_pap]HHAy[FQSocP{nc@YFBruTIBCr`AOhJiCqNXG]NwOn_`sdOxklNCM?Cqa@AbWxc@OtOW?@CqN")
print("g2 vertices:", g2.order())
outer_face = [v for v, _ in g2.faces()[0]]
weights = {}
for u, v in g2.edges(labels=False):
    weights[u, v] = weights[v, u] = 1.0
pos = tutte_layout(g2, outer_face, weights)
print("iterated-Tutte cell: pos now covers g2:", set(pos.keys()) == set(g2.vertices()))
print("does g2-pos cover g's vertices?:", set(g.vertices()).issubset(set(pos.keys())))

# euclidean-tsp cell uses g with pos (which is now g2's layout!)
try:
    for u, v in g.edges(labels=False):
        d = (vector(pos[u]) - vector(pos[v])).length()
        g.set_edge_label(u, v, d)
    print("TSP labeling with stale pos: NO crash, but uses WRONG graph's coordinates")
except KeyError as e:
    print("TSP crashes with KeyError:", e)
