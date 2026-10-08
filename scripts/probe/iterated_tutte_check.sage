# Iterated Tutte on its OWN graph g2 (independent of g used by TSP), 12 iterations, 4/row grid.
def tutte_layout(G,outer_face,weights):
	V=G.vertices();pos=dict();l=len(outer_face);a0=pi/l+pi/2
	for i in range(l):
		ai=a0+pi*2*i/l;pos[outer_face[i]]=(cos(ai),sin(ai))
	n=len(V);M=zero_matrix(RR,n,n);b=zero_matrix(RR,n,2)
	for i in range(n):
		v=V[i]
		if v in pos: M[i,i]=1;b[i,0]=pos[v][0];b[i,1]=pos[v][1]
		else:
			nv=G.neighbors(v);sm=0
			for u in nv:
				j=V.index(u);wu=weights[u,v];sm+=wu;M[i,j]=-wu
			M[i,i]=sm
	sol=M.pseudoinverse()*b
	return {V[i]:sol[i] for i in range(n)}
from scipy.spatial import ConvexHull

g2 = Graph(":_gBca`GH`GI`HIdGeHfIbKOcJMaLNbMcNaOfPSdQTeRUdSeTfU_WYZ_VY[_XZ[")
outer_face = [v for v,_ in g2.faces()[0]]

weights={}
for u,v in g2.edges(labels=False): weights[u,v]=weights[v,u]=1.0
pos=tutte_layout(g2,outer_face,weights)
frames=[dict(pos)]
eps=0.1
for it in range(2,13):
    dist2=lambda u,v:(pos[u][0]-pos[v][0])^2+(pos[u][1]-pos[v][1])^2
    old=weights;weights={}
    for u,v in g2.edges(labels=False):
        w=RR(old[u,v]+eps*(dist2(u,v)-old[u,v]));weights[u,v]=weights[v,u]=w
    for f in g2.faces():
        verts=[e[0] for e in f]
        try:
            area=RR(ConvexHull([pos[w] for w in verts]).volume)
            for a in range(len(verts)):
                u,v=verts[a],verts[(a+1)%len(verts)]
                add=float(it*area)^4
                weights[u,v]=weights.get((u,v),0)+add;weights[v,u]=weights.get((v,u),0)+add
        except Exception: pass
    pos=tutte_layout(g2,outer_face,weights);frames.append(dict(pos))

import matplotlib;matplotlib.use("Agg")
plots=[g2.plot(pos=fr,vertex_size=8,vertex_labels=0,title=f"iteration {k}") for k,fr in enumerate(frames,1)]
graphics_array(plots, ncols=4).save("/tmp/g2_grid.png", figsize=12)
print("OK frames",len(frames),"rows",(len(frames)+3)//4)
