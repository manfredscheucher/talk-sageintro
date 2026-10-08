# Durable check: reproduce g + pos (Tutte), then run the exact notebook TSP cell.
s = ":_gBca`GH`GI`HIdGeHfIbKOcJMaLNbMcNaOfPSdQTeRUdSeTfU_WYZ_VY[_XZ["
g = Graph(s)

def tutte_layout(G,outer_face,weights):
	V = G.vertices(); pos = dict(); l = len(outer_face); a0 = pi/l+pi/2
	for i in range(l):
		ai = a0+pi*2*i/l; pos[outer_face[i]] = (cos(ai),sin(ai))
	n = len(V); M = zero_matrix(RR,n,n); b = zero_matrix(RR,n,2)
	for i in range(n):
		v = V[i]
		if v in pos:
			M[i,i]=1; b[i,0]=pos[v][0]; b[i,1]=pos[v][1]
		else:
			nv=G.neighbors(v); sm=0
			for u in nv:
				j=V.index(u); wu=weights[u,v]; sm+=wu; M[i,j]=-wu
			M[i,i]=sm
	sol=M.pseudoinverse()*b
	return {V[i]:sol[i] for i in range(n)}

outer_face=[v for v,_ in g.faces()[0]]
weights={(u,v):1 for u,v in g.edges(labels=False) for u,v in [(u,v),(v,u)]}
pos=tutte_layout(g,outer_face,weights)

# --- exact notebook TSP cell below ---
for u,v in g.edges(labels=False):
    d = (vector(pos[u]) - vector(pos[v])).length()
    g.set_edge_label(u, v, d)

tour = g.traveling_salesman_problem(use_edge_labels=True)

plt  = g.plot(pos=pos, vertex_size=10, vertex_labels=0, edge_color='lightgray')
plt += tour.plot(pos=pos, vertex_size=10, vertex_labels=0, edge_color='red', edge_thickness=3)
plt.save('/tmp/tsp_out.png')   # exercise the full plot path without a GUI
print("plot rendered OK; tour is cycle (all deg2):", all(deg==2 for deg in tour.degree()))
