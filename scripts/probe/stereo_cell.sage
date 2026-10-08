# notebook-cell candidate: self-contained, ends with fig.show() for interactive viewer
import numpy as np

def unit(v):
    v = np.array(v, float); return v/np.linalg.norm(v)
def ortho_frame(n):
    n = unit(n)
    a = np.array([1.0,0,0]) if abs(n[0])<0.9 else np.array([0,1.0,0])
    u = unit(a - (a@n)*n); v = np.cross(n,u); return n,u,v
def small_circle_point(n,c,theta):
    n,u,v = ortho_frame(n); rho = np.sqrt(1-c*c)
    return c*n + rho*(np.cos(theta)*u + np.sin(theta)*v)
def project(Q):
    Q = np.array(Q,float); P = np.array([0,0,1.0]); t = 2.0/(1.0-Q[2]); return P + t*(Q-P)
def line_sphere(n1,c1,n2,c2):
    n1=unit(n1);n2=unit(n2)
    G = np.array([[n1@n1,n1@n2],[n2@n1,n2@n2]]); ab = np.linalg.solve(G,[c1,c2])
    p0 = ab[0]*n1+ab[1]*n2; d = unit(np.cross(n1,n2))
    b = 2*(p0@d); cc = p0@p0-1; disc = b*b-4*cc
    s1=(-b+np.sqrt(disc))/2; s2=(-b-np.sqrt(disc))/2
    return p0+s1*d, p0+s2*d
def circle3(p1,p2,p3):
    p1,p2,p3=[np.array(p,float) for p in (p1,p2,p3)]
    (x1,y1),(x2,y2),(x3,y3)=p1[:2],p2[:2],p3[:2]
    A=np.array([[x2-x1,y2-y1],[x3-x1,y3-y1]]); rhs=0.5*np.array([x2*x2-x1*x1+y2*y2-y1*y1, x3*x3-x1*x1+y3*y3-y1*y1])
    cx,cy=np.linalg.solve(A,rhs); R=np.hypot(cx-x1,cy-y1); return np.array([cx,cy,-1.0]),R

nR=unit([0.49,0,-0.87]); cR=0.4
nB=unit([-0.49,0,-0.87]); cB=0.4
P=np.array([0,0,1.0])
A3,B3=line_sphere(nR,cR,nB,cB)
C3=small_circle_point(nR,cR,0.0); D3=small_circle_point(nB,cB,np.pi)
Ap,Bp,Cp,Dp=project(A3),project(B3),project(C3),project(D3)
redCenter,redR=circle3(Ap,Bp,Cp); blueCenter,blueR=circle3(Ap,Bp,Dp)

RED=(0.85,0.12,0.12); BLUE=(0.12,0.30,0.85); DARK=(0.08,0.08,0.08)
fig=Graphics()
fig+=sphere((0,0,0),1,color=(0.80,0.85,0.92),opacity=0.18,mesh=False)
S=3.0
fig+=polygon3d([(-S,-S,-1),(S,-S,-1),(S,S,-1),(-S,S,-1)],color=(0.70,0.80,0.95),opacity=0.15)
th=var('th')
def sc(n,c,col):
    nn,u,v=ortho_frame(n); rho=float(np.sqrt(1-c*c)); cn=[float(c)*float(nn[i]) for i in range(3)]
    return parametric_plot3d(tuple(cn[i]+rho*(cos(th)*float(u[i])+sin(th)*float(v[i])) for i in range(3)),(th,0,2*pi),color=col,thickness=6)
fig+=sc(nR,cR,RED); fig+=sc(nB,cB,BLUE)
def pc(center,R,col):
    cx,cy,cz=float(center[0]),float(center[1]),float(center[2]); R=float(R)
    return parametric_plot3d((cx+R*cos(th),cy+R*sin(th),cz),(th,0,2*pi),color=col,thickness=6)
fig+=pc(redCenter,redR,RED); fig+=pc(blueCenter,blueR,BLUE)
for Qp in [Ap,Bp,Cp,Dp]:
    fig+=line3d([tuple(P),tuple(Qp)],color=(0.55,0.55,0.55),thickness=1)
def pt(coord,col,name,off):
    g=point3d(tuple(coord),size=16,color=col)
    g+=text3d(name,tuple(np.array(coord,float)+np.array(off)),color=DARK,fontsize=16)
    return g
fig+=pt(P,DARK,"P",(0.10,0,0.16)); fig+=pt(A3,DARK,"A",(0,0.16,0.12)); fig+=pt(B3,DARK,"B",(0,-0.20,0.12))
fig+=pt(C3,RED,"C",(0.16,0,0.12)); fig+=pt(D3,BLUE,"D",(-0.22,0,0.12))
fig+=pt(Ap,DARK,"A'",(0.12,0.18,0)); fig+=pt(Bp,DARK,"B'",(0.12,-0.24,0))
fig+=pt(Cp,RED,"C'",(0.20,0,0)); fig+=pt(Dp,BLUE,"D'",(-0.28,0,0))
# Default orientation: elevated 3/4 view looking down onto the z=-1 plane.
# Applied to fig itself so BOTH the interactive Three.js viewer (fig.show())
# and any static render start from this angle. The scene's symmetry plane is
# x=0; rotateZ swings it off-axis (no axis-on collapse), rotateX tilts the
# camera down so the plane reads as a floor below the sphere.
fig = fig.rotateZ(35*pi/180).rotateX(-65*pi/180)
# in the notebook: fig.show()  (interactive Three.js viewer)
fig.save('/tmp/stereo_cell.png', frame=True, figsize=[7,7])   # headless proof it builds
print("cell builds OK")
