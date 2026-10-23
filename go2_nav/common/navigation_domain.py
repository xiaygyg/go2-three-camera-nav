"""Offline connected goal/spawn domain. Never exposed to the navigation actor."""
from collections import deque
import numpy as np
import mujoco

def connected_domain(model,data,radius=.65,resolution=.1,bounds=(-.5,13.5,-4.2,4.2)):
    xs=np.arange(bounds[0],bounds[1]+resolution*.1,resolution);ys=np.arange(bounds[2],bounds[3]+resolution*.1,resolution);X,Y=np.meshgrid(xs,ys);points=np.stack([X,Y],-1);clearance=np.full(X.shape,100.)
    for gid in np.flatnonzero(model.geom_group==3):
        if 'dynamic' in model.geom(gid).name:continue
        c=data.geom_xpos[gid,:2];size=model.geom_size[gid];R=data.geom_xmat[gid].reshape(3,3)
        if model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX:
            local=(points-c)@R[:2,:2];gap=np.linalg.norm(local-np.clip(local,-size[:2],size[:2]),axis=-1)
        elif model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis=R[:2,2];a=c-axis*size[1];b=c+axis*size[1];line=b-a;closest=a+np.clip(np.sum((points-a)*line,-1)/max(line@line,1e-8),0,1)[...,None]*line;gap=np.linalg.norm(points-closest,axis=-1)-size[0]
        else:gap=np.linalg.norm(points-c,axis=-1)-size[0]
        clearance=np.minimum(clearance,gap)
    free=clearance>=radius;connected=np.zeros_like(free);start=(int(np.argmin(abs(ys))),int(np.argmin(abs(xs))));queue=deque([start]);connected[start]=bool(free[start])
    while queue:
        yy,xx=queue.popleft()
        for dy,dx in [(0,1),(0,-1),(1,0),(-1,0)]:
            y,x=yy+dy,xx+dx
            if 0<=y<len(ys) and 0<=x<len(xs) and free[y,x] and not connected[y,x]:connected[y,x]=True;queue.append((y,x))
    return xs,ys,connected,clearance

def contains(domain,point):
    xs,ys,mask,_=domain
    if not (xs[0]<=point[0]<=xs[-1] and ys[0]<=point[1]<=ys[-1]):return False
    return bool(mask[int(np.argmin(abs(ys-point[1]))),int(np.argmin(abs(xs-point[0])))])
