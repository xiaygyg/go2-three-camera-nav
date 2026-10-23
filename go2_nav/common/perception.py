"""Planar range sensing and shape-aware nearby obstacle proxies."""
import numpy as np
import mujoco

def circle_ray(origin,direction,center,radius):
    offset=center-origin;proj=offset@direction;perp2=offset@offset-proj*proj
    if perp2>radius*radius:return np.inf
    root=np.sqrt(max(0,radius*radius-perp2))
    return max(.1,proj-root) if proj+root>=0 else np.inf

def box_ray(origin,direction,center,rotation,half):
    o=rotation.T@(origin-center);d=rotation.T@direction;lo=-np.inf;hi=np.inf
    for i in range(2):
        if abs(d[i])<1e-9:
            if abs(o[i])>half[i]:return np.inf
        else:
            a,b=(-half[i]-o[i])/d[i],(half[i]-o[i])/d[i];lo=max(lo,min(a,b));hi=min(hi,max(a,b))
    return max(.1,lo) if hi>=max(0,lo) else np.inf

def nearest_segment(position,a,b):
    delta=b-a;return a+np.clip((position-a)@delta/max(delta@delta,1e-9),0,1)*delta

def scan_and_obstacles(model,data,position,reference,inflation,velocities):
    position=np.asarray(position)[:2];obstacles=[];shapes=[]
    for gid in range(model.ngeom):
        if model.geom_group[gid]!=3:continue
        center=data.geom_xpos[gid,:2].copy();size=model.geom_size[gid];kind=model.geom_type[gid];rotation=data.geom_xmat[gid].reshape(3,3)
        if kind==mujoco.mjtGeom.mjGEOM_BOX:
            R=rotation[:2,:2];local=R.T@(position-center);nearest=center+R@np.clip(local,-size[:2],size[:2]);radius=0.;bound=np.linalg.norm(size[:2]);shape=('box',center,R,size[:2]+inflation)
        elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis=rotation[:2,2];a=center-axis*size[1];b=center+axis*size[1];nearest=nearest_segment(position,a,b);radius=float(size[0]);bound=size[0]+size[1];shape=('capsule',a,b,radius+inflation)
        else:
            nearest=center;radius=float(max(size[:2]) if kind==mujoco.mjtGeom.mjGEOM_ELLIPSOID else size[0]);bound=radius;shape=('circle',center,radius+inflation)
        if np.linalg.norm(center-position)-bound>4.:continue
        shapes.append(shape)
        if np.linalg.norm(nearest-position)-radius<4.:obstacles.append((nearest,radius,np.asarray(velocities.get(gid,[0.,0.]))))
    theta=np.arctan2(reference[1],reference[0]);ranges=np.full((36,4),4.)
    for i in range(36):
        angle=theta+i*np.pi/18;direction=np.array([np.cos(angle),np.sin(angle)]);distance=4.
        for shape in shapes:
            if shape[0]=='circle':hit=circle_ray(position,direction,shape[1],shape[2])
            elif shape[0]=='box':hit=box_ray(position,direction,*shape[1:])
            else:
                a,b,radius=shape[1:];axis=b-a;length=np.linalg.norm(axis);axis=axis/max(length,1e-9);R=np.column_stack([axis,[-axis[1],axis[0]]])
                hit=min(circle_ray(position,direction,a,radius),circle_ray(position,direction,b,radius),box_ray(position,direction,(a+b)/2,R,np.array([length/2,radius])))
            distance=min(distance,hit)
        ranges[i,1:]=np.clip(distance,.1,4.)
    return (4.-ranges).astype(np.float32)[None,None],obstacles

def dynamic_input(model,data,position,reference,velocities):
    out=np.zeros((1,1,5,10),np.float32);theta=np.arctan2(reference[1],reference[0]);c,s=np.cos(theta),np.sin(theta);R=np.array([[c,s],[-s,c]])
    nearby=sorted(velocities,key=lambda gid:np.linalg.norm(data.geom_xpos[gid,:2]-position[:2]))
    k=0
    for gid in nearby:
        delta=data.geom_xpos[gid,:2]-position[:2];distance=np.linalg.norm(delta)
        if distance>=4. or k==5:continue
        state=out[0,0,k];state[:2]=R@delta/max(distance,1e-6);state[3]=distance;state[5:7]=R@np.asarray(velocities[gid]);state[8]=2*model.geom_size[gid,0]/.25-1;state[9]=0;k+=1
    return out
