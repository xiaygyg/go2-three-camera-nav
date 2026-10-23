"""Hardware-independent RGB-D preprocessing prototype; not a deployed controller.
Inputs are rectified depth in metres, calibrated optical-to-level-body extrinsics,
and a gravity-aligned body frame. Unknown sectors are conservative, never free.
"""
import numpy as np

def deproject_depth(depth_m,intrinsics,T_level_optical,stride=2):
    d=np.asarray(depth_m,float);K=np.asarray(intrinsics,float);T=np.asarray(T_level_optical,float)
    if d.ndim!=2 or K.shape!=(3,3) or T.shape!=(4,4) or stride<1:raise ValueError('Bad depth/K/T/stride shape')
    if not np.isfinite(K).all() or not np.isfinite(T).all() or K[0,0]<=0 or K[1,1]<=0:raise ValueError('Bad calibration')
    if not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4) or np.linalg.det(T[:3,:3])<.99:raise ValueError('Extrinsic must be a rigid transform')
    v,u=np.mgrid[0:d.shape[0]:stride,0:d.shape[1]:stride];z=d[::stride,::stride]
    valid=np.isfinite(z)&(z>.05)&(z<20)
    points=np.stack([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z],-1)[valid]
    return points@T[:3,:3].T+T[:3,3]

def pointcloud_scan(points,coverage_ranges,body_height=.32,robot_radius=.55):
    """Rasterize ground/self-filtered points into the adapter's inflated 36x4 scan.
    coverage_ranges: known free extent per sector from actual valid depth rays.
    A zero coverage range means unknown, including rear blind spots and dropouts.
    Ground is z=-body_height in the level frame; caller must remove robot returns.
    """
    p=np.asarray(points,float).reshape(-1,3);coverage=np.asarray(coverage_ranges,float)
    if coverage.shape!=(36,) or not np.isfinite(coverage).all() or np.any(coverage<0):raise ValueError('Expected 36 finite nonnegative coverage distances')
    observed=coverage>.1;ranges=np.where(observed,np.clip(coverage,.1,4),.1)
    p=p[np.isfinite(p).all(1)]
    above=p[:,2]+body_height;p=p[(above>.12)&(above<1.5)]
    theta=np.arange(36)*np.pi/18;directions=np.stack([np.cos(theta),np.sin(theta)],-1)
    for chunk in np.array_split(p[:,:2],max(1,int(np.ceil(len(p)/2048)))):
        if not len(chunk):continue
        projection=chunk@directions.T;perp2=np.sum(chunk*chunk,1)[:,None]-projection**2
        root=np.sqrt(np.maximum(0,robot_radius**2-perp2))
        hit=np.where((perp2<=robot_radius**2)&(projection+root>=0),np.maximum(.1,projection-root),4)
        ranges=np.minimum(ranges,hit.min(0))
    lidar=np.zeros((1,36,4),np.float32);lidar[0,:,1:]=(4-np.clip(ranges,.1,4))[:,None]
    return lidar,observed,ranges

def fuse_rectified_depth(frames,body_height=.32,stride=2):
    """frames are dict(depth_m,K,T_level_optical); geometric valid-ray coverage.
    This conservative prototype uses a narrow central image band to establish
    free extent, not the nominal camera FOV. It must be validated/calibrated on
    hardware; it does not model cliffs, moving objects, latency or self masking.
    """
    clouds=[];coverage=np.zeros(36)
    for frame in frames:
        depth=np.asarray(frame['depth_m']);K=np.asarray(frame['K']);T=np.asarray(frame['T_level_optical'])
        clouds.append(deproject_depth(depth,K,T,stride))
        band=depth.copy();vv=np.arange(len(depth))[:,None]
        band[np.broadcast_to(np.abs(vv-K[1,2])>max(2,len(depth)*.025),depth.shape)]=0
        ray_ends=deproject_depth(band,K,T,stride)
        if not len(ray_ends):continue
        azimuth=np.arctan2(ray_ends[:,1],ray_ends[:,0]);ids=np.rint(azimuth/(np.pi/18)).astype(int)%36
        distances=np.linalg.norm(ray_ends[:,:2],axis=1)
        # Require several returns and retain conservative free extent.
        for i in np.unique(ids):
            values=distances[ids==i]
            if len(values)>=3:coverage[i]=max(coverage[i],min(4.,max(0.,float(values.min())-.55)))
    points=np.concatenate(clouds) if clouds else np.empty((0,3))
    lidar,observed,ranges=pointcloud_scan(points,coverage,body_height)
    return {'lidar':lidar,'observed':observed,'ranges':ranges,'points':points,'deployment_validated':False}

def pack_navigation_state(goal_body_xy,velocity_body,omega_body,last_command,height):
    goal=np.asarray(goal_body_xy,float);distance=float(np.linalg.norm(goal))
    state=np.zeros(15,np.float32);state[:2]=goal/max(distance,1e-6);state[3]=min(distance,20)
    state[5:8]=velocity_body;state[8:11]=np.asarray(omega_body)*.25
    state[11:14]=np.asarray(last_command)/[1.7,.8,1.];state[14]=(height-.32)*2
    if not np.isfinite(state).all():raise ValueError('Non-finite navigation state')
    return state
