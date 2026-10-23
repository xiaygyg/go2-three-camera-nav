"""Three real rendered depth images -> blind-aware scan. No geometry-query sensing."""
import copy,json,math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .camera_observation import deproject_depth,pointcloud_scan

ROOT=Path(__file__).resolve().parents[2]
OPTICAL_TO_MJ=np.diag([1.,-1.,-1.])
FRONT_OPTICAL=np.array([[0,0,1],[-1,0,0],[0,-1,0.]])

def load_config(config=None):
    if config is None:config=ROOT/'configs/go2_three_d435i.json'
    c=json.loads(Path(config).read_text()) if isinstance(config,(str,Path)) else copy.deepcopy(config)
    if c['schema']!='three_d435i_depth_v1' or len(c['cameras'])!=3:raise ValueError('Expected front/left/right camera schema')
    if c['width']<32 or c['height']<24 or not 0<c['hfov_deg']<180 or not 0<c['vfov_deg']<180:raise ValueError('Bad camera projection')
    return c

def camera_intrinsics(config):
    w,h=config['width'],config['height']
    fx=w/2/np.tan(np.deg2rad(config['hfov_deg']/2));fy=h/2/np.tan(np.deg2rad(config['vfov_deg']/2))
    return np.array([[fx,0,(w-1)/2],[0,fy,(h-1)/2],[0,0,1.]])

def attach_cameras(world,config):
    c=load_config(config);body=world.find(".//body[@name='base']");K=camera_intrinsics(c)
    if body is None:raise ValueError('No base body')
    for camera in c['cameras']:
        yaw,pitch=np.deg2rad([camera['yaw_deg'],camera['pitch_deg']]);cy,sy=np.cos(yaw),np.sin(yaw);cp,sp=np.cos(pitch),np.sin(pitch)
        R=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])@np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])@FRONT_OPTICAL@OPTICAL_TO_MJ
        quat=np.zeros(4);mujoco.mju_mat2Quat(quat,R.ravel())
        ET.SubElement(body,'camera',name='d435i_'+camera['name'],mode='fixed',pos=' '.join(map(str,camera['position_body'])),quat=' '.join(map(str,quat)),resolution=f"{c['width']} {c['height']}",sensorsize='.00128 .0008',focalpixel=f'{K[0,0]} {K[1,1]}',principalpixel='0 0')

def blind_guard(command,observation,min_visibility=1.0,turn_to_observe=False):
    """Keep yaw available; stop translation into unobserved sectors. No rear oracle."""
    result=np.asarray(command,float).copy();visibility=observation[1][0,:,0]
    if np.linalg.norm(result[:2])>.02:
        angle=math.atan2(result[1],result[0]);index=int(np.rint(angle/(np.pi/18)))%36
        neighbors=[(index-1)%36,index,(index+1)%36]
        if min(visibility[neighbors])<min_visibility:
            result[:2]=0
            if turn_to_observe and abs(angle)>np.deg2rad(10):result[2]=np.clip(result[2]+1.8*angle,-1.5,1.5)
    return result

class ThreeCameraDepth:
    def __init__(self,model,config=None,seed=0,noise=True):
        self.config=load_config(config);self.model=model;self.rng=np.random.default_rng(seed);self.noise=noise
        c=self.config;self.K=camera_intrinsics(c);self.base=model.body('base').id
        model.vis.map.znear=c.get('near_clip_m',.03)/max(model.stat.extent,1e-9)
        self.renderer=mujoco.Renderer(model,c['height'],c['width']);self.renderer.enable_depth_rendering()
        self.options=mujoco.MjvOption();self.options.geomgroup[:]=[1,0,0,1,1,0]
        self.camera_ids=[model.camera('d435i_'+camera['name']).id for camera in c['cameras']]
        angles=(np.arange(36)[:,None]*10+np.array([-4.,-2.,0.,2.,4.])[None,:])*np.pi/180
        self.sample_angles=angles
        self.last=None;self.read_count=0
    def read(self,data):
        c=self.config;position=data.xpos[self.base].copy();body=data.xmat[self.base].reshape(3,3)
        yaw=np.arctan2(body[1,0],body[0,0]);cc,ss=np.cos(yaw),np.sin(yaw);R=np.array([[cc,ss,0],[-ss,cc,0],[0,0,1]])
        clouds=[];frames=[];coverage=np.zeros((36,5),bool)
        for camid in self.camera_ids:
            self.renderer.update_scene(data,camera=camid,scene_option=self.options)
            depth=self.renderer.render().copy()
            depth[(depth<c.get('min_depth_m',.1))|(depth>c['max_depth_m'])|(~np.isfinite(depth))]=0
            if self.noise:
                valid=depth>0;depth[valid]+=self.rng.normal(0,c['noise_std_m'],int(valid.sum()))
                depth[self.rng.random(depth.shape)<c['pixel_dropout']]=0
                if self.rng.random()<c['camera_dropout']:depth[:]=0
            T=np.eye(4);T[:3,:3]=R@data.cam_xmat[camid].reshape(3,3)@OPTICAL_TO_MJ;T[:3,3]=R@(data.cam_xpos[camid]-position)
            points=deproject_depth(depth,self.K,T,c['stride']);clouds.append(points)
            # Valid ray support at several obstacle-relevant heights; floor
            # returns are allowed to establish visibility, not obstacles.
            for world_height in [.15,.4,.7]:
                targets=np.stack([4*np.cos(self.sample_angles),4*np.sin(self.sample_angles),np.full((36,5),world_height-position[2])],-1)
                optical=(targets-T[:3,3])@T[:3,:3];z=optical[:,:,2]
                u=np.rint(self.K[0,0]*optical[:,:,0]/np.maximum(z,1e-6)+self.K[0,2]).astype(int)
                v=np.rint(self.K[1,1]*optical[:,:,1]/np.maximum(z,1e-6)+self.K[1,2]).astype(int)
                inside=(z>.05)&(u>=0)&(u<c['width'])&(v>=0)&(v<c['height'])
                valid=np.zeros((36,5),bool);valid[inside]=depth[v[inside],u[inside]]>.05
                coverage|=valid
            frames.append(depth)
        points=np.concatenate(clouds);above=points[:,2]+position[2]
        obstacle_points=points[(above>.12)&(above<1.5)]
        fraction=coverage.mean(1)
        # Unknown distance is a placeholder; mask zero explicitly disambiguates
        # it from known free. Never query hidden geometry for completion.
        scan,_,ranges=pointcloud_scan(obstacle_points,np.where(fraction>0,4.,0.),body_height=position[2])
        scan[0,:,0]=fraction
        scan[0,fraction==0,1:]=0
        ranges[fraction==0]=np.nan
        # CBF proxies from observed surface points only, at most 12 sectors.
        proxies=[]
        if len(obstacle_points):
            xy=obstacle_points[:,:2];radius=np.linalg.norm(xy,axis=1);angle=np.arctan2(xy[:,1],xy[:,0]);ids=(np.floor((angle+np.pi)/(2*np.pi)*12).astype(int))%12
            for i in range(12):
                candidates=np.flatnonzero((ids==i)&(radius<4.2)&(radius>.1))
                if len(candidates):
                    point=xy[candidates[np.argmin(radius[candidates])]]
                    world=position[:2]+R[:2,:2].T@point
                    proxies.append((world,.08,np.zeros(2)))
        result={'lidar':scan,'visibility':fraction.astype(np.float32),'ranges':ranges,'depth':np.stack(frames),'points':obstacle_points,'obstacles':proxies,'time':float(data.time),'source':'rendered_depth_only'}
        self.last=result;self.read_count+=1
        return result
    def close(self):self.renderer.close()

def validate_camera_checkpoint(metadata,config,allow_mismatch=False):
    actual=load_config(config);trained=metadata.get('camera_config',{})
    keys=['schema','width','height','hfov_deg','vfov_deg','scan_range_m','robot_radius_m','near_clip_m','min_depth_m','min_motion_visibility','cameras']
    matches=metadata.get('sensor_mode')==actual['schema'] and all(trained.get(k)==actual.get(k) for k in keys)
    if not matches and not allow_mismatch:raise ValueError('Camera checkpoint schema or rig differs; retrain or explicitly allow diagnostic mismatch')
    return matches
