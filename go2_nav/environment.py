"""Go2 articulated physics with SEA-Nav locomotion and the shared navigation task."""
from pathlib import Path
import sys,math
import numpy as np
import torch
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from .model import make_model
from .robot import JOINTS,DEFAULT,KP,KD,LIMIT,LOW,HIGH,PERM,SeaGo2Lower,proprioception

from go2_nav.common.scenes import update_motions
from go2_nav.common.perception import dynamic_input
from go2_nav.common.navigation_domain import connected_domain,contains
from go2_nav.common.forest_scene import BOUNDS as FOREST_BOUNDS




def local_observation(model,data,base,command,goal,velocities):
    """15-D body state, body-frame 36x4 scans, and 5x10 dynamic tracks."""
    position=data.xpos[base].copy();rotation=data.xmat[base].reshape(3,3);yaw=math.atan2(rotation[1,0],rotation[0,0]);c,s=math.cos(yaw),math.sin(yaw);R=np.array([[c,s],[-s,c]])
    spatial=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,base,spatial,0)
    delta=goal-position[:2];distance=np.linalg.norm(delta);body_velocity=rotation.T@spatial[3:];omega=rotation.T@spatial[:3]
    state=np.zeros(15,np.float32);state[:2]=R@delta/max(distance,1e-6);state[3]=min(distance,20);state[5:8]=body_velocity;state[8:11]=omega*.25;state[11:14]=command/np.array([1.7,.8,1.]);state[14]=(position[2]-.32)*2
    theta=yaw+np.arange(36)*np.pi/18;directions=np.stack([np.cos(theta),np.sin(theta)],axis=-1);ranges=np.full(36,4.);clearance=10.
    for gid in np.flatnonzero(model.geom_group==3):
        center=data.geom_xpos[gid,:2];size=model.geom_size[gid];kind=model.geom_type[gid];rot=data.geom_xmat[gid].reshape(3,3)
        offset=center-position[:2]
        if kind==mujoco.mjtGeom.mjGEOM_BOX:
            rr=rot[:2,:2];local=rr.T@(-offset);nearest=rr@np.clip(local,-size[:2],size[:2])+offset;gap=np.linalg.norm(nearest)
            clearance=min(clearance,gap-.55)
            if gap>4.6:continue
            half=size[:2]+.55;o=local;d=directions@rr
            valid=np.ones(36,bool);lower=np.full(36,-np.inf);upper=np.full(36,np.inf)
            for axis in range(2):
                parallel=np.abs(d[:,axis])<1e-8;valid &= (~parallel)|(abs(o[axis])<=half[axis]);denom=np.where(parallel,1,d[:,axis]);aa=(-half[axis]-o[axis])/denom;bb=(half[axis]-o[axis])/denom
                lower=np.maximum(lower,np.where(parallel,-np.inf,np.minimum(aa,bb)));upper=np.minimum(upper,np.where(parallel,np.inf,np.maximum(aa,bb)))
            hit=np.where(valid&(upper>=np.maximum(0,lower)),np.maximum(.1,lower),4.);ranges=np.minimum(ranges,hit)
        else:
            if kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
                axis=rot[:2,2];a=center-axis*size[1];b=center+axis*size[1];line=b-a;nearest=a+np.clip((position[:2]-a)@line/max(line@line,1e-8),0,1)*line
                gap=np.linalg.norm(nearest-position[:2])-size[0];centers=[a,b,(a+b)/2]
            else:gap=np.linalg.norm(offset)-size[0];centers=[center]
            clearance=min(clearance,gap-.55)
            if gap>4.6:continue
            radius=size[0]+.55
            for cc in centers:
                off=cc-position[:2];proj=directions@off;perp2=off@off-proj*proj;root=np.sqrt(np.maximum(0,radius*radius-perp2));hit=np.where((perp2<=radius*radius)&(proj+root>=0),np.maximum(.1,proj-root),4.);ranges=np.minimum(ranges,hit)
    lidar=np.zeros((1,36,4),np.float32);lidar[0,:,1:]=(4-np.clip(ranges,.1,4))[:,None]
    dynamic=dynamic_input(model,data,position,np.array([c,s]),velocities)[0]
    return (state,lidar,dynamic),dict(position=position,rotation=rotation,spatial=spatial,body_velocity=body_velocity,omega=omega,distance=float(distance),clearance=float(clearance))


class Go2Batch:
    def __init__(self,num_envs=16,scenario='warehouse',seed=0,horizon=250,camera_config=None,camera_noise=True,goal_distance=(3.,float('inf'))):
        torch.set_num_threads(1);self.n=num_envs;self.scenario=scenario;self.rng=np.random.default_rng(seed);self.horizon=horizon;self.goal_distance=goal_distance
        self.model=make_model(scenario,goal=(12,0),camera_config=camera_config);self.data=[mujoco.MjData(self.model) for _ in range(num_envs)]
        self.obstacle_mask=self.model.geom_group==3;self.base=self.model.body('base').id;self.qi=np.array([self.model.joint(n).qposadr[0] for n in JOINTS]);self.vi=np.array([self.model.joint(n).dofadr[0] for n in JOINTS]);self.ai=np.array([self.model.actuator(n.replace('_joint','')).id for n in JOINTS])
        self.lower=SeaGo2Lower()
        self.last=torch.zeros(num_envs,12);self.history=torch.zeros(num_envs,10,45);self.primed=np.zeros(num_envs,bool)
        self.command=np.zeros((num_envs,3));self.target=np.tile(DEFAULT,(num_envs,1));self.goal=np.zeros((num_envs,2));self.steps=np.zeros(num_envs,int);self.previous_distance=np.zeros(num_envs);self.episode_return=np.zeros(num_envs);self.completed=[];self.infos=[None]*num_envs
        mujoco.mj_forward(self.model,self.data[0]);self.domain=connected_domain(self.model,self.data[0],resolution=.2,bounds=((-50.,60.,-23.,23.) if scenario=='forest_sprint' else FOREST_BOUNDS)) if scenario in ('forest','forest_sprint') else connected_domain(self.model,self.data[0])
        self.depth_sensor=None;self.sensor_infos=[None]*num_envs
        if camera_config is not None:
            from go2_nav.common.three_camera_depth import ThreeCameraDepth
            self.depth_sensor=ThreeCameraDepth(self.model,camera_config,seed,noise=camera_noise)
        self.reset()
    def free(self,point,margin=.9):
        data=self.data[0]
        for gid in np.flatnonzero(self.model.geom_group==3):
            center=data.geom_xpos[gid,:2];size=self.model.geom_size[gid];kind=self.model.geom_type[gid]
            if kind==mujoco.mjtGeom.mjGEOM_BOX:
                rot=data.geom_xmat[gid].reshape(3,3)[:2,:2];local=rot.T@(point-center);gap=np.linalg.norm(local-np.clip(local,-size[:2],size[:2]))
            else:gap=np.linalg.norm(point-center)-size[0]-(size[1] if kind==mujoco.mjtGeom.mjGEOM_CAPSULE else 0)
            if gap<margin:return False
        return True
    def sample_point(self):
        for _ in range(1000):
            p=self.rng.uniform([FOREST_BOUNDS[0]+1,FOREST_BOUNDS[2]+1],[FOREST_BOUNDS[1]-1,FOREST_BOUNDS[3]-1]) if self.scenario in ('forest','forest_sprint') else self.rng.uniform([.2,-3.1],[12.2,3.1])
            if contains(self.domain,p) and self.free(p):return p
        return np.array([0.,0.])
    def reset(self,indices=None,fixed_start=None,fixed_goal=None):
        indices=range(self.n) if indices is None else indices
        for i in indices:
            d=self.data[i];mujoco.mj_resetData(self.model,d);mujoco.mj_forward(self.model,d)
            start=np.asarray(fixed_start) if fixed_start is not None else (np.array([0.,0.]) if self.scenario=='empty' else self.sample_point())
            goal=np.asarray(fixed_goal) if fixed_goal is not None else self.sample_point()
            for _ in range(100):
                if self.goal_distance[0]<np.linalg.norm(goal-start)<=self.goal_distance[1]:break
                goal=self.sample_point()
            yaw=self.rng.uniform(-np.pi,np.pi) if fixed_start is None else 0.
            d.qpos[:3]=[*start,.42];d.qpos[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)];d.qpos[self.qi]=DEFAULT;self.goal[i]=goal;d.mocap_pos[self.model.body('goal_marker').mocapid[0]]=[*goal,.04];self.command[i]=0;self.target[i]=DEFAULT;self.steps[i]=0;self.episode_return[i]=0;self.last[i]=0;self.history[i]=0;self.primed[i]=False;mujoco.mj_forward(self.model,d)
        self.observe();self.previous_distance[list(indices)]=[self.infos[i]['distance'] for i in indices]
        return self.observation
    def observe(self):
        obs=[]
        for i,d in enumerate(self.data):
            velocities=update_motions(self.model,d,self.scenario);mujoco.mj_forward(self.model,d)
            observation,info=local_observation(self.model,d,self.base,self.command[i],self.goal[i],velocities);
            if self.depth_sensor is not None:
                sensor=self.depth_sensor.read(d);self.sensor_infos[i]=sensor
                observation=(observation[0],sensor['lidar'],np.zeros((1,5,10),np.float32))
            self.infos[i]=info;obs.append(observation)
        self.observation=tuple(np.stack([o[k] for o in obs]) for k in range(3));return self.observation
    @torch.inference_mode()
    def advance(self,commands,auto_reset=True,render_callback=None):
        commands=np.clip(np.asarray(commands),LOW,HIGH);previous=self.command.copy()
        if self.depth_sensor is not None:
            from go2_nav.common.three_camera_depth import blind_guard
            commands=commands.copy()
            for i in range(self.n):commands[i]=blind_guard(commands[i],tuple(x[i] for x in self.observation),min_visibility=self.depth_sensor.config.get('min_motion_visibility',1.))
        # Physically feasible slew rate; this is an execution filter, not a speed cap.
        self.command += np.clip(commands-self.command,[-.7,-.4,-.5],[.7,.4,.5])
        if self.depth_sensor is not None:
            for i in range(self.n):self.command[i]=blind_guard(self.command[i],tuple(x[i] for x in self.observation),min_visibility=self.depth_sensor.config.get('min_motion_visibility',1.))
        collision=np.zeros(self.n,bool);fall=np.zeros(self.n,bool)
        for tick in range(20):
            if tick%4==0:
                observations=torch.zeros(self.n,45)
                for i,d in enumerate(self.data):
                    mujoco.mj_forward(self.model,d);rot=d.xmat[self.base].reshape(3,3);spatial=np.zeros(6);mujoco.mj_objectVelocity(self.model,d,mujoco.mjtObj.mjOBJ_BODY,self.base,spatial,0)
                    observations[i]=torch.from_numpy(proprioception(rot,rot.T@spatial[:3],d.qpos[self.qi],d.qvel[self.vi],self.command[i] if d.time>=1 else np.zeros(3),self.last[i].numpy()))
                    if not self.primed[i]:self.history[i]=observations[i].expand(10,-1);self.primed[i]=True
                self.history[:,:-1]=self.history[:,1:].clone();self.history[:,-1]=observations
                self.last.copy_(self.lower(observations,self.history))
                action=self.last.numpy()[:,PERM]
                for i,d in enumerate(self.data):
                    blend=np.clip((d.time-.2)/.8,0,1);self.target[i]=DEFAULT+blend*.25*action[i]
            for i,d in enumerate(self.data):
                d.ctrl[self.ai]=np.clip(KP*(self.target[i]-d.qpos[self.qi])-KD*d.qvel[self.vi],-LIMIT,LIMIT);mujoco.mj_step(self.model,d)
                fall[i]|=d.qpos[2]<.16
                if d.ncon:collision[i]|=bool(np.any(self.obstacle_mask[d.contact.geom1]|self.obstacle_mask[d.contact.geom2]))
            if render_callback is not None:render_callback(self,tick)
        self.steps+=1;self.observe()
        for i,d in enumerate(self.data):
            fall[i]|=self.infos[i]['rotation'][2,2]<.4
            for contact in d.contact[:d.ncon]:
                names=[self.model.geom(int(g)).name or '' for g in (contact.geom1,contact.geom2)]
                if any(n.startswith('obstacle_') for n in names):collision[i]=True
        distance=np.array([x['distance'] for x in self.infos]);clearance=np.array([x['clearance'] for x in self.infos]);reached=distance<.55;timeout=self.steps>=self.horizon
        progress=self.previous_distance-distance
        reward=8*progress-.03-.015*np.sum((self.command-previous)**2,axis=1)-.15*np.maximum(0,.25-clearance)-.025*np.array([np.linalg.norm(x['omega'][:2]) for x in self.infos])
        reward+=reached*15-collision*20-fall*25;done=reached|collision|fall|timeout;self.episode_return+=reward;self.previous_distance=distance
        terminal=[dict(reached=bool(reached[i]),collision=bool(collision[i]),fall=bool(fall[i]),distance=float(distance[i]),steps=int(self.steps[i]),return_=float(self.episode_return[i])) for i in range(self.n)]
        for i in np.flatnonzero(done):self.completed.append(terminal[i])
        if auto_reset and done.any():self.reset(np.flatnonzero(done))
        return self.observation,reward.astype(np.float32),done,terminal
