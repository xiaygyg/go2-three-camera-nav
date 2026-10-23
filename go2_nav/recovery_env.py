"""Identical recovery, camera safety and reward processing for train/evaluation."""
import numpy as np
from .environment import Go2Batch as Go2Batch
from go2_nav.common.recovery import EscapeMemory, SCHEMA
from go2_nav.common.three_camera_depth import blind_guard
from go2_nav.common.execution import planar_rotation
from go2_nav.common.safety import safe_velocity
from .lateral_escape import LateralEscape,LateralV2Escape,LateralHybridEscape

def camera_execution(command, observation, position, rotation, obstacles, min_visibility=1., yaw_mode="preserve"):
    """Keep the selected yaw when CBF translation points into a blind sector.

    Adding another yaw correction from the projected translation can cancel the
    actor/supervisor yaw and create a stationary equilibrium. Unknown translation
    is stopped, while the chosen observation turn is preserved.
    """
    # Preserve legacy float32 projection rounding for revision-2 replay.
    command=np.asarray(command).copy() if yaw_mode=="projected" else np.asarray(command,float).copy()
    safe,infeasible=safe_velocity(rotation@command[:2],np.asarray(position)[:2],obstacles,speed=float(np.hypot(1.7,.8)),barrier_gain=1.4)
    command[:2]=rotation.T@safe
    return blind_guard(command,observation,min_visibility=min_visibility,turn_to_observe=yaw_mode=="projected"),infeasible

class RecoveryGo2Batch(Go2Batch):
    def __init__(self,*args,recovery_enabled=True,recovery_features=True,yaw_mode="preserve",escape_mode='legacy',**kwargs):
        if yaw_mode not in ("preserve","projected"):raise ValueError("Bad yaw mode")
        if escape_mode not in ('legacy','lateral','lateral_v2','lateral_hybrid'):raise ValueError('Bad escape mode')
        self.escape_mode=escape_mode
        self.yaw_mode=yaw_mode
        self.recovery_enabled=recovery_enabled
        self.recovery_features=recovery_features
        self.memories=None
        super().__init__(*args,**kwargs)
        if self.depth_sensor is None: raise ValueError('Recovery requires rendered three-camera input')
        self.last_requested=np.zeros((self.n,3)); self.last_executed_target=self.last_requested.copy()
        self.last_shaping=np.zeros(self.n)
        self.last_infeasible=np.zeros(self.n,bool)

    def reset(self,indices=None,**kwargs):
        if self.memories is None: self.memories=[EscapeMemory() for _ in range(self.n)]
        ids=list(range(self.n) if indices is None else indices)
        for i in ids:
            self.memories[i]={'legacy':EscapeMemory,'lateral':LateralEscape,'lateral_v2':LateralV2Escape,'lateral_hybrid':LateralHybridEscape}[self.escape_mode]()
            self.memories[i].trace={'last_yaw':None,'turn_rad':0.,'lateral_distance':0.,'stopped_seconds':0.,'active_seconds':0.,'turn_events':0,'turning':False}
        return super().reset(ids,**kwargs)

    def observe(self):
        observation=super().observe()
        if self.memories is not None and self.recovery_features:
            for i,m in enumerate(self.memories):
                observation[0][i,2]=min(1.,m.stall_seconds/3.)
                observation[0][i,4]=float(m.active)
        return observation

    def advance(self,commands,auto_reset=True,render_callback=None):
        commands=np.asarray(commands).copy(); self.last_requested=commands.copy(); blind_penalty=np.zeros(self.n)
        for i,m in enumerate(self.memories):
            info=self.infos[i]; R=planar_rotation(info['rotation']); yaw=np.arctan2(R[1,0],R[0,0])
            guarded_raw=blind_guard(commands[i],tuple(x[i] for x in self.observation),min_visibility=self.depth_sensor.config.get('min_motion_visibility',1.))
            blind_penalty[i]=.02*min(4.,np.linalg.norm(guarded_raw[:2]-commands[i,:2]))
            if self.recovery_enabled:
                commands[i]=m.command(commands[i],info['position'],yaw,self.goal[i],self.observation[1][i])
            commands[i],self.last_infeasible[i]=camera_execution(commands[i],tuple(x[i] for x in self.observation),info['position'],R,self.sensor_infos[i]['obstacles'],self.depth_sensor.config.get('min_motion_visibility',1.),self.yaw_mode)
        self.last_executed_target=commands.copy()
        observation,reward,done,terminal=super().advance(commands,auto_reset=False,render_callback=render_callback)
        shaping=np.zeros(self.n)
        for i,m in enumerate(self.memories):
            info=self.infos[i]; R=planar_rotation(info['rotation']); yaw=np.arctan2(R[1,0],R[0,0])
            shaping[i]=m.update(info['position'],yaw,self.goal[i],self.observation[1][i],self.data[i].time)
            trace=m.trace
            if trace['last_yaw'] is not None:trace['turn_rad']+=abs(float((yaw-trace['last_yaw']+np.pi)%(2*np.pi)-np.pi))
            trace['last_yaw']=yaw
            yaw_rate=abs(float(info['omega'][2]))
            if not trace['turning'] and yaw_rate>.35:trace['turn_events']+=1;trace['turning']=True
            elif trace['turning'] and yaw_rate<.15:trace['turning']=False
            trace['lateral_distance']+=.1*abs(float(info['body_velocity'][1]))
            trace['stopped_seconds']+=.1*float(np.linalg.norm(info['body_velocity'][:2])<.15)
            trace['active_seconds']+=.1*float(m.active)
            # Penalize a raw action that repeatedly asks to enter blind sectors.
            shaping[i]-=blind_penalty[i]
            terminal[i].update(recovery_activations=m.activations,recovery_escapes=m.escapes)
            terminal[i].update(escape_mode=self.escape_mode,execution_metrics={k:v for k,v in trace.items() if k!='last_yaw'},escape_metrics=dict(getattr(m,'stats',{})))
        self.last_shaping=shaping; reward+=shaping.astype(np.float32); self.episode_return+=shaping
        for i in range(self.n):
            terminal[i]['return_']=float(self.episode_return[i])
            if self.recovery_features:
                observation[0][i,2]=min(1.,self.memories[i].stall_seconds/3.)
                observation[0][i,4]=float(self.memories[i].active)
        if auto_reset and done.any(): self.reset(np.flatnonzero(done))
        return self.observation,reward,done,terminal
