"""Depth-only lateral-first escape. Historical unknown space is never made free."""
from collections import Counter
import numpy as np
from go2_nav.common.recovery import EscapeMemory,wrap

PHASES={'goal':0,'align_goal':1,'slide':2,'forward':3,'turn_out':4,'advance_out':5,'scan':6}


class LateralEscape(EscapeMemory):
    front_threshold=1.0
    side_speed=.45
    side_seconds=5.
    forward_speed=1.
    forward_margin=.4
    outward_distance=1.2
    outward_speed=.9
    def __init__(self):
        super().__init__()
        self.phase='goal';self.clock=0.;self.phase_start=0.;self.phase_origin=None
        self.side=0;self.tried=set();self.slide_axis=None;self.last_yaw=None
        self.stats=Counter();self.forward_start=None;self.clear_frames=0;self.blocked_frames=0

    @staticmethod
    def sectors(scan):
        v=scan[0,:,0]>=.999;r=4-scan[0,:,1]
        valid=v&np.roll(v,1)&np.roll(v,-1)
        clearance=np.minimum(r,np.minimum(np.roll(r,1),np.roll(r,-1)))
        return valid,clearance

    def _phase(self,name,position):
        self.phase=name;self.phase_start=self.clock;self.phase_origin=np.asarray(position)[:2].copy()
        self.no_motion=0.;self.clear_frames=0;self.blocked_frames=0
        self.stats['enter_'+name]+=1

    def update(self,position,yaw,goal,scan,time):
        p=np.asarray(position)[:2];goal=np.asarray(goal)
        dt=.1 if not self.history else float(np.clip(time-self.clock,0,.2))
        self.clock=float(time)
        if self.last_yaw is not None:self.stats['absolute_turn_rad']+=abs(float(wrap(yaw-self.last_yaw)))
        self.last_yaw=float(yaw)
        if self.last_goal is None or np.linalg.norm(goal-self.last_goal)>.1:
            self.history.clear();self.active=False;self.phase='goal';self.heading=None;self.tried.clear()
            self.stall_seconds=0.;self.last_goal=goal.copy()
        distance=float(np.linalg.norm(goal-p));self.history.append((time,p.copy(),distance))
        t0,p0,d0=self.history[0]
        displacement=float(np.linalg.norm(p-p0))
        stalled=time-t0>=2.8 and (displacement<.25 or (d0-distance<.08 and displacement<.8))
        self.stall_seconds=min(10.,self.stall_seconds+dt) if stalled else max(0.,self.stall_seconds-2*dt)
        cell=tuple(np.floor(p/.6).astype(int));novel=cell not in self.visits
        if cell!=self.last_cell:self.visits[cell]+=1;self.last_cell=cell
        if stalled and not self.active and distance>.6:
            self.active=True;self.activations+=1;self.origin=p.copy();self.start_distance=distance
            self.age=0.;self.tried.clear();self._phase('align_goal',p)
        step=0. if self.last_position is None else float(np.linalg.norm(p-self.last_position))
        self.last_position=p.copy()
        if self.active:
            self.age+=dt
            self.no_motion=self.no_motion+dt if step<.006 else 0.
            self.stats['active_seconds']+=dt;self.stats[self.phase+'_seconds']+=dt
            # Success requires measured movement AND resumed final-goal progress.
            if self.phase=='forward' and distance<self.start_distance-.4 and np.linalg.norm(p-self.origin)>.6:
                self.active=False;self.phase='goal';self.heading=None;self.escapes+=1;self.history.clear()
            elif self.age>35:
                self.heading=None;self.tried.clear();self.age=0.;self._phase('align_goal',p)
                self.stats['budget_restarts']+=1
        bonus=min(.12 if novel and self.active else 0.,max(0.,3-self.reward_paid));self.reward_paid+=bonus
        return bonus-(.12 if stalled else 0.)

    def _choose_side(self,p,yaw,goal,scan):
        valid,clearance=self.sectors(scan)
        goal_heading=np.arctan2(*(np.asarray(goal)-p)[::-1]);left=np.array([-np.sin(goal_heading),np.cos(goal_heading)])
        choices=[]
        for side in [1,-1]:
            if side in self.tried:continue
            angle=wrap(goal_heading+side*np.pi/2-yaw);k=int(np.rint(angle/(np.pi/18)))%36
            if not valid[k] or clearance[k]<.7:continue
            endpoint=p+left*side*min(1.2,clearance[k]-.2)
            cost=self.visits[tuple(np.floor(endpoint/.6).astype(int))]
            choices.append((min(clearance[k],2.5)-.3*cost,side))
        if not choices:return False
        _,self.side=max(choices);self.slide_axis=self.side*left
        self._phase('slide',p);self.stats['side_attempts']+=1
        return True

    def _outward(self,p,yaw,goal,scan):
        valid,clearance=self.sectors(scan);ids=np.flatnonzero(valid&(clearance>.8))
        if not len(ids):
            self._phase('scan',p);self.stats['no_visible_exit']+=1;return
        angles=yaw+ids*np.pi/18;length=np.minimum(2.,clearance[ids]-.3)
        endpoints=p+length[:,None]*np.stack([np.cos(angles),np.sin(angles)],-1)
        visits=np.array([self.visits[tuple(np.floor(q/.6).astype(int))] for q in endpoints])
        # Here clearance wins over goal attraction: commit to an outward move.
        goal_angle=np.arctan2(*(np.asarray(goal)-p)[::-1])
        scores=length-.35*visits+.2*np.cos(angles-goal_angle)
        if self.failed_heading is not None:scores-=.7*np.cos(angles-self.failed_heading)
        self.heading=float(angles[int(scores.argmax())]);self._phase('turn_out',p)
        self.stats['fallback_turns']+=1

    def command(self,raw,position,yaw,goal,scan):
        if not self.active:return np.asarray(raw).copy()
        p=np.asarray(position)[:2];goal=np.asarray(goal)
        angle=np.arctan2(*(goal-p)[::-1]);error=float(wrap(angle-yaw))
        goal_turn=float(np.clip(2*error,-1.,1.));valid,clearance=self.sectors(scan)
        front=bool(valid[0] and clearance[0]>self.front_threshold)
        # Face the goal during lateral search. Do not translate while aligning.
        if self.phase=='align_goal':
            if abs(error)>.25:return np.array([0.,0.,goal_turn])
            if front:self._phase('forward',p)
            elif not self._choose_side(p,yaw,goal,scan):self._outward(p,yaw,goal,scan)
        if self.phase=='slide':
            if abs(error)>.35:return np.array([0.,0.,goal_turn])
            progress=float((p-self.phase_origin)@self.slide_axis)
            self.clear_frames=self.clear_frames+1 if front else 0
            if self.clear_frames>=3 and progress>.25:
                self._phase('forward',p);self.stats['slide_opened_front']+=1
            else:
                # Translate in the plane perpendicular to the CURRENT target
                # direction; yaw tracks that target rather than the travel vector.
                direction=angle+self.side*np.pi/2
                body_angle=float(wrap(direction-yaw));k=int(np.rint(body_angle/(np.pi/18)))%36
                blocked=not valid[k] or clearance[k]<.55
                self.blocked_frames=self.blocked_frames+1 if blocked else 0
                failed=(self.blocked_frames>=(6 if not valid[k] else 3) or self.no_motion>2.0 or self.clock-self.phase_start>self.side_seconds or progress>1.1)
                if failed:
                    self.tried.add(self.side);self.stats['side_failures']+=1
                    if not self._choose_side(p,yaw,goal,scan):self._outward(p,yaw,goal,scan)
                else:
                    if blocked:return np.array([0.,0.,goal_turn])
                    speed=min(self.side_speed,max(0.,(clearance[k]-.35)*.65))
                    return np.array([speed*np.cos(body_angle),speed*np.sin(body_angle),goal_turn])
        if self.phase=='forward':
            if abs(error)>.3:return np.array([0.,0.,goal_turn])
            if not front or self.no_motion>2.0 or self.clock-self.phase_start>8:
                self._phase('align_goal',p)
                return np.array([0.,0.,goal_turn])
            return np.array([min(self.forward_speed,(clearance[0]-self.forward_margin)*.8),0.,goal_turn])
        if self.phase=='scan':
            if self.clock-self.phase_start>1.5:self._outward(p,yaw,goal,scan)
            return np.array([0.,0.,.8*self.scan_direction])
        if self.phase=='turn_out':
            turn=float(wrap(self.heading-yaw))
            if abs(turn)>.25:return np.array([0.,0.,np.clip(2*turn,-1.,1.)])
            self._phase('advance_out',p)
        if self.phase=='advance_out':
            moved=float(np.linalg.norm(p-self.phase_origin))
            if moved>=self.outward_distance:
                self.tried.clear();self._phase('align_goal',p);self.stats['outward_moves']+=1
                return np.array([0.,0.,goal_turn])
            if not valid[0] or clearance[0]<.55 or self.no_motion>2.0 or self.clock-self.phase_start>7:
                self.failed_heading=self.heading;self._outward(p,yaw,goal,scan)
                return np.array([0.,0.,0.])
            turn=float(wrap(self.heading-yaw))
            return np.array([min(self.outward_speed,max(0.,(clearance[0]-.3)*.8)) if abs(turn)<.3 else 0.,0.,np.clip(2*turn,-1.,1.)])
        return np.array([0.,0.,goal_turn])


class LateralV2Escape(LateralEscape):
    """Match legacy forward gap/speed; keep a finite, faster lateral attempt."""
    front_threshold=.7
    side_speed=.6
    side_seconds=4.
    forward_speed=1.2
    forward_margin=.3
    outward_distance=1.8
    outward_speed=1.2


class LateralHybridEscape(LateralV2Escape):
    """Lateral first, then retain the proven persistent legacy escape controller."""
    def __init__(self):
        super().__init__();self.fallback=None

    def _outward(self,p,yaw,goal,scan):
        f=EscapeMemory();f.active=True;f.origin=np.asarray(p).copy()
        f.start_distance=float(np.linalg.norm(np.asarray(goal)-p));f.last_goal=np.asarray(goal).copy()
        f.last_position=np.asarray(p).copy();f.visits=self.visits.copy();f.last_cell=self.last_cell
        f.history=self.history.copy();f.seen=self.seen.copy()
        self.fallback=f;self.heading=None;self._phase('legacy_out',p)
        self.stats['legacy_fallbacks']+=1

    def update(self,position,yaw,goal,scan,time):
        changed=self.last_goal is not None and np.linalg.norm(np.asarray(goal)-self.last_goal)>.1
        if changed:self.fallback=None
        f=self.fallback
        if f is not None:self.age=0.  # legacy controller owns persistence/retries
        reward=super().update(position,yaw,goal,scan,time)
        if f is not None:
            f.update(position,yaw,goal,scan,time)
            self.visits=f.visits.copy();self.active=f.active;self.phase='legacy_out' if f.active else 'goal'
            if not f.active:
                self.fallback=None;self.escapes+=1;self.history.clear();self.tried.clear()
                self.stats['legacy_exits']+=1
        return reward

    def command(self,raw,position,yaw,goal,scan):
        if self.fallback is not None:return self.fallback.command(raw,position,yaw,goal,scan)
        if self.active and self.phase=='forward' and (self.no_motion>2.0 or self.clock-self.phase_start>8.):
            # A ray can look open while the safety layer or real tracking blocks
            # the requested motion. Do not restart the same forward probe forever.
            self.stats['forward_probe_failures']+=1
            p=np.asarray(position)[:2]
            if not self._choose_side(p,yaw,goal,scan):self._outward(p,yaw,goal,scan)
        result=super().command(raw,position,yaw,goal,scan)
        if self.fallback is not None:return self.fallback.command(raw,position,yaw,goal,scan)
        return result
