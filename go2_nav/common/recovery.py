"""Depth-only persistent escape supervisor; no simulator or map imports.

Pose must come from odometry on hardware. Scan ranges already include the robot
footprint. Unknown sectors never authorize translation. Memory is episode-local.
"""
from collections import deque, Counter
import math
import numpy as np

SCHEMA = 'three_camera_recovery_v1'

def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class EscapeMemory:
    def __init__(self):
        self.history = deque(maxlen=31)
        self.visits = Counter()
        self.seen = set()
        self.active = False
        self.stall_seconds = 0.
        self.heading = None
        self.age = 0.
        self.attempts = 0
        self.activations = 0
        self.escapes = 0
        self.reward_paid = 0.
        self.last_cell = None
        self.last_goal = None
        self.scan_direction = 1.
        self.failed_heading = None
        self.scan_remaining = 0
        self.no_motion = 0.
        self.last_position = None

    def update(self, position, yaw, goal, scan, time):
        position = np.asarray(position)[:2]
        goal = np.asarray(goal)
        if self.last_goal is None or np.linalg.norm(goal-self.last_goal) > .1:
            self.history.clear(); self.active=False; self.heading=None
            self.last_goal=goal.copy(); self.stall_seconds=0.
        distance = float(np.linalg.norm(goal-position))
        self.history.append((time, position.copy(), distance))
        old_time, old_pos, old_distance = self.history[0]
        duration = time-old_time
        displacement = np.linalg.norm(position-old_pos)
        progress = old_distance-distance
        stalled = duration >= 2.8 and (displacement < .25 or (progress < .08 and displacement < .8))
        self.stall_seconds = min(10., self.stall_seconds+.1) if stalled else max(0.,self.stall_seconds-.2)
        cell = tuple(np.floor(position/.6).astype(int))
        novel = cell not in self.visits
        if cell != self.last_cell:
            self.visits[cell] += 1; self.last_cell = cell
        # Only new, fully observed free locations count as information gain.
        visibility=scan[0,:,0]; ranges=4-scan[0,:,1]
        new_seen=0
        for k in np.flatnonzero((visibility >= .999)&(ranges > .8)):
            angle=yaw+k*np.pi/18
            endpoint=position+min(2.,ranges[k]-.2)*np.array([np.cos(angle),np.sin(angle)])
            key=tuple(np.floor(endpoint/.6).astype(int))
            if key not in self.seen: self.seen.add(key); new_seen+=1
        if stalled and not self.active and distance>.6:
            self.active=True; self.heading=None; self.age=0.; self.attempts=0
            self.origin=position.copy(); self.start_distance=distance; self.activations+=1
        if self.active:
            moved_step=0. if self.last_position is None else np.linalg.norm(position-self.last_position)
            self.no_motion=self.no_motion+.1 if moved_step<.015 else 0.
            self.age+=.1
            moved=np.linalg.norm(position-self.origin)
            if moved>1.5 and distance<self.start_distance-.4:
                self.active=False; self.heading=None; self.escapes+=1; self.history.clear()
            elif self.age>10.:
                self.failed_heading=self.heading
                self.heading=None; self.age=0.; self.attempts+=1
        self.last_position=position.copy()
        # Bounded novelty prevents indefinite exploration being more profitable
        # than arrival. Repeated turns in an already observed place pay nothing.
        bonus=min(.04*new_seen + (.12 if novel else 0.), max(0.,3.-self.reward_paid)) if self.active else 0.
        self.reward_paid+=bonus
        return bonus - (.12 if stalled else 0.)

    def command(self, raw, position, yaw, goal, scan):
        if not self.active: return np.asarray(raw).copy()
        position=np.asarray(position)[:2]; goal=np.asarray(goal)
        if self.scan_remaining>0:
            self.scan_remaining-=1
            return np.array([0.,0.,.8*self.scan_direction])
        visibility=scan[0,:,0]; ranges=4-scan[0,:,1]
        angles=np.arange(36)*np.pi/18
        valid=(visibility>=.999)&(np.roll(visibility,1)>=.999)&(np.roll(visibility,-1)>=.999)
        clearance=np.minimum(ranges,np.minimum(np.roll(ranges,1),np.roll(ranges,-1)))
        valid &= clearance>.7
        if self.heading is None:
            ids=np.flatnonzero(valid)
            if len(ids):
                world=yaw+angles[ids]
                length=np.minimum(2.,clearance[ids]-.25)
                endpoints=position+length[:,None]*np.stack([np.cos(world),np.sin(world)],-1)
                costs=np.array([self.visits[tuple(np.floor(p/.6).astype(int))] for p in endpoints])
                gain=np.linalg.norm(goal-position)-np.linalg.norm(goal-endpoints,axis=1)
                score=1.3*gain+.35*length-1.2*costs-.15*np.abs(wrap(angles[ids]))
                # Subsequent attempts prefer an untried side of the obstacle.
                if self.failed_heading is not None: score-=.8*np.cos(world-self.failed_heading)
                chosen=int(np.argmax(score)); self.heading=float(world[chosen])
                self.scan_direction=1. if wrap(self.heading-yaw)>=0 else -1.
            else:
                return np.array([0.,0.,.8*self.scan_direction])
        error=float(wrap(self.heading-yaw))
        # Escape itself may stall after alignment because CBF/visibility changed.
        # Commit to a scan, rather than repeatedly choosing the same blocked yaw.
        if self.no_motion>2. and (abs(error)<.35 or self.age>8.):
            endpoint=position+2*np.array([np.cos(self.heading),np.sin(self.heading)])
            self.visits[tuple(np.floor(endpoint/.6).astype(int))]+=3
            self.failed_heading=self.heading;self.heading=None
            self.scan_remaining=15;self.no_motion=0.;self.age=0.;self.attempts+=1
            return np.array([0.,0.,.8*self.scan_direction])
        forward=min(ranges[35],ranges[0],ranges[1])
        speed=min(1.2,max(0.,(forward-.3)*.8)) if valid[0] and abs(error)<.3 else 0.
        # If alignment completes into a newly seen obstacle, choose another gap.
        if abs(error)<.3 and forward<.7: self.heading=None
        return np.array([speed,0.,np.clip(2.*error,-1.,1.)])
