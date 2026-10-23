"""Explicit execution adapters, separate from the learned navigation policy."""
import numpy as np

def planar_rotation(rotation):
    """Yaw-only rotation; roll/pitch must not shrink planar velocity commands."""
    yaw=np.arctan2(rotation[1,0],rotation[0,0]);c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c,-s],[s,c]])

def straight_command(position,rotation,speed=1.7):
    yaw=np.arctan2(rotation[1,0],rotation[0,0])
    desired_yaw=np.clip(np.arctan2(-.8*position[1],speed),-.3,.3)
    error=(desired_yaw-yaw+np.pi)%(2*np.pi)-np.pi
    return np.array([speed,0,np.clip(2.5*error,-1.5,1.5)])

def prefer_heading(command):
    result=np.array(command,dtype=float,copy=True)
    angle=np.arctan2(result[1],max(result[0],.15))
    result[2]=np.clip(result[2]+1.8*angle,-1.5,1.5)
    result[1]*=.2
    return result
