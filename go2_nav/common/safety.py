"""Planar velocity projection for Go2, separate from upstream ROS safety."""
import numpy as np

def safe_velocity(desired,position,obstacles,speed=.6,robot_radius=.55,tracking_margin=.2,barrier_gain=.7):
    """Nearest feasible velocity under linearized distance barriers and speed cap.

    obstacles contains (center_xy, radius, velocity_xy). This first-order model
    is a practical filter, not a guarantee for an articulated robot with delay.
    """
    desired=np.asarray(desired,float);desired*=min(1.,speed/max(np.linalg.norm(desired),1e-8))
    normals=[];bounds=[]
    # An inscribed regular 16-gon conservatively respects the Euclidean speed cap.
    for angle in np.arange(16)*2*np.pi/16:
        normals.append([np.cos(angle),np.sin(angle)]);bounds.append(speed*np.cos(np.pi/16))
    for center,radius,velocity in obstacles:
        relative=np.asarray(center)-position
        if np.linalg.norm(relative)>4.:continue
        radius=radius+robot_radius+tracking_margin
        normals.append(relative);bounds.append(float(relative@velocity)+barrier_gain*(relative@relative-radius*radius)/2.)
    A=np.array(normals);b=np.array(bounds)
    def feasible(v):return np.all(A@v<=b+1e-7)
    if feasible(desired):return desired,False
    candidates=[]
    for n,bound in zip(A,b):
        v=desired-n*((n@desired-bound)/max(n@n,1e-9))
        if feasible(v):candidates.append(v)
    for i in range(len(A)):
        for j in range(i):
            pair=A[[i,j]]
            if abs(np.linalg.det(pair))<1e-8:continue
            v=np.linalg.solve(pair,b[[i,j]])
            if feasible(v):candidates.append(v)
    if candidates:return min(candidates,key=lambda v:np.sum((v-desired)**2)),False
    # Infeasibility is exposed to logs; choose the speed-bounded least-violation action.
    velocities=np.array([[np.cos(a),np.sin(a)] for a in np.linspace(0,2*np.pi,128,endpoint=False)])*speed*.98
    violations=np.maximum(velocities@A.T-b,0).max(-1)
    return velocities[np.argmin(violations)],True
