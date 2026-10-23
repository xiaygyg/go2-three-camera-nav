"""Switch goals without resetting physics, controller memory or simulation time."""
import numpy as np
import mujoco

def switch_goal(env,goal):
    goal=np.asarray(goal,dtype=float)
    if goal.shape!=(2,) or not np.isfinite(goal).all():raise ValueError('Expected finite x,y goal')
    env.goal[0]=goal
    env.data[0].mocap_pos[env.model.body('goal_marker').mocapid[0]]=[*goal,.04]
    env.steps[0]=0
    env.episode_return[0]=0
    mujoco.mj_forward(env.model,env.data[0])
    observation=env.observe()
    env.previous_distance[0]=env.infos[0]['distance']
    return observation
