"""Continuous Go2 forest patrol, using the trained local navigation policy."""
import argparse,csv,json,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
# Use the training depth-rendering backend on Linux unless explicitly overridden.
_camera_cli=('--sensor=three_camera' in sys.argv) or any(x=='--sensor' and i+1<len(sys.argv) and sys.argv[i+1]=='three_camera' for i,x in enumerate(sys.argv))
if _camera_cli and sys.platform.startswith('linux'):os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
from go2_nav.environment import local_observation
from go2_nav.robot import LOW,HIGH
from go2_nav.recovery_env import RecoveryGo2Batch as Go2Batch
from go2_nav.common.recovery import SCHEMA
from go2_nav.policy import Go2NavigationPolicy as NavigationPolicy
from go2_nav.common.forest_scene import DEFAULT_GOALS,BOUNDS,layout,SEED
from go2_nav.common.navigation_domain import contains
from go2_nav.common.patrol import switch_goal
from go2_nav.common.perception import scan_and_obstacles
from go2_nav.common.safety import safe_velocity
from go2_nav.common.execution import planar_rotation,straight_command,prefer_heading

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--approach-length',type=float,default=40,help='Straight approach before forest entrance x=-6; 40 gives start x=-46')
    p.add_argument('--sprint-mode',choices=['command','policy'],default='command')
    p.add_argument('--sprint-speed',type=float,default=1.7)
    p.add_argument('--heading-first',action='store_true',help='Explicit execution adapter: convert lateral preference into yaw; network unchanged')
    p.add_argument('--sensor',choices=['geometry','three_camera'],default='three_camera')
    p.add_argument('--camera-config',default='configs/go2_three_d435i.json')
    p.add_argument('--camera-noise',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--allow-sensor-mismatch',action='store_true',help='Explicit diagnostic baseline only')
    p.add_argument('--checkpoint',default=None)
    p.add_argument('--goals',nargs='+',help='Sequential x,y pairs, e.g. 18,0 36,13 53,0')
    p.add_argument('--random-goals',type=int,default=0,help='Sample a sequence of connected goals at least 18 m apart')
    p.add_argument('--seed',type=int,default=SEED,help='Random goal seed; forest geometry stays fixed')
    p.add_argument('--seconds-per-goal',type=float,default=100)
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--video',action='store_true')
    p.add_argument('--name',default='forest_patrol')
    p.add_argument('--fps',type=int,default=25)
    p.add_argument('--recovery',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--escape-mode',choices=['legacy','lateral','lateral_v2','lateral_hybrid'],default='legacy')
    a=p.parse_args()
    if a.sensor!='three_camera':p.error('Recovery runner requires three_camera')
    if a.checkpoint is None:a.checkpoint='assets/checkpoints/go2_three_camera.pt'
    if not a.name.replace('_','').replace('-','').isalnum():p.error('name must be alphanumeric')
    if a.seconds_per_goal<=0 or a.fps<1:p.error('duration and fps must be positive')
    if a.random_goals<0 or (a.random_goals and a.goals):p.error('Choose --goals or a positive --random-goals count')
    if not 0<=a.approach_length<=40 or not 0<a.sprint_speed<=1.7:p.error('approach length must be 0..40 m and sprint speed 0..1.7 m/s')
    env=Go2Batch(1,'forest_sprint' if a.approach_length else 'forest',a.seed,horizon=round(a.seconds_per_goal*10),camera_config=a.camera_config if a.sensor=='three_camera' else None,camera_noise=a.camera_noise,recovery_enabled=a.recovery,escape_mode=a.escape_mode)
    if a.random_goals:
        sampled=[];previous=np.zeros(2)
        for _ in range(a.random_goals):
            for attempt in range(1000):
                candidate=env.sample_point()
                if 18<=np.linalg.norm(candidate-previous)<=35:break
            else:p.error('Could not sample sufficiently distant goals')
            sampled.append(candidate);previous=candidate
        goals=np.asarray(sampled)
    else:goals=np.array([list(map(float,g.split(','))) for g in a.goals]) if a.goals else DEFAULT_GOALS.copy()
    if goals.ndim!=2 or goals.shape[1]!=2 or not np.isfinite(goals).all():p.error('goals must be finite x,y pairs')
    for goal in goals:
        if not contains(env.domain,goal) or not env.free(goal):p.error(f'Goal {goal} has insufficient connected clearance')
    for j in range(len(DEFAULT_GOALS)):
        gid=env.model.geom(f'patrol_waypoint_{j}').id
        if j<len(goals):env.model.geom_pos[gid,:2]=goals[j]
        else:env.model.geom_rgba[gid,3]=0
    env.reset(fixed_start=[-6-a.approach_length,0] if a.approach_length else [0,0],fixed_goal=goals[0])
    policy,metadata=NavigationPolicy.load(ROOT/a.checkpoint)
    if metadata.get('robot')!='go2':p.error('Use a Go2 checkpoint')
    env.yaw_mode=metadata.get('yaw_mode','preserve' if metadata.get('supervisor_revision',0)>=3 else 'projected')
    env.recovery_features=metadata.get('recovery_schema')==SCHEMA
    env.observe()
    if a.sensor=='three_camera':
        from go2_nav.common.three_camera_depth import validate_camera_checkpoint
        try:validate_camera_checkpoint(metadata,a.camera_config,a.allow_sensor_mismatch)
        except ValueError as error:p.error(str(error))
    out=ROOT/'results';out.mkdir(exist_ok=True)
    viewer=recorder=None;index=0;rows=[];events=[];status='running';last_frame=-1
    trees,rocks=layout(approach=bool(a.approach_length))
    if a.viewer:
        from go2_nav.rendering import visual_options
        from mujoco import viewer as mujoco_viewer
        viewer=mujoco_viewer.launch_passive(env.model,env.data[0])
        viewer.opt.geomgroup[:]=visual_options().geomgroup
    if a.video:
        from go2_nav.rendering import Recorder
        recorder=Recorder(env.model,out/f'{a.name}.mp4',fps=a.fps,complex_scene=True)
        recorder.follow=True
        def actual_speed(data):
            velocity=np.zeros(6);mujoco.mj_objectVelocity(env.model,data,mujoco.mjtObj.mjOBJ_BODY,env.base,velocity,0)
            body=data.xmat[env.base].reshape(3,3).T@velocity[3:]
            return f'{a.escape_mode}/{getattr(env.memories[0],"phase","legacy")} | {"ESCAPE" if env.memories[0].active else "POLICY"} / recoveries={env.memories[0].activations} | MEASURED body vx={body[0]:+.2f} vy={body[1]:+.2f} m/s | planar speed={np.linalg.norm(body[:2]):.2f} m/s | simulation, not hardware'
        recorder.details=actual_speed
        # This overview is for the human viewer and is never fed to the policy.
        def draw_map(frame,draw,data):
            x0,y0,w,h=1010,82,250,180
            draw.rectangle((x0-4,y0-4,x0+w+4,y0+h+25),fill=(20,27,36))
            def pix(p):
                return (x0+(p[0]-(BOUNDS[0]-a.approach_length))/(BOUNDS[1]-BOUNDS[0]+a.approach_length)*w,y0+h-(p[1]-BOUNDS[2])/(BOUNDS[3]-BOUNDS[2])*h)
            for x,y,r,height in trees:
                px,py=pix((x,y));draw.ellipse((px-1.5,py-1.5,px+1.5,py+1.5),fill=(70,130,65))
            if len(rows)>1:draw.line([pix(r[2:4]) for r in rows],fill=(255,200,70),width=2)
            for j,g in enumerate(goals):
                px,py=pix(g);color=(60,255,90) if j==index else ((255,200,70) if j<index else (80,155,245))
                draw.ellipse((px-4,py-4,px+4,py+4),fill=color);draw.text((px+5,py-8),str(j+1),fill=color)
            px,py=pix(data.qpos[:2]);draw.ellipse((px-3,py-3,px+3,py+3),fill='white')
            draw.text((x0,y0+h+5),'Overview: display only / not policy input',fill=(210,220,230))
            if env.depth_sensor is not None:
                from PIL import Image
                depth=env.sensor_infos[0]['depth']
                for j,label in enumerate(['FRONT','LEFT','RIGHT']):
                    v=np.clip(depth[j]/env.depth_sensor.config['max_depth_m'],0,1)
                    rgb=np.stack([v*40,v*200,v*250],-1).astype(np.uint8);rgb[depth[j]<=0]=0
                    x=14+j*188;y=110
                    frame.paste(Image.fromarray(rgb).resize((180,112)),(x,y+20))
                    draw.text((x,y),'DEPTH '+label,fill=(235,245,255))
                draw.text((14,246),'Gray radar: unknown/partial, NOT confirmed free | depth-only controller',fill=(220,230,240))
        recorder.after_draw=draw_map
    wall=time.monotonic();leg_start=0.;sprinting=bool(a.approach_length);handoff_time=None
    def render(env,tick):
        nonlocal last_frame
        d=env.data[0]
        if viewer:
            if not viewer.is_running():return
            with viewer.lock():
                viewer.cam.lookat[:]=d.xpos[env.base]+[.3,0,.25]
                viewer.cam.distance=7;viewer.cam.azimuth=110;viewer.cam.elevation=-18
            if tick%4==0:viewer.sync()
            delay=wall+d.time-time.monotonic()
            if delay>0:time.sleep(min(delay,.02))
        frame=int(d.time*a.fps+1e-7)
        if recorder and frame>last_frame:
            last_frame=frame
            obs,info=local_observation(env.model,d,env.base,env.command[0],env.goal[0],{})
            if env.depth_sensor is not None:obs=tuple(x[0] for x in env.observation)
            recorder.append(d,info['position'],env.goal[0],env.command[0],f'{("STRAIGHT / direct Go2 velocity command" if a.sprint_mode=="command" else "STRAIGHT / network control") if sprinting else "FOREST / local navigation"} / goal {index+1}/{len(goals)} / {a.sensor}',obs,visibility=env.sensor_infos[0]['visibility'] if env.depth_sensor is not None else None)
    print(f'Forest {BOUNDS[1]-BOUNDS[0]:.0f} x {BOUNDS[3]-BOUNDS[2]:.0f} m; {len(trees)} trees + {len(rocks)} rocks; {len(goals)} continuous goals',flush=True)
    print(f'Policy sensor={a.sensor}: local 4 m scan + body state + goal direction/distance. No global map.',flush=True)
    try:
        for step in range(round(a.seconds_per_goal*10)*(len(goals)+int(bool(a.approach_length)))):
            if viewer and not viewer.is_running():status='viewer_closed';break
            if sprinting and env.data[0].qpos[0]>=-6:
                sprinting=False;handoff_time=float(env.data[0].time);env.steps[0]=0
                print(f'Continuous handoff to forest at t={handoff_time:.2f}s; no robot/controller reset',flush=True)
            commands=policy.plan(env.observation)
            raw=commands[0].copy()
            info=env.infos[0]
            if sprinting and a.sprint_mode=='command':
                commands[0]=straight_command(info['position'],info['rotation'],a.sprint_speed)
            elif a.heading_first:
                commands[0]=prefer_heading(commands[0])
            requested=commands[0].copy();infeasible=False
            # Safety and recovery run inside the shared training/evaluation env.
            filtered=commands[0].copy();filter_delta=float(np.linalg.norm(filtered[:2]-requested[:2]))
            _,_,done,term=env.advance(np.clip(commands,LOW,HIGH),auto_reset=False,render_callback=render if (viewer or recorder) else None)
            infeasible=bool(env.last_infeasible[0])
            filtered=env.last_executed_target[0].copy();filter_delta=float(np.linalg.norm(filtered[:2]-requested[:2]))
            info=env.infos[0];d=env.data[0]
            rows.append([float(d.time),index+1,*info['position'],*env.command[0],*info['body_velocity'][:2],info['distance'],info['clearance'],*raw,*requested,*filtered,int(not sprinting),filter_delta,int(infeasible)])
            if step%100==0:print(f't={d.time:.1f}s goal={index+1}/{len(goals)} xy={d.qpos[:2].round(2).tolist()} remaining={info["distance"]:.2f}m',flush=True)
            if done[0]:
                event={**term[0],'goal_index':index+1,'goal':goals[index].tolist(),'time_seconds':float(d.time),'leg_seconds':float(d.time-leg_start),'position':d.qpos[:2].tolist()}
                events.append(event);print(json.dumps(event),flush=True)
                if term[0]['collision'] or term[0]['fall']:status='collision' if term[0]['collision'] else 'fall';break
                if not term[0]['reached']:status='timeout';break
                index+=1
                if index==len(goals):status='completed';break
                leg_start=float(d.time)
                switch_goal(env,goals[index])
                memory=env.memories[0]
                memory.active=False;memory.heading=None;memory.history.clear()
                memory.scan_remaining=0;memory.stall_seconds=0.;memory.no_motion=0.
                memory.last_goal=goals[index].copy();env.observe()
                print(f'Switching to goal {index+1}/{len(goals)} without resetting robot or controller',flush=True)
        else:status='timeout'
    finally:
        if recorder:recorder.close_writer()
        if viewer:viewer.close()
        if env.depth_sensor is not None:env.depth_sensor.close()
    r=np.asarray(rows)
    summary={'status':status,'goals_reached':sum(e['reached'] and not e['collision'] and not e['fall'] for e in events),'goals_total':len(goals),'simulation_seconds':float(env.data[0].time),'wall_seconds':time.monotonic()-wall,'trees':len(trees),'rocks':len(rocks),'bounds':BOUNDS,'collision':any(e['collision'] for e in events),'fall':any(e['fall'] for e in events),'robot_resets_during_patrol':0,'handoff_time':handoff_time,'path_length':float(np.linalg.norm(np.diff(r[:,2:4],axis=0),axis=1).sum()) if len(r)>1 else 0.}
    def metrics(part):
        if not len(part):return None
        speed=np.linalg.norm(part[:,8:10],axis=1)
        steady=part[part[:,0]>2]
        return {'seconds':len(part)*.1,'mean_speed':float(speed.mean()),'peak_speed':float(speed.max()),'p95_speed':float(np.quantile(speed,.95)),'mean_forward_speed_after_startup':float(steady[:,8].mean()) if len(steady) else None,'mean_abs_lateral_speed':float(np.abs(part[:,9]).mean()),'mean_abs_lateral_command':float(np.abs(part[:,6]).mean()),'network_mean_vx':float(part[:,12].mean()),'requested_mean_vx':float(part[:,15].mean()),'safety_intervention_fraction':float(np.mean(part[:,22]>1e-4)),'stopped_seconds':float(np.sum(speed<.15)*.1),'safety_infeasible_steps':int(part[:,23].sum())}
    summary['straight']=metrics(r[r[:,21]==0]) if len(r) else None
    summary['forest']=metrics(r[r[:,21]==1]) if len(r) else None
    report={'escape_mode':a.escape_mode,'escape_metrics':dict(getattr(env.memories[0],'stats',{})),'execution_metrics':{k:v for k,v in env.memories[0].trace.items() if k!='last_yaw'},'yaw_mode':env.yaw_mode,'filter_delta_includes_recovery':True,'recovery_enabled':a.recovery,'recovery_schema':SCHEMA,'recovery_activations':env.memories[0].activations,'recovery_escapes':env.memories[0].escapes,'sensor':a.sensor,'blind_turn_recovery':True,'camera_config':env.depth_sensor.config if env.depth_sensor is not None else None,'perception_truth_leak':False if env.depth_sensor is not None else 'geometry mode','approach_length':a.approach_length,'sprint_mode':a.sprint_mode,'sprint_speed':a.sprint_speed,'heading_first':a.heading_first,'summary':summary,'goals':goals.tolist(),'events':events,'checkpoint':a.checkpoint,'policy_has_global_map':False,'scene_seed':SEED,'goal_seed':a.seed}
    (out/f'{a.name}.json').write_text(json.dumps(report,indent=2))
    with (out/f'{a.name}.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['time','goal_index','x','y','z','vx_cmd','vy_cmd','wz_cmd','vx_body','vy_body','distance','clearance','raw_vx','raw_vy','raw_wz','requested_vx','requested_vy','requested_wz','filtered_vx','filtered_vy','filtered_wz','forest_phase','safety_delta','safety_infeasible']);writer.writerows(rows)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig,ax=plt.subplots(figsize=(14,10))
    for x,y,radius,height in trees:ax.add_patch(Circle((x,y),radius,color='#496f36'))
    for x,y,radius,height in rocks:ax.add_patch(Circle((x,y),radius,color='#77776a'))
    if len(r):ax.plot(r[:,2],r[:,3],color='#d38e09',lw=2,label='actual robot trajectory')
    for j,g in enumerate(goals):ax.plot(*g,'*',ms=14);ax.annotate(str(j+1),g+np.array([.5,.5]))
    ax.set(xlim=(-8-a.approach_length,BOUNDS[1]),ylim=BOUNDS[2:],xlabel='x / m',ylabel='y / m',title=f'Forest patrol: {summary["goals_reached"]}/{len(goals)} goals / {status}')
    ax.set_aspect('equal');ax.grid(alpha=.2);fig.savefig(out/f'{a.name}_route.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if status=='completed' else 2

if __name__=='__main__':sys.exit(main())
