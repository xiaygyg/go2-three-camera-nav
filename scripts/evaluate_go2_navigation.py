"""Predeclared six-route evaluation with depth-only recovery and honest failures."""
import os,sys,argparse,json,time,hashlib
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from go2_nav.recovery_env import RecoveryGo2Batch as RecoveryGo2Batch
from go2_nav.common.recovery import SCHEMA
from go2_nav.policy import Go2NavigationPolicy as NavigationPolicy
from go2_nav.common.three_camera_depth import validate_camera_checkpoint

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True);p.add_argument('--seconds',type=float,default=45);p.add_argument('--recovery',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--noise',action='store_true');p.add_argument('--yaw-mode',choices=['auto','preserve','projected'],default='auto');p.add_argument('--plan',default='configs/evaluation_routes.json');p.add_argument('--escape-mode',choices=['legacy','lateral','lateral_v2','lateral_hybrid'],default='legacy');a=p.parse_args()
 if a.seconds<=0 or not a.name.replace('_','').replace('-','').isalnum():p.error('Invalid duration/name')
 out=ROOT/'results';out.mkdir(exist_ok=True);path=out/(a.name+'.json')
 if path.exists():p.error('Choose a new name')
 plans=json.loads((ROOT/a.plan).read_text())['routes'];n=len(plans)
 policy,meta=NavigationPolicy.load(ROOT/a.checkpoint);validate_camera_checkpoint(meta,'configs/go2_three_d435i.json')
 if meta.get('robot')!='go2':p.error('Use Go2 navigation weights')
 env=RecoveryGo2Batch(n,'forest',2918,horizon=round(a.seconds*10),camera_config='configs/go2_three_d435i.json',camera_noise=a.noise,escape_mode=a.escape_mode,recovery_enabled=a.recovery,recovery_features=meta.get('recovery_schema')==SCHEMA,yaw_mode=(meta.get('yaw_mode','preserve' if meta.get('supervisor_revision',0)>=3 else 'projected') if a.yaw_mode=='auto' else a.yaw_mode))
 for i,case in enumerate(plans):env.reset([i],fixed_start=[0,0],fixed_goal=case['goals'][0])
 active=np.ones(n,bool);result=[None]*n;tracks=[[] for _ in range(n)];started=time.monotonic()
 for step in range(round(a.seconds*10)):
  commands=policy.plan(env.observation);commands[~active]=0
  _,_,done,terms=env.advance(commands,auto_reset=False)
  for i in np.flatnonzero(active):
   info=env.infos[i];m=env.memories[i]
   tracks[i].append([float(env.data[i].time),*info['position'][:2],*env.command[i],info['distance'],float(m.active),m.stall_seconds,*commands[i],*info['body_velocity'][:2]])
   if done[i]:result[i]={**terms[i],'route':int(i+1),'goal':plans[i]['goals'][0],'time':float(env.data[i].time)};active[i]=False
  if step%100==0:print(json.dumps({'step':step,'distances':[round(x['distance'],2) for x in env.infos],'active':active.tolist()}),flush=True)
  if not active.any():break
 for i,rows in enumerate(tracks):
  arr=np.array(rows);result[i]['stopped_seconds']=float(np.sum(np.linalg.norm(arr[:,-2:],axis=1)<.15)*.1)
  result[i]['stopped_fraction']=float(np.mean(np.linalg.norm(arr[:,-2:],axis=1)<.15))
  result[i]['path_length']=float(np.linalg.norm(np.diff(arr[:,1:3],axis=0),axis=1).sum())
  np.savetxt(out/f'{a.name}_r{i+1:02d}.csv',arr,delimiter=',',header='time,x,y,vx,vy,wz,distance,recovery,stall,raw_vx,raw_vy,raw_wz,measured_vx,measured_vy',comments='')
 summary={'success':sum(r['reached'] and not r['collision'] and not r['fall'] for r in result),'collision':sum(r['collision'] for r in result),'fall':sum(r['fall'] for r in result),'timeout':sum(not r['reached'] and not r['collision'] and not r['fall'] for r in result),'mean_stopped_seconds':float(np.mean([r['stopped_seconds'] for r in result])),'mean_final_distance':float(np.mean([r['distance'] for r in result])),'wall_seconds':time.monotonic()-started}
 summary['mean_turn_rad']=float(np.mean([r['execution_metrics']['turn_rad'] for r in result]));summary['mean_lateral_distance']=float(np.mean([r['execution_metrics']['lateral_distance'] for r in result]));report={'escape_mode':a.escape_mode,'summary':summary,'runs':result,'checkpoint':a.checkpoint,'checkpoint_sha256':hashlib.sha256((ROOT/a.checkpoint).read_bytes()).hexdigest(),'yaw_mode':env.yaw_mode,'recovery':a.recovery,'noise':a.noise,'plan':a.plan,'goal_scope':'first goal of each route, no later goals claimed','metadata':meta}
 path.write_text(json.dumps(report,indent=2));env.depth_sensor.close();print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
