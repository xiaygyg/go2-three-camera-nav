"""Behavior warm-up followed by PPO in actual articulated MuJoCo physics."""
import sys,argparse,json,time,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from go2_nav.policy import Go2NavigationPolicy as NavigationPolicy
from go2_nav.robot import LOW,HIGH
from go2_nav.recovery_env import RecoveryGo2Batch as Go2Batch
from go2_nav.common.recovery import SCHEMA
from go2_nav.common.three_camera_training import mask_synthetic,camera_teacher
from go2_nav.common.three_camera_depth import load_config

def synthetic(batch,device):
    state=torch.zeros(batch,15,device=device);angle=(torch.rand(batch,device=device)*2-1)*torch.pi;state[:,0]=angle.cos();state[:,1]=angle.sin();state[:,3]=torch.rand(batch,device=device)*14+.5;state[:,5]=torch.rand(batch,device=device)*1.7;state[:,6:8]=torch.randn(batch,2,device=device)*.2;state[:,8:11]=torch.randn(batch,3,device=device)*.05;state[:,11:14]=torch.rand(batch,3,device=device)*.7;state[:,14]=torch.randn(batch,device=device)*.03
    angles=torch.rand(batch,12,device=device)*2*torch.pi;radii=torch.rand(batch,12,device=device)*5+1.3;obstacle_radius=torch.rand(batch,12,device=device)*.45+.7;active=torch.rand(batch,12,device=device)<torch.rand(batch,1,device=device)*.65
    rays=torch.arange(36,device=device)*torch.pi/18;projection=radii[:,:,None]*torch.cos(angles[:,:,None]-rays);perp2=radii[:,:,None]**2-projection**2;rr=obstacle_radius[:,:,None]**2;root=torch.sqrt((rr-perp2).clamp(min=0));hit=torch.where((perp2<rr)&(projection+root>0)&active[:,:,None],(projection-root).clamp(min=.1),torch.full_like(projection,4));ranges=hit.amin(1).clamp(.1,4)
    lidar=torch.zeros(batch,1,36,4,device=device);lidar[:,0,:,1:]=(4-ranges)[:,:,None]
    return state,lidar,torch.zeros(batch,1,5,10,device=device)

def checkpoint(model,path,metadata):
    temporary=path.with_suffix('.tmp');torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'metadata':metadata},temporary);temporary.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=200000);ap.add_argument('--envs',type=int,default=32);ap.add_argument('--warmup',type=int,default=800);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--resume');ap.add_argument('--output',default='runs/go2_three_camera_new');ap.add_argument('--sensor',choices=['geometry','three_camera'],default='three_camera');ap.add_argument('--camera-config',default='configs/go2_three_d435i.json');ap.add_argument('--scenario',default='forest');ap.add_argument('--goal-max-distance',type=float,default=35.);ap.add_argument('--bc-weight',type=float,default=.1);ap.add_argument('--horizon',type=int,default=600);ap.add_argument('--escape-mode',choices=['legacy','lateral','lateral_v2','lateral_hybrid'],default='legacy');args=ap.parse_args()
    output=ROOT/args.output
    if (output/'final.pt').exists(): raise FileExistsError('Choose a fresh training output directory')
    output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);np.random.seed(args.seed);torch.manual_seed(args.seed);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=NavigationPolicy().to(device) if not args.resume else NavigationPolicy.load(args.resume,device)[0];optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);start=time.monotonic()
    model.set_go2_limits()
    is_camera=args.sensor=='three_camera';teacher_fn=lambda state,lidar:camera_teacher(state,lidar).clamp(model.low,model.high)
    sensor_metadata={'sensor_mode':'three_d435i_depth_v1','camera_config':load_config(args.camera_config),'scan_column0':'visibility_fraction','dynamic_source':'zero_no_tracker'} if is_camera else {'sensor_mode':'geometry'}
    if not is_camera: raise ValueError('Recovery training requires three_camera')
    sensor_metadata.update(escape_mode=args.escape_mode,robot='go2',lower_policy='SEA-Nav body_latest + encoder_vel + encoder_latent frozen',lower_policy_commit='560de73c094a750a6a358bc1920ebc9ef1d9f3c5',lower_proprioception_dim=45,lower_history_shape=[10,45],training_supervisor_enabled=True,supervisor_revision=3,yaw_mode="preserve",recovery_schema=SCHEMA,state_index_2='stall_seconds_div3_clipped',state_index_4='escape_active',execution='depth_only_CBF_blind_guard_persistent_escape',novelty_reward_episode_cap=3.)
    if args.warmup>0:

        for iteration in range(args.warmup):
            obs=synthetic(512,device);obs=mask_synthetic(obs) if is_camera else obs;teacher=teacher_fn(obs[0],obs[1]);prediction=model(*obs);loss=(((prediction-teacher)/(model.high-model.low))**2).mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            if iteration%100==0:print(json.dumps({'warmup':iteration,'loss':float(loss),'wall':time.monotonic()-start}),flush=True)
        checkpoint(model,output/'warmup.pt',{'phase':'synthetic_local_teacher_warmup','seed':args.seed,**sensor_metadata})
    env=Go2Batch(args.envs,args.scenario,args.seed,horizon=args.horizon,camera_config=args.camera_config if is_camera else None,goal_distance=(3,args.goal_max_distance),escape_mode=args.escape_mode);rollout=64;updates=math.ceil(args.steps/(args.envs*rollout));logs=[]
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    for update in range(updates):
        buffer=[]
        for step in range(rollout):
            obs=[torch.as_tensor(x,dtype=torch.float32,device=device) for x in env.observation]
            with torch.no_grad():
                dist,value=model.distribution(*obs);action=dist.sample().clamp(1e-5,1-1e-5);logprob=dist.log_prob(action);command=model.low+(model.high-model.low)*action;teacher=teacher_fn(obs[0],obs[1])
            next_obs,reward,done,info=env.advance(command.cpu().numpy())
            # Teach the persistent escape command on intervention steps; PPO
            # still scores the sampled action in the shielded environment.
            recovering=torch.as_tensor(obs[0][:,4]>0,device=device)
            teacher=torch.where(recovering[:,None],torch.as_tensor(env.last_executed_target,dtype=torch.float32,device=device),teacher)
            buffer.append((obs,action,logprob,value,torch.as_tensor(reward,device=device),torch.as_tensor(done,device=device),teacher))
        with torch.no_grad():_,next_value=model.distribution(*[torch.as_tensor(x,dtype=torch.float32,device=device) for x in env.observation])
        advantages=[];gae=torch.zeros(args.envs,device=device)
        for item in reversed(buffer):
            mask=~item[5];delta=item[4]+.99*next_value*mask-item[3];gae=delta+.99*.95*mask*gae;advantages.append(gae);next_value=item[3]
        advantages=torch.stack(advantages[::-1]);values=torch.stack([x[3] for x in buffer]);returns=(advantages+values).flatten();adv=advantages.flatten();adv=(adv-adv.mean())/(adv.std()+1e-8)
        states=[torch.cat([x[0][k] for x in buffer]) for k in range(3)];actions=torch.cat([x[1] for x in buffer]);old_log=torch.cat([x[2] for x in buffer]);teachers=torch.cat([x[6] for x in buffer]);N=actions.shape[0]
        losses=[]
        for epoch in range(4):
            for ids in torch.randperm(N,device=device).split(256):
                dist,value=model.distribution(*[x[ids] for x in states]);logprob=dist.log_prob(actions[ids]);ratio=(logprob-old_log[ids]).exp();policy_loss=-torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean();value_loss=((value-returns[ids])**2).mean();mean=model.low+(model.high-model.low)*dist.mean;bc=(((mean-teachers[ids])/(model.high-model.low))**2).mean()
                loss=policy_loss+.03*value_loss-.001*dist.entropy().mean()+args.bc_weight*bc;optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step();losses.append(float(loss))
        recent=env.completed[-200:];log={'update':update+1,'physics_transitions':(update+1)*rollout*args.envs,'episodes':len(env.completed),'recent_success':np.mean([x['reached'] for x in recent]).item() if recent else 0.,'recent_collision':np.mean([x['collision'] for x in recent]).item() if recent else 0.,'recent_fall':np.mean([x['fall'] for x in recent]).item() if recent else 0.,'recent_recovery_activations':float(np.mean([x.get('recovery_activations',0) for x in recent])) if recent else 0.,'recent_recovery_escapes':float(np.mean([x.get('recovery_escapes',0) for x in recent])) if recent else 0.,'mean_loss':np.mean(losses).item(),'depth_frames_rendered':3*env.depth_sensor.read_count if is_camera else 0,'wall_seconds':time.monotonic()-start};logs.append(log);print(json.dumps(log),flush=True)
        (output/'training.json').write_text(json.dumps(logs,indent=2));checkpoint(model,output/'latest.pt',{**sensor_metadata,'phase':'MuJoCo_PPO_with_training_only_BC_regularization','seed':args.seed,**log,'action_low':LOW.tolist(),'action_high':HIGH.tolist(),'state_dim':15,'initialized_from':args.resume,'goal_sampling':'connected_065m_domain'})
        if (update+1)%10==0:checkpoint(model,output/f'update_{update+1}.pt',{**sensor_metadata,'phase':'MuJoCo_PPO','seed':args.seed,**log})
    checkpoint(model,output/'final.pt',{**sensor_metadata,'phase':'MuJoCo_PPO_with_training_only_BC_regularization','seed':args.seed,**logs[-1],'action_low':LOW.tolist(),'action_high':HIGH.tolist(),'state_dim':15,'initialized_from':args.resume,'goal_sampling':'connected_065m_domain'});print('FINAL',output/'final.pt',flush=True)
    if env.depth_sensor is not None:env.depth_sensor.close()
if __name__=='__main__':main()
