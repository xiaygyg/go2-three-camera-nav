"""Camera-conditioned training teacher and synthetic prewarm; teacher is not deployed."""
import torch

def mask_synthetic(observation):
    state,lidar,dynamic=observation
    theta=torch.arange(36,device=state.device)*10
    a=(theta+180)%360-180
    coverage=((a.abs()<=43.5)|((a-90).abs()<=43.5)|((a+90).abs()<=43.5)).float()
    coverage[(a.abs()==40)|(a.abs()==50)]=.6
    mask=coverage[None].expand(len(state),-1).clone()
    mask[torch.rand_like(mask)<.025]=0
    lidar=lidar.clone();lidar[:,0,:,0]=mask
    lidar[:,0,:,1:]=torch.where(mask[:,:,None]>0,lidar[:,0,:,1:],0)
    return state,lidar,torch.zeros_like(dynamic)

def camera_teacher(state,lidar):
    theta=torch.arange(36,device=state.device)*torch.pi/18;theta=(theta+torch.pi)%(2*torch.pi)-torch.pi
    goal=torch.atan2(state[:,1],state[:,0]);ranges=4-lidar[:,0,:,1];visibility=lidar[:,0,:,0]
    score=2.2*torch.cos(theta[None]-goal[:,None])+.45*ranges.clamp(max=3)-2*torch.exp(-ranges/.8)-.18*theta.abs()[None]
    score=torch.where(visibility>=.6,score,torch.full_like(score,-100))
    selected=score.argmax(-1);angle=theta[selected];clearance=ranges.gather(1,selected[:,None]).squeeze(1)
    speed=torch.minimum(torch.full_like(goal,3.2),state[:,3]*1.1)
    speed=torch.minimum(speed,(clearance-.35).clamp(min=0)*1.0)
    vx=speed*torch.cos(angle).clamp(min=0)
    vx=torch.where(angle.abs()>1.,torch.zeros_like(vx),vx)
    vy=.1*torch.sin(angle)*torch.cos(angle).clamp(min=0)
    wz=(2.2*angle).clamp(-1.5,1.5)
    no_view=visibility.max(-1).values<.6
    vx=torch.where(no_view,0,vx);vy=torch.where(no_view,0,vy)
    wz=torch.where(no_view,torch.full_like(wz,.8),wz)
    return torch.stack([vx,vy,wz],-1)
