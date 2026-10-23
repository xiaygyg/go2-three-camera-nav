"""SEA-Nav locomotion interface in MuJoCo model joint order FL FR RL RR."""
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
JOINTS=[f'{leg}_{joint}_joint' for leg in ['FL','FR','RL','RR'] for joint in ['hip','thigh','calf']]
PERM=[3,4,5,0,1,2,9,10,11,6,7,8]
DEFAULT=np.array([.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5])
KP=30.;KD=.75;LIMIT=np.tile([23.7,23.7,45.43],4)
LOW=np.array([-.5,-.8,-1.],np.float32);HIGH=np.array([1.7,.8,1.],np.float32)


class SeaGo2Lower:
    def __init__(self):
        folder=ROOT/'assets/go2/downloaded/sea'
        self.body=torch.jit.load(str(folder/'body_latest.jit'),map_location='cpu').eval()
        self.velocity=torch.jit.load(str(folder/'encoder_vel.jit'),map_location='cpu').eval()
        self.latent=torch.jit.load(str(folder/'encoder_latent.jit'),map_location='cpu').eval()
        for module in (self.body,self.velocity,self.latent):
            for p in module.parameters():p.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self,prop,history):
        velocity=self.velocity(history.flatten(1));latent=self.latent(history.flatten(1))
        # Released velocity encoder outputs TWO horizontal components, not three.
        actor=torch.cat([velocity,prop,prop[:,2:3],latent],dim=-1)
        if actor.shape[-1]!=64:raise ValueError('SEA Go2 actor requires 64 inputs')
        action=self.body(actor).clamp(-100,100)
        if not torch.isfinite(action).all():raise ValueError('Non-finite Go2 action')
        return action


def proprioception(rotation,angular_velocity,q,dq,command,previous_action):
    return np.concatenate([np.asarray(angular_velocity)*.25,np.asarray(rotation).T@np.array([0.,0.,-1.]),
                           np.asarray(command)*[2.,2.,.25],(np.asarray(q)-DEFAULT)[PERM],
                           np.asarray(dq)[PERM]*.05,np.asarray(previous_action)]).astype(np.float32)
