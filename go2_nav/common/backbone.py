"""Pure PyTorch evaluation of the unchanged upstream NavRL Beta actor."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]

def mlp(dims):
    layers=[]
    for a,b in zip(dims[:-1],dims[1:]):
        layers.extend([nn.Linear(a,b),nn.LeakyReLU(),nn.LayerNorm(b)])
    return nn.Sequential(*layers)

class NavRLPolicy(nn.Module):
    def __init__(self, checkpoint=None, device='cpu', load_weights=True):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(1,4,(5,3),padding=(2,1)),nn.ELU(),nn.Conv2d(4,16,(5,3),stride=(2,1),padding=(2,1)),nn.ELU(),nn.Conv2d(16,16,(5,3),stride=(2,2),padding=(2,1)),nn.ELU(),nn.Flatten(1),nn.Linear(288,128),nn.LayerNorm(128))
        self.dynamic=mlp([50,128,64])
        self.fusion=mlp([200,256,256])
        self.alpha=nn.Linear(256,3)
        self.beta=nn.Linear(256,3)
        self.critic=nn.Linear(256,1)
        self.device=torch.device(device)
        if load_weights:
            p=torch.load(checkpoint or ROOT/'assets/pretrained/navrl_checkpoint.pt',map_location='cpu',weights_only=True)
            for module,prefix in [(self.cnn,'feature_extractor.module.0.module.'),(self.dynamic,'feature_extractor.module.1.module.1.'),(self.fusion,'feature_extractor.module.3.module.'),(self.alpha,'actor.module.0.module.alpha_layer.'),(self.beta,'actor.module.0.module.beta_layer.'),(self.critic,'critic.module.')]:
                module.load_state_dict({k[len(prefix):]:v for k,v in p.items() if k.startswith(prefix)},strict=True)
        self.to(self.device).eval()

    def distribution(self,state,lidar,dynamic):
        feature=self.fusion(torch.cat([self.cnn(lidar),self.dynamic(dynamic.reshape(-1,50)),state.reshape(-1,8)],-1))
        a=1+nn.functional.softplus(self.alpha(feature))+1e-6
        b=1+nn.functional.softplus(self.beta(feature))+1e-6
        return torch.distributions.Independent(torch.distributions.Beta(a,b),1),self.critic(feature)

    def forward(self,state,lidar,dynamic):
        dist,_=self.distribution(state,lidar,dynamic)
        return 4*dist.mean-2

    @torch.inference_mode()
    def plan(self,state,lidar,dynamic,target_dir):
        state,lidar,dynamic=[torch.as_tensor(x,dtype=torch.float32,device=self.device) for x in (state,lidar,dynamic)]
        local=self(state,lidar,dynamic)[0].cpu().numpy()
        direction=np.asarray(target_dir,dtype=float).reshape(-1)[:2]
        theta=np.arctan2(direction[1],direction[0])
        c,s=np.cos(theta),np.sin(theta)
        return np.array([c*local[0]-s*local[1],s*local[0]+c*local[1]])
