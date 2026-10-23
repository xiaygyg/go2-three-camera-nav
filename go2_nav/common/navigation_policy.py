"""Go2 PPO: body-frame 15-D state, unchanged scan layout, direct vx/vy/wz."""
import numpy as np
import torch
from torch import nn
from .backbone import NavRLPolicy,mlp,ROOT

class NavigationPolicy(nn.Module):
    def __init__(self,warm_start=True):
        super().__init__()
        source=NavRLPolicy(load_weights=warm_start)
        self.cnn=source.cnn;self.dynamic=source.dynamic;self.fusion=mlp([207,256,256])
        if warm_start:
            with torch.no_grad():
                self.fusion[0].weight.zero_();self.fusion[0].weight[:,:200]=source.fusion[0].weight;self.fusion[0].bias.copy_(source.fusion[0].bias)
            for index in (2,3,5):self.fusion[index].load_state_dict(source.fusion[index].state_dict())
        self.alpha=nn.Linear(256,3);self.beta=nn.Linear(256,3);self.critic=nn.Linear(256,1)
        nn.init.zeros_(self.critic.weight);nn.init.zeros_(self.critic.bias)
        self.register_buffer('low',torch.tensor([-.5,-.8,-1.]));self.register_buffer('high',torch.tensor([1.7,.8,1.]));self.register_buffer('state_scale',torch.tensor([1.,1.,1.,10.,1.,1.7,1.,1.,1.,1.,1.,1.,1.,1.,1.]))
    def distribution(self,state,lidar,dynamic):
        features=self.fusion(torch.cat([self.cnn(lidar),self.dynamic(dynamic.flatten(1)),state/self.state_scale],-1))
        a=(1+nn.functional.softplus(self.alpha(features))).clamp(max=100);b=(1+nn.functional.softplus(self.beta(features))).clamp(max=100)
        return torch.distributions.Independent(torch.distributions.Beta(a,b),1),self.critic(features).squeeze(-1)
    def forward(self,state,lidar,dynamic):
        dist,value=self.distribution(state,lidar,dynamic);return self.low+(self.high-self.low)*dist.mean
    @torch.inference_mode()
    def plan(self,observation):
        device=self.low.device;obs=[torch.as_tensor(x,dtype=torch.float32,device=device) for x in observation];return self(*obs).cpu().numpy()
    @classmethod
    def load(cls,path,device='cpu'):
        payload=torch.load(path,map_location='cpu',weights_only=True);model=cls(False);model.load_state_dict(payload['model']);return model.to(device).eval(),payload.get('metadata',{})
