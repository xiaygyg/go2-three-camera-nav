import torch
from go2_nav.common.navigation_policy import NavigationPolicy
from .robot import LOW,HIGH


class Go2NavigationPolicy(NavigationPolicy):
    def __init__(self,warm_start=True):
        super().__init__(warm_start)
        self.set_go2_limits()

    def set_go2_limits(self):
        with torch.no_grad():
            self.low.copy_(torch.as_tensor(LOW,device=self.low.device))
            self.high.copy_(torch.as_tensor(HIGH,device=self.high.device))
            self.state_scale[5]=1.7
