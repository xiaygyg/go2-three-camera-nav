import numpy as np
import torch
from go2_nav.robot import DEFAULT,PERM,LOW,HIGH,proprioception,SeaGo2Lower


def test_go2_proprioception_matches_source_order_and_scales():
    q=DEFAULT+np.arange(12)*.1;dq=np.arange(12)*2
    p=proprioception(np.eye(3),[1,2,3],q,dq,[1,.2,.4],np.arange(12))
    assert p.shape==(45,)
    assert np.allclose(p[:9],[.25,.5,.75,0,0,-1,2,.4,.1])
    assert np.allclose(p[9:21],np.arange(12)[PERM]*.1)
    assert np.allclose(p[21:33],dq[PERM]*.05)
    assert np.array_equal(p[33:],np.arange(12))
    assert np.array_equal(np.arange(12)[PERM][PERM],np.arange(12))


def test_released_models_accept_exact_450_history_and_64_actor_inputs():
    lower=SeaGo2Lower();prop=torch.zeros(2,45);prop[:,5]=-1
    action=lower(prop,prop[:,None].expand(-1,10,-1).clone())
    assert action.shape==(2,12) and torch.isfinite(action).all()
    assert all(not p.requires_grad for p in lower.body.parameters())


def test_go2_model_has_physical_mass_meshes_and_motor_limits():
    from go2_nav.model import make_model
    model=make_model('empty')
    assert 10<model.body_mass.sum()<25 and model.nu==12 and model.nmesh>=16
    assert model.opt.timestep==.005
    assert np.allclose(model.actuator_ctrlrange[[0,1]],[-23.7,23.7])
    assert np.allclose(model.actuator_ctrlrange[2],[-45.43,45.43])
    assert not np.any(model.geom_group==3)


def test_three_camera_go2_reset_and_articulated_step():
    from go2_nav.recovery_env import RecoveryGo2Batch
    env=RecoveryGo2Batch(1,'forest',2222,horizon=30,camera_config='configs/go2_three_d435i.json',camera_noise=False)
    try:
        env.reset(fixed_start=[0,0],fixed_goal=[1.1279,18.76877])
        before=env.data[0].qpos.copy()
        for _ in range(12):obs,reward,done,term=env.advance([[.5,0,0]],auto_reset=False)
        assert env.history.shape==(1,10,45) and env.last.shape==(1,12)
        assert not term[0]['fall'] and np.isfinite(reward).all()
        assert not np.allclose(before[7:],env.data[0].qpos[7:])
        assert obs[0].shape==(1,15) and obs[1].shape==(1,1,36,4)
        assert obs[1][0,0,18,0]==0  # rear is unknown, not completed from geometry
    finally:env.depth_sensor.close()


def test_go2_network_commands_use_go2_limits():
    from go2_nav.policy import Go2NavigationPolicy
    policy=Go2NavigationPolicy(False)
    actions=policy(torch.zeros(3,15),torch.zeros(3,1,36,4),torch.zeros(3,1,5,10))
    assert torch.all(actions>=torch.tensor(LOW)) and torch.all(actions<=torch.tensor(HIGH))
