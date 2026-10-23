import numpy as np
from go2_nav.common.recovery import EscapeMemory

def scan():
    s=np.zeros((1,36,4));s[0,:14,0]=1;s[0,23:,0]=1
    return s

def test_stall_detects_rotation_and_unknown_never_moves():
    m=EscapeMemory();s=np.zeros((1,36,4))
    for step in range(40):m.update([0,0],step*.1,[10,0],s,step*.1)
    assert m.active and m.activations==1
    for yaw in np.arange(0,6,.2):
        cmd=m.command([4,0,0],[0,0],yaw,[10,0],s)
        assert np.allclose(cmd[:2],0) and abs(cmd[2])>0

def test_turning_cannot_farm_novelty_and_escape_stays_persistent():
    m=EscapeMemory();s=scan()
    for step in range(100):m.update([0,0],step*.2,[10,0],s,step*.1)
    assert m.active and m.reward_paid<=3.000001
    first=m.command([0,0,0],[0,0],0,[10,0],s)
    heading=m.heading
    m.command([-1,1,-1],[0,0],.1,[10,0],s)
    assert m.heading==heading
    assert np.isfinite(first).all()

def test_observed_progress_exits_recovery_and_goal_change_clears_history():
    m=EscapeMemory();s=scan()
    for step in range(40):m.update([0,0],0,[10,0],s,step*.1)
    assert m.active
    m.update([2,0],0,[10,0],s,4.)
    assert not m.active and m.escapes==1
    m.update([2,0],0,[-10,0],s,4.1)
    assert len(m.history)==1 and not m.active

def test_initial_view_has_no_exploration_bonus_before_stall():
    m=EscapeMemory()
    assert m.update([0,0],0,[10,0],scan(),0)==0
    assert m.reward_paid==0 and not m.active

def test_recovery_itself_stalling_forces_persistent_scan():
    m=EscapeMemory();s=scan()
    for step in range(65):m.update([0,0],0,[10,0],s,step*.1)
    m.heading=0.
    command=m.command([0,0,0],[0,0],0,[10,0],s)
    assert np.allclose(command[:2],0) and abs(command[2])>.5
    assert m.scan_remaining==15 and m.heading is None
    for _ in range(10):
        command=m.command([1.7,0,0],[0,0],0,[10,0],s)
        assert np.allclose(command[:2],0) and abs(command[2])>.5

def test_blind_cbf_projection_does_not_cancel_selected_yaw():
    from go2_nav.recovery_env import camera_execution
    s=scan();obs=(np.zeros(15),s,np.zeros((1,5,10)))
    command,infeasible=camera_execution([2.,0.,.8],obs,[0,0],np.eye(2),[(np.array([.55,0]),.08,np.zeros(2))])
    assert np.allclose(command[:2],0)
    assert command[2]==.8

def test_revision2_execution_replays_original_float32_pipeline():
    from go2_nav.recovery_env import camera_execution
    from go2_nav.common.safety import safe_velocity
    from go2_nav.common.three_camera_depth import blind_guard
    rng=np.random.default_rng(1910);s=scan();obs=(np.zeros(15),s,np.zeros((1,5,10)))
    obstacles=[(np.array([1.,.4]),.08,np.zeros(2))]
    for _ in range(12):
        command=rng.uniform([-.5,-.8,-1.],[1.7,.8,1.]).astype(np.float32)
        old=command.copy();safe,_=safe_velocity(old[:2].astype(float),[0,0],obstacles,speed=float(np.hypot(1.7,.8)),barrier_gain=1.4)
        old[:2]=safe;old=blind_guard(old,obs,turn_to_observe=True)
        current,_=camera_execution(command,obs,[0,0],np.eye(2),obstacles,yaw_mode='projected')
        assert np.array_equal(current,old)
