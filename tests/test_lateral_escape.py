import numpy as np
from go2_nav.lateral_escape import LateralEscape


def scan(front=.3,left=2.,right=2.):
    s=np.zeros((1,36,4));s[0,:,1:]=3.9
    for ids,r in [(range(34,36),front),(range(0,3),front),(range(7,12),left),(range(25,30),right)]:
        for k in ids:s[0,k,0]=1;s[0,k,1:]=4-r
    return s


def active(s=None):
    m=LateralEscape();s=scan() if s is None else s
    for k in range(35):m.update([0,0],0,[10,0],s,k*.1)
    assert m.active
    return m


def test_clear_side_strafes_while_yaw_stays_on_goal():
    m=active();cmd=m.command([1,0,.8],[0,0],0,[10,0],scan(left=2.,right=.3))
    assert m.phase=='slide' and abs(cmd[0])<1e-6 and cmd[1]>.1 and cmd[2]==0
    m.command([0,0,0],[0,0],.1,[10,0],scan())
    assert m.side==1


def test_unknown_side_never_authorizes_translation():
    s=np.zeros((1,36,4));m=active(s)
    cmd=m.command([1,1,0],[0,0],0,[10,0],s)
    assert np.allclose(cmd[:2],0) and abs(cmd[2])>0


def test_one_camera_dropout_pauses_without_switching_sides():
    s=scan(left=2.,right=.3);m=active(s);m.command([0,0,0],[0,0],0,[10,0],s)
    missing=s.copy();missing[0,7:12,0]=0
    cmd=m.command([1,0,0],[0,0],0,[10,0],missing)
    assert m.phase=='slide' and m.side==1 and np.allclose(cmd[:2],0)


def test_failed_lateral_motion_tries_other_side_before_turning_out():
    s=scan();m=active(s);m.command([0,0,0],[0,0],0,[10,0],s);first=m.side
    m.no_motion=2.1;m.command([0,0,0],[0,0],0,[10,0],s)
    assert m.phase=='slide' and m.side==-first
    m.no_motion=2.1;m.command([0,0,0],[0,0],0,[10,0],s)
    assert m.phase=='turn_out' and m.stats['fallback_turns']==1


def test_goal_alignment_precedes_lateral_translation():
    m=active();cmd=m.command([1,0,0],[0,0],1.,[10,0],scan())
    assert np.allclose(cmd[:2],0) and cmd[2]<0


def test_outward_move_reacquires_goal_without_waiting_for_goal_progress():
    m=active();m.heading=np.pi/2;m._phase('advance_out',[0,0])
    cmd=m.command([1,0,0],[0,1.3],np.pi/2,[10,0],scan(front=2.))
    assert m.phase=='align_goal' and np.allclose(cmd[:2],0) and m.stats['outward_moves']==1


def test_goal_change_resets_phase_and_attempts():
    m=active();m.tried={1,-1};m.update([0,0],0,[-10,0],scan(),4.)
    assert not m.active and m.phase=='goal' and not m.tried


def test_second_variant_uses_legacy_forward_gap_and_reacquires_after_outward_move():
    from go2_nav.lateral_escape import LateralV2Escape
    m=LateralV2Escape();s=scan(front=.8)
    for k in range(35):m.update([0,0],0,[10,0],s,k*.1)
    cmd=m.command([0,0,0],[0,0],0,[10,0],s)
    assert m.phase=='forward' and cmd[0]>0
    m.heading=np.pi/2;m._phase('advance_out',[0,0])
    m.command([0,0,0],[0,1.3],np.pi/2,[10,0],scan(front=2.))
    assert m.phase=='advance_out'
    m.command([0,0,0],[0,1.9],np.pi/2,[10,0],scan(front=2.))
    assert m.phase=='align_goal'


def test_hybrid_retains_legacy_escape_until_actual_goal_progress():
    from go2_nav.lateral_escape import LateralHybridEscape
    m=LateralHybridEscape();s=scan()
    for k in range(35):m.update([0,0],0,[10,0],s,k*.1)
    m._outward(np.array([0.,0.]),0,np.array([10,0]),s)
    assert m.phase=='legacy_out' and m.fallback.active
    command=m.command([0,0,0],[0,0],0,[10,0],s)
    assert np.isfinite(command).all()
    m.update([0,1.8],0,[10,0],s,3.6)
    assert m.fallback is not None and m.active  # outward alone is not success
    m.update([2,0],0,[10,0],s,3.7)
    assert m.fallback is None and not m.active and m.stats['legacy_exits']==1


def test_hybrid_goal_switch_discards_old_escape():
    from go2_nav.lateral_escape import LateralHybridEscape
    m=LateralHybridEscape();s=scan()
    for k in range(35):m.update([0,0],0,[10,0],s,k*.1)
    m._outward(np.array([0.,0.]),0,np.array([10,0]),s)
    m.update([0,0],0,[-10,0],s,4.)
    assert m.fallback is None and not m.active


def test_hybrid_blocked_forward_probe_does_not_loop_on_open_ray():
    from go2_nav.lateral_escape import LateralHybridEscape
    m=LateralHybridEscape();s=scan()
    for k in range(35):m.update([0,0],0,[10,0],s,k*.1)
    clear=scan(front=2.)
    m.command([0,0,0],[0,0],0,[10,0],clear)
    assert m.phase=='forward'
    m.no_motion=2.1
    m.command([0,0,0],[0,0],0,[10,0],clear)
    assert m.phase=='slide' and m.stats['forward_probe_failures']==1
