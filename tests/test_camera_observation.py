import numpy as np
import pytest
from go2_nav.common.camera_observation import deproject_depth,pointcloud_scan,fuse_rectified_depth,pack_navigation_state

def test_optical_depth_projects_forward_and_invalid_is_not_free():
    # Optical x right/y down/z forward -> level body x forward/y left/z up.
    T=np.eye(4);T[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]
    K=np.array([[100,0,1],[0,100,1],[0,0,1.]])
    d=np.zeros((3,3));d[1,1]=2
    np.testing.assert_allclose(deproject_depth(d,K,T,1),[[2,0,0]])
    result=fuse_rectified_depth([dict(depth_m=np.zeros((3,3)),K=K,T_level_optical=T)],stride=1)
    assert not result['observed'].any()
    np.testing.assert_allclose(result['lidar'][0,:,1:],3.9)
    assert result['deployment_validated'] is False

def test_inflation_ground_filter_and_blind_rear():
    coverage=np.full(36,4.);coverage[18]=0
    lidar,observed,ranges=pointcloud_scan([[2,0,0],[.3,0,-.32]],coverage)
    assert ranges[0]==pytest.approx(1.45)
    assert not observed[18] and ranges[18]==.1
    assert lidar.shape==(1,36,4) and np.all(lidar[:,:,0]==0)
    assert ranges[9]==4

def test_state_matches_sim_scaling():
    state=pack_navigation_state([30,0],[4,0,0],[0,0,1.2],[1.7,.8,1.],.4)
    np.testing.assert_allclose(state[[0,1,2,3,4]],[1,0,0,20,0])
    np.testing.assert_allclose(state[8:11],[0,0,.3])
    np.testing.assert_allclose(state[11:14],[1,1,1])
    assert state[14]==pytest.approx(.16)

def test_reject_nonrigid_calibration():
    T=np.eye(4);T[0,0]=2
    with pytest.raises(ValueError):deproject_depth(np.ones((3,3)),np.eye(3),T)
