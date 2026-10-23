"""Go2 Menagerie dynamics/meshes plus the shared forest and three RGB-D cameras."""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
from go2_nav.common.scenes import populate
from go2_nav.common.three_camera_depth import attach_cameras

ROOT=Path(__file__).resolve().parents[1]


def make_model(scenario='forest',goal=(7,0),camera_config=None):
    folder=ROOT/'assets/go2/downloaded/menagerie'
    tree=ET.parse(folder/'go2.xml');root=tree.getroot()
    root.find('compiler').set('meshdir',str(folder/'assets'))
    root.find('option').set('timestep','.005')
    root.find('option').set('gravity','0 0 -9.81')
    root.find(".//default[@class='visual']/geom").set('group','0')
    root.find(".//default[@class='collision']/geom").set('group','1')
    # Low joint damping with SEA PD instead of the Menagerie demo's extra
    # 2 Nm s/rad. This MuJoCo adaptation is validated by command tracking tests.
    joint=root.find(".//default[@class='go2']/joint")
    joint.set('damping','.01');joint.set('frictionloss','0')
    keyframe=root.find('keyframe')
    if keyframe is not None:root.remove(keyframe)
    world=root.find('worldbody')
    ET.SubElement(world,'light',pos='0 0 8',dir='0 0 -1',directional='true')
    ET.SubElement(world,'geom',name='ground',type='plane',size='100 60 .1',rgba='.25 .32 .17 1',friction='.8 .02 .001')
    marker=ET.SubElement(world,'body',name='goal_marker',mocap='true',pos=f'{goal[0]} {goal[1]} .04')
    ET.SubElement(marker,'geom',type='cylinder',size='.2 .04',density='0',group='2',contype='0',conaffinity='0',rgba='.1 1 .2 1')
    populate(world,scenario)
    if camera_config is not None:attach_cameras(world,camera_config)
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'headlight',ambient='.5 .5 .5',diffuse='.7 .7 .7')
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
