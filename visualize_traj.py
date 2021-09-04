import math
import os
import random
import sys
import glob
import json
import quaternion

import numpy as np
import cv2
import imageio

import habitat_sim
from habitat_sim.utils import viz_utils as vut
from habitat_sim.gfx import (
    DEFAULT_LIGHTING_KEY,
    NO_LIGHT_KEY,
    LightInfo,
    LightPositionModel,
)

import matplotlib.pyplot as plt

rot_z = lambda th : np.array([
    [np.cos(th),-np.sin(th),0],
    [np.sin(th), np.cos(th),0],
    [0,0,1]])

rot_y = lambda phi : np.array([
    [np.cos(phi),0, np.sin(phi)],
    [0,1,0],
    [-np.sin(phi), 0, np.cos(phi)]])

rot_x = lambda psi : np.array([
    [1 ,0, 0],
    [0,np.cos(psi), -np.sin(psi)],
    [0, np.sin(psi), np.cos(psi)]])

scene_dir = './scenes/playground_test/scene.gltf'
pose_dir = './paths/49_testing.json'
object_dir = './scenes/objects'
output_path = './media/'

H = 800
W = 800

make_video = True
show_video = True

pos_overlay = (20, 20)
overlay_res = (200, 200)

sim_settings = {
    "scene": scene_dir,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 0.5,  # Height of sensors in meters, relative to the agent
    "width": W,  # Spatial resolution of the observations
    "height": H,
}

def convert_blender_to_sim_pose(pose):
    #Incoming pose converts body canonical frame to world canonical frame. We want a pose conversion from body
    #sim frame to world sim frame.
    world2sim = np.array([[1., 0., 0.],
                        [0., 0., 1.],
                        [0., -1., 0.]])
    body2cam = world2sim
    rot = pose[:3, :3]          #Rotation from body to world canonical
    trans = pose[:3, 3]

    rot_c2s = world2sim @ rot @ body2cam.T
    trans_sim = world2sim @ trans

    print('Trans', trans)
    print('Trans sim', trans_sim)

    c2w = np.zeros((4, 4))
    c2w[:3, :3] = rot_c2s
    c2w[:3, 3] = trans_sim
    c2w[3, 3] = 1.

    return c2w

def load_poses(pose_dir):
    files = []
    for file in glob.glob('pose_dir' + '*.json'):
        files.append(file)

    print('Files', files)
    poses = []    
    for file in files:
        with open(file, 'r') as fp:
            meta = json.load(fp)
            poses += meta["frames"]

    return poses

def set_agent_state(pose, agent):

    #trans = pose[:3, 3] @ rot_x(np.pi/2)
    #rot = rot_x(-np.pi/2) @ pose[:3, :3]

    rot = pose[:3, :3]
    trans = pose[:3, 3]

    #Set state
    agent_state = habitat_sim.AgentState()
    agent_state.position = trans
    agent_state.rotation = quaternion.from_rotation_matrix(rot)
    agent.set_state(agent_state)

    agent_state = agent.get_state()
    print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
    print('Rotation Matrix', quaternion.as_rotation_matrix(agent_state.rotation))

    return

def remove_all_objects(sim):
    for id_ in sim.get_existing_object_ids():
        sim.remove_object(id_)

def place_agent(sim):
    # place our agent in the scene
    agent_state = habitat_sim.AgentState()
    agent_state.position = [0., 0., 0.]
    #agent_state.rotation = 
    agent = sim.initialize_agent(0, agent_state)
    return agent, agent.scene_node.transformation_matrix()

def make_configuration(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    assert os.path.exists(sim_cfg.scene_id)
    sim_cfg.enable_physics = False

    sensor_specs = []

    rgba_camera_1stperson_spec = habitat_sim.CameraSensorSpec()
    rgba_camera_1stperson_spec.uuid = "rgba_camera_1stperson"
    rgba_camera_1stperson_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgba_camera_1stperson_spec.resolution = [settings["height"], settings["width"]]
    rgba_camera_1stperson_spec.position = [0.0, 0., -0.1]
    rgba_camera_1stperson_spec.orientation = [0., 0., 0.0]
    rgba_camera_1stperson_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    rgba_camera_1stperson_spec.hfov = 40

    sensor_specs.append(rgba_camera_1stperson_spec)

    depth_camera_1stperson_spec = habitat_sim.CameraSensorSpec()
    depth_camera_1stperson_spec.uuid = "depth_camera_1stperson"
    depth_camera_1stperson_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_camera_1stperson_spec.resolution = [settings["height"], settings["width"]]
    depth_camera_1stperson_spec.position = [0.0, 0., 0.0]
    depth_camera_1stperson_spec.orientation = [0.0, 0., 0.0]
    depth_camera_1stperson_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    depth_camera_1stperson_spec.hfov = 40

    sensor_specs.append(depth_camera_1stperson_spec)

    rgba_camera_3rdperson_spec = habitat_sim.CameraSensorSpec()
    rgba_camera_3rdperson_spec.uuid = "rgba_camera_3rdperson"
    rgba_camera_3rdperson_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgba_camera_3rdperson_spec.resolution = [settings["height"], settings["width"]]
    rgba_camera_3rdperson_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgba_camera_3rdperson_spec.orientation = [-np.pi/2., 0., 0.0]
    rgba_camera_3rdperson_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    rgba_camera_3rdperson_spec.hfov = 75

    sensor_specs.append(rgba_camera_3rdperson_spec)

    # agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


################################### START OF MAIN ###########################################3
 # create the simulators AND resets the simulator

cfg = make_configuration(sim_settings)

cfg.sim_cfg.scene_light_setup = DEFAULT_LIGHTING_KEY
sim = habitat_sim.Simulator(cfg)

'''
###LIGHTS###
# define a directional light (w == 0)
light_setup = [LightInfo(vector=[1.0, 1.0, 1.0, 0.0])]

sim.set_light_setup(light_setup)
assert sim.get_light_setup() == light_setup

# ensure modifications to local light setup variable are not reflected in sim
#light_setup[0].model = LightPositionModel.Camera
#assert sim.get_light_setup() != light_setup

#sim.set_light_setup(light_setup, DEFAULT_LIGHTING_KEY)
#assert sim.get_light_setup() == light_setup

################
'''

#poses = load_poses(pose_dir)

poses = []
with open(pose_dir, 'r') as fp:
    meta = json.load(fp)
    poses = meta['poses']
    
# get the primitive assets attributes manager
prim_templates_mgr = sim.get_asset_template_manager()

# get the physics object attributes manager
obj_templates_mgr = sim.get_object_template_manager()

object_handle = obj_templates_mgr.load_configs(object_dir)[0]

drone_temp_handle = obj_templates_mgr.get_template_handles(object_dir + '/drone')[0]

#agent, agent_transform = place_agent(sim)
# initialize all agents
objects = []

agent = sim.initialize_agent(sim_settings["default_agent"])

'''
# add robot object to the scene with the agent/camera SceneNode attached
id_1 = sim.add_object(object_handle, sim.agents[0].scene_node)

# set one object to kinematic
sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, id_1)

'''

for object_num in range(len(poses)):
    objects.append(sim.add_object_by_handle(drone_temp_handle))

    # set one object to kinematic
    sim.set_object_motion_type(habitat_sim.physics.MotionType.KINEMATIC, objects[object_num])



observations = []
imgs = []

'''
# Convert poses into camera images as if agent is moving in trajectory

for ind, pose in enumerate(poses):
    pose = np.array(pose)
    pose = convert_blender_to_sim_pose(pose)
    set_agent_state(pose, agent)
    observations.append(sim.get_sensor_observations())

    img = sim.get_sensor_observations()['rgba_camera_3rdperson']
    img_overlay = sim.get_sensor_observations()['rgba_camera_1stperson']

    #Border color
    color = [128, 128, 128]

    # border widths; I set them all to 150
    top, bottom, left, right = [10]*4

    img_overlay = cv2.copyMakeBorder(img_overlay, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    img_overlay = cv2.resize(img_overlay, overlay_res, interpolation = cv2.INTER_AREA)

    x, y = pos_overlay[0], pos_overlay[1]
    img[x:x+overlay_res[0] , y:y+overlay_res[1]] = img_overlay
    imgs.append(img)

    plt.imsave(output_path + f'{ind}.png', img)
'''

imageio.mimwrite(os.path.join(output_path, 'video.gif'), imgs, fps=8)

# remove the agent's body while preserving the SceneNode
sim.remove_object(id_1, False)

sim.close()