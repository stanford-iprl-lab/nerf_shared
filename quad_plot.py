import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import json

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

from load_nerf import get_nerf

torch.manual_seed(0)
np.random.seed(0)

# # hard coded "nerf" for testing. see below to import real nerf
def get_manual_nerf(name):
    if name =='empty':
        class FakeRenderer:
            @typechecked
            def get_density(self, points: TensorType["batch":..., 3]) -> TensorType["batch":...]:
                return torch.zeros_like( points[...,0] )
        return FakeRenderer()

    if name =='cylinder':
        class FakeRenderer:
            @typechecked
            def get_density(self, points: TensorType["batch":..., 3]) -> TensorType["batch":...]:
                x = points[..., 0]
                y = points[..., 1] - 1

                return torch.sigmoid( (2 -(x**2 + y**2)) * 8 )
        return FakeRenderer()

    raise ValueError


def plot_nerf(ax_map, nerf):
    # can plot nerf in matplotlib but hard to be interpretable
    pass


class System:
    def __init__(self, renderer, start_full_state, end_full_state, cfg):
        self.nerf = renderer.get_density

        self.T_final            = cfg['T_final']
        self.steps              = cfg['steps']
        self.lr                 = cfg['lr']
        self.epochs_init        = cfg['epochs_init']
        self.epochs_update      = cfg['epochs_update']
        self.fade_out_epoch     = cfg['fade_out_epoch']
        self.fade_out_sharpness = cfg['fade_out_sharpness']

        self.dt = self.T_final / self.steps

        self.g = torch.tensor([0,0,-10])

        # # create initial and final 3 states to constrain: position, velocity and possibly angle in the future
        # self.start_states = start_state[None,:] + torch.tensor([-1,0,1])[:,None] * self.dt * start_vel
        # self.end_states   = end_state[None,:]   + torch.tensor([-1,0,1])[:,None] * self.dt * end_vel  

        self.start_full_state = start_full_state
        self.start_action = torch.tensor([10.0,0,0,0])
        start_reduced_state = self.get_4d_state(start_full_state[None, :])[0,:]

        self.end_full_state = end_full_state
        self.end_action = torch.tensor([10.0,0,0,0])
        end_reduced_state = self.get_4d_state(end_full_state[None, :])[0,:]

        slider = torch.linspace(0, 1, self.steps)[1:-1, None]
        states = (1-slider) * start_reduced_state + slider * end_reduced_state 
        self.states = states.clone().detach().requires_grad_(True)

        #PARAM this sets the shape of the robot body point cloud
        body = torch.stack( torch.meshgrid( torch.linspace(-0.05, 0.05, 10),
                                            torch.linspace(-0.05, 0.05, 10),
                                            torch.linspace(-0.02, 0.02,  5)), dim=-1)
        self.robot_body = body.reshape(-1, 3)
        # self.robot_body = torch.zeros(1,3)

        self.epoch = 0

    def params(self):
        return [self.states]

    def get_states(self):
        start_full_states = torch.stack( [ self.next_state(self.start_full_state, self.start_action, -self.dt),
                                         self.start_full_state,
                                         self.next_state(self.start_full_state, self.start_action, self.dt)] , dim=0)

        start_states = self.get_4d_state(start_full_states)

        end_full_states = torch.stack( [ self.next_state(self.end_full_state, self.end_action, -self.dt),
                                         self.end_full_state,
                                         self.next_state(self.end_full_state, self.end_action, self.dt)], dim=0 )

        end_states = self.get_4d_state(end_full_states)

        ##solve
        ## (v1 + v2) = target_vel
        ## (v2 - v1)/dt = target_accel

        #target_accel = (start_rot_matrix @ torch.tensor([0,0,1.0])) * start_thrust + self.g

        #v2 = (target_vel + target_accel * self.dt)/2
        #v1 = (target_vel - target_accel * self.dt)/2

        #self.start_states = start_state[None,:] + self.dt * torch.stack([ -v1, torch.zeros(3), v2], dim=0)

        return torch.cat( [start_states, self.states, end_states], dim=0)

    @staticmethod
    def get_4d_state(states):
        pos = states[:, 0:3]
        v   = states[:, 3:6]
        R_flat = states[:, 6:15]
        R = R_flat.reshape((-1, 3, 3))
        # omega = self.states[-1, 15:]

        # print(R.shape)
        forward = R @ torch.tensor( [1.0, 0, 0 ] )
        # print(forward.shape)
        x = forward[:,0]
        y = forward[:,1]
        angle = torch.atan2(y, x)

        # print(pos.shape)
        # print(angle.shape)
        # print(pos.shape)

        return torch.cat( [pos, angle[:,None] ], dim = -1).detach()


    def get_actions(self):
        self.mass = 1
        self.J = torch.eye(3)

        rot_matrix, z_accel = self.get_rots_and_accel()

        # print(rot_matrix.shape)
        # print(z_accel.shape)



        #TODO horrible -> there should be a better way without rotation matricies
        #calculate angular velocities
        ang_vel = rot_matrix_to_vec( rot_matrix[1:, ...] @ rot_matrix[:-1, ...].swapdims(-1,-2) ) / self.dt

        # if not torch.allclose( rot_matrix @ rot_matrix.swapdims(-1,-2), torch.eye(3)):
        #     print( rot_matrix @ rot_matrix.swapdims(-1,-2), torch.eye(3) )
        #     assert False

        #calculate angular acceleration
        angular_accel = (ang_vel[1:,...] - ang_vel[:-1,...])/self.dt

        # S, 3    3,3      S, 3, 1
        torques = (self.J @ angular_accel[...,None])[...,0]

        # print(z_accel.shape)
        # print(torques.shape)

        middle_action = torch.cat([ z_accel*self.mass, torques ], dim=-1)

        # print(middle_action.shape)
        # print(self.start_action.shape)
        # print(self.end_action.shape)

        return middle_action
        # return torch.cat([ self.start_action[None,:], middle_action, self.end_action[None,:] ], dim=0)

    def get_rots_and_accel(self, return_z_accel = False):

        states = self.get_states()
        # print(states.shape)
        prev_state = states[:-1, :]
        next_state = states[1:, :]

        diff = (next_state - prev_state)/self.dt
        vel = diff[..., :3]

        prev_vel = vel[:-1, :]
        next_vel = vel[1:, :]
        current_vel = (prev_vel + next_vel)/2

        target_accel = (next_vel - prev_vel)/self.dt - self.g
        z_accel     = torch.norm(target_accel, dim=-1, keepdim=True)

        # needs to be pointing in direction of acceleration
        z_axis_body = target_accel/z_accel

        #duplicate first and last angle to enforce zero angular velocity constraint
        # z_axis_body = torch.cat( [ z_axis_body[:1,:], z_axis_body, z_axis_body[-1:,:]], dim=0)

        z_angle = states[1:-1,3]
        in_plane_heading = torch.stack( [torch.sin(z_angle), -torch.cos(z_angle), torch.zeros_like(z_angle)], dim=-1)

        x_axis_body = torch.cross(z_axis_body, in_plane_heading, dim=-1)
        x_axis_body = x_axis_body/torch.norm(x_axis_body, dim=-1, keepdim=True)
        y_axis_body = torch.cross(z_axis_body, x_axis_body, dim=-1)

        # S, 3, 3 # assembled manually from basis vectors
        rot_matrix = torch.stack( [x_axis_body, y_axis_body, z_axis_body], dim=-1)

        # # add known start and end matricies
        # R_start = self.start_full_state[6:15].reshape(1, 3,3)
        # R_next = self.next_state(self.start_full_state, self.start_action, self.dt)[6:15].reshape(1,3,3)

        start_full_states = torch.stack( [ self.next_state(self.start_full_state, self.start_action, -self.dt),
                                         self.start_full_state,
                                         self.next_state(self.start_full_state, self.start_action, self.dt)] , dim=0)

        R_start = start_full_states[:,6:15].reshape(-1, 3,3)

        end_full_states = torch.stack( [ self.next_state(self.end_full_state, self.end_action, -self.dt),
                                         self.end_full_state,
                                         self.next_state(self.end_full_state, self.end_action, self.dt)], dim=0 )

        R_end = end_full_states[:,6:15].reshape(-1, 3,3)

        # print(rot_matrix.shape)
        # exit()

        # print(rot_matrix.shape)
        # R_end = self.end_full_state[6:15].reshape(1, 3,3)
        # R_last = self.next_state(self.end_full_state, self.end_action, -self.dt)[6:15].reshape(1,3,3)
        rot_matrix = torch.cat( [R_start, rot_matrix[2:-2,...], R_end], dim=0)

        # print(z_accel.shape)
        # print(rot_matrix.shape)
        # exit()
        # pos, vel, rot_matrix, omega
        # ang_vel = rot_matrix_to_vec( rot_matrix[1:, ...] @ rot_matrix[:-1, ...].swapdims(-1,-2) ) / self.dt
        return rot_matrix, z_accel



    @typechecked
    def get_next_action(self) -> TensorType["state_dim"]:
        # return self.get_actions()[1, :]
        return (self.get_actions()[1, :] + self.start_action.clone().detach())/2
        return self.get_actions()[0, :]


        # next_action_index = self.start_states.shape[0] - 3
        # fz, tx, ty, tz
        return self.start_action.clone().detach()

    @typechecked
    def update_state(self, measured_state: TensorType["state_dim"], action):
        # measured_state = measured_state[None, :]
        # measured_vel = measured_vel[None, :]
        # print(self.start_states.shape)
        # print(measured_state.shape)

        self.start_action = self.get_actions()[1,:].clone().detach()
        # self.start_action = action.clone().detach()

        self.start_full_state = measured_state.clone().detach()

        # self.start_states = torch.cat( [self.start_states[:-1, :], measured_state, measured_state + measured_vel * self.dt], dim=0 )

        self.states = self.states[1:, :].detach().requires_grad_(True)

    @typechecked
    def next_state(self, state: TensorType[18], action: TensorType[4], dt):
        #State is 18 dimensional [pos(3), vel(3), R (9), omega(3)] where pos, vel are in the world frame, R is the rotation from points in the body frame to world frame
        # and omega are angular rates in the body frame
        next_state = torch.zeros(18)

        #Actions are [total thrust, torque x, torque y, torque z]
        fz = action[0]
        tau = action[1:]

        #Define state vector
        pos = state[0:3]
        v   = state[3:6]
        R_flat = state[6:15]
        R = R_flat.reshape((3, 3))
        omega = state[15:]

        # The acceleration
        sum_action = torch.zeros(3)
        sum_action[2] = fz

        # print(self.mass*self.g)
        # print(R.shape)
        # print(sum_action)
        dv = (self.mass * self.g + R @ sum_action)/self.mass

        # The angular accelerations
        domega = torch.inverse(self.J) @ (tau - torch.cross(omega, self.J @ omega))

        # Propagate rotation matrix using exponential map of the angle displacements
        angle = omega*dt
        theta = torch.norm(angle, p=2)
        if theta == 0:
            exp_i = torch.eye(3)
        else:
            exp_i = torch.eye(3)
            angle_norm = angle/theta
            K = skew_matrix(angle_norm)

            exp_i = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

        next_R = R @ exp_i

        next_state[0:3] = pos + v * dt
        next_state[3:6] = v + dv * dt

        next_state[6:15] = next_R.reshape(-1)

        next_state[15:] = omega + domega * dt

        return next_state

    @typechecked
    def body_to_world(self, points: TensorType["batch", 3]) -> TensorType["states", "batch", 3]:
        states = self.get_states()
        pos = states[:, :3]
        rot_matrix, _ = self.get_rots_and_accel()

        # S, 3, P    =    S,3,3       3,P       S, 3, _
        world_points =  rot_matrix @ points.T + pos[..., None]
        return world_points.swapdims(-1,-2)

    def get_cost(self):
        actions = self.get_actions()

        fz = actions[:, 0]
        torques = torch.norm(actions[:, 1:], dim=-1)**2

        states = self.get_states()
        prev_state = states[:-1, :]
        next_state = states[1:, :]

        # multiplied by distance to prevent it from just speed tunnelling
        distance = torch.sum( (next_state - prev_state)[...,:3]**2 + 1e-5, dim = -1)**0.5
        density = self.nerf( self.body_to_world(self.robot_body)[1:,...] )**2
        colision_prob = torch.mean( density, dim = -1) * distance
        colision_prob = colision_prob[1:]
        # colision_prob = torch.cat([ colision_prob[0, None], colision_prob], dim=0)

        # if self.epoch < self.fade_out_epoch:
        #     t = torch.linspace(0,1, colision_prob.shape[0])
        #     position = self.epoch/self.fade_out_epoch
        #     mask = torch.sigmoid(self.fade_out_sharpness * (position - t))
        #     colision_prob = colision_prob * mask

        # print("here")
        # print(fz.shape)
        # print(torques.shape)
        # print(colision_prob.shape)

        #PARAM cost function shaping
        return 1000*fz**2 + 0.01*torques**2 + colision_prob * 1e6, colision_prob*1e6

    def total_cost(self):
        total_cost, colision_loss = self.get_cost()
        return torch.mean(total_cost)

    def learn_init(self):
        opt = torch.optim.Adam(self.params(), lr=self.lr)

        try:
            for it in range(self.epochs_init):
                opt.zero_grad()
                self.epoch = it
                loss = self.total_cost()
                print(it, loss)
                loss.backward()
                opt.step()

                save_step = 50
                if it%save_step == 0:
                    self.save_poses("paths/"+str(it//save_step)+"_testing.json")

        except KeyboardInterrupt:
            print("finishing early")

    def learn_update(self):
        opt = torch.optim.Adam(self.params(), lr=self.lr)

        for it in range(self.epochs_update):
            opt.zero_grad()
            self.epoch = it
            loss = self.total_cost()
            # print(it, loss)
            loss.backward()
            opt.step()

            # save_step = 50
            # if it%save_step == 0:
        # self.save_poses("paths/"+str(it//save_step)+"_testing.json")


    def plot(self, fig = None):
        if fig == None:
            fig = plt.figure(figsize=(16, 8))

        ax_map = fig.add_subplot(1, 2, 1, projection='3d')
        ax_graph = fig.add_subplot(1, 2, 2)
        self.plot_map(ax_map)
        plot_nerf(ax_map, self.nerf)

        self.plot_graph(ax_graph) 
        plt.tight_layout()
        plt.show()

    def plot_graph(self, ax):
        actions = self.get_actions().detach().numpy() 
        ax.plot(actions[...,0], label="fz")
        ax.plot(actions[...,1], label="tx")
        ax.plot(actions[...,2], label="ty")
        ax.plot(actions[...,3], label="tz")

        # states = self.states.detach().numpy()
        # ax.plot(states[...,0], label="px")
        # ax.plot(states[...,4], label="vx")
        # ax.plot(states[...,7], label="ey")

        ax_right = ax.twinx()

        total_cost, colision_loss = self.get_cost()
        ax_right.plot(total_cost.detach().numpy(), 'black', label="cost")
        ax_right.plot(colision_loss.detach().numpy(), 'cyan', label="colision")
        ax.legend()
        ax_right.legend()

    def plot_map(self, ax):
        ax.auto_scale_xyz([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        ax.set_ylim3d(-1, 1)
        ax.set_xlim3d(-1, 1)
        ax.set_zlim3d( 0, 1)

        # PLOT PATH
        # S, 1, 3
        pos = self.body_to_world( torch.zeros((1,3))).detach().numpy()
        # print(pos.shape)
        ax.plot( pos[:,0,0], pos[:,0,1],   pos[:,0,2],  )

        # PLOTS BODY POINTS
        # S, P, 2
        body_points = self.body_to_world( self.robot_body ).detach().numpy()
        for i, state_body in enumerate(body_points):
            if i == 0:
                color = 'k.'
            elif i == 1:
                color = 'r.'
            elif i == 2:
                color = 'b.'
            else:
                color = 'g.'
            ax.plot( *state_body.T, color, ms=72./ax.figure.dpi, alpha = 0.5)

        # PLOTS AXIS
        # create point for origin, plus a right-handed coordinate indicator.
        size = 0.05
        points = torch.tensor( [[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]])
        colors = ["r", "g", "b"]

        # S, 4, 2
        points_world_frame = self.body_to_world(points).detach().numpy()
        for state_axis in points_world_frame:
            for i in range(1, 4):
                ax.plot(state_axis[[0,i], 0],
                        state_axis[[0,i], 1],
                        state_axis[[0,i], 2],
                    c=colors[i - 1],)


    def save_poses(self, filename):
        states = self.get_states()
        rot_mats, _ = self.get_rots_and_accel()

        with open(filename,"w+") as f:
            for pos, rot in zip(states[...,:3], rot_mats):
                pose = np.zeros((4,4))
                pose[:3, :3] = rot.detach().numpy()
                pose[:3, 3]  = pos.detach().numpy()
                pose[3,3] = 1

                json.dump(pose.tolist(), f)
                f.write('\n')


class Simulator:
    @typechecked
    def __init__(self, start_pos: TensorType[3], start_vel: TensorType[3]):
        self.states = torch.cat( [start_pos, start_vel, torch.eye(3).reshape(-1), torch.zeros(3)], dim=0 )
        self.states = self.states[None,:]

        self.mass = 1
        self.I = torch.eye(3)
        self.invI = torch.eye(3)
        self.dt = 0.1 /5
        self.g = 10

    @typechecked
    def advance(self, action: TensorType[4]):
        for _ in range(5):
            self.internal_advance(action)

    @typechecked
    def internal_advance(self, action: TensorType[4]):
        next_state = self.next_state(self.states[-1, :], action)
        self.states = torch.cat( [self.states, next_state[None,:] ], dim=0 )

    def get_currrent_state(self):
        return self.states[-1,:]

    # def get_4d_state(self, states):
    #     pos = states[:, 0:3]
    #     v   = states[:, 3:6]
    #     R_flat = states[:, 6:15]
    #     R = R_flat.reshape((-1, 3, 3))
    #     # omega = self.states[-1, 15:]

    #     forward = R @ torch.tensor( [1.0, 0, 0 ] )
    #     x = forward[:,0]
    #     y = forward[:,1]
    #     angle = torch.atan2(y, x)

    #     return torch.cat( [pos, torch.tensor([angle]) ], dim = -1).detach()

    @typechecked
    def body_to_world(self, points: TensorType["batch", 3]) -> TensorType["states", "batch", 3]:
        pos = self.states[:, 0:3]
        v   = self.states[:, 3:6]
        R_flat = self.states[:, 6:15]
        R = R_flat.reshape((-1, 3, 3))
        omega = self.states[:, 15:]

        # S, 3, P    =    S,3,3       3,P       S, 3, _
        world_points =  R @ points.T + pos[..., None]
        return world_points.swapdims(-1,-2)

    @typechecked
    def next_state(self, state: TensorType[18], action: TensorType[4]):
        #State is 18 dimensional [pos(3), vel(3), R (9), omega(3)] where pos, vel are in the world frame, R is the rotation from points in the body frame to world frame
        # and omega are angular rates in the body frame
        next_state = torch.zeros(18)

        #Actions are [total thrust, torque x, torque y, torque z]
        fz = action[0]
        tau = action[1:]

        #Define state vector
        pos = state[0:3]
        v   = state[3:6]
        R_flat = state[6:15]
        R = R_flat.reshape((3, 3))
        omega = state[15:]

        # The acceleration
        sum_action = torch.zeros(3)
        sum_action[2] = fz

        dv = (torch.tensor([0,0,-self.mass*self.g]) + R @ sum_action)/self.mass

        # The angular accelerations
        domega = self.invI @ (tau - torch.cross(omega, self.I @ omega))

        # Propagate rotation matrix using exponential map of the angle displacements
        angle = omega*self.dt
        theta = torch.norm(angle, p=2)
        if theta == 0:
            exp_i = torch.eye(3)
        else:
            exp_i = torch.eye(3)
            angle_norm = angle/theta
            K = skew_matrix(angle_norm)

            exp_i = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

        next_R = R @ exp_i

        next_state[0:3] = pos + v * self.dt
        next_state[3:6] = v + dv * self.dt

        next_state[6:15] = next_R.reshape(-1)

        next_state[15:] = omega + domega * self.dt

        return next_state


def main():

    #PARAM initial and final velocities
    start_vel = torch.tensor([0, 0, 0, 0])
    end_vel   = torch.tensor([0, 0, 0, 0])

    renderer = get_manual_nerf("empty")
    start_state = torch.tensor([-0.05,-0.9, 0.2, 0])
    end_state   = torch.tensor([-0.2 , 0.7, 0.15 , 0])

    cfg = {"T_final": 2,
            "steps": 20,
            "lr": 0.001,
            "epochs_init": 2500,
            "fade_out_epoch": 500,
            "fade_out_sharpness": 10,
            "epochs_update": 200,
            }

    start = torch.cat( [start_state[:3], start_vel[:3], torch.eye(3).reshape(-1), torch.zeros(3)], dim=0 )
    end   = torch.cat( [end_state[:3], end_vel[:3], torch.eye(3).reshape(-1), torch.zeros(3)], dim=0 )

    traj = System(renderer, start, end, cfg)
    traj.learn_init()
    print(traj.get_next_action())
    traj.plot()

    sim = Simulator(start_state[:3], start_vel[:3])


    if True:
        for step in range(cfg['steps']):
            # # idealy something like this but we jank it for now
            
            action = traj.get_next_action()
            # action = torch.zeros(4)
            sim.advance(action)
            current_state = sim.get_currrent_state()
            # current_state = sim.advance(traj.get_actions()[0,:])
            # current_state = sim.advance(traj.get_actions()[1,:])

            print(action)
            print(current_state)
            print(traj.next_state(traj.start_full_state, traj.start_action, traj.dt))
            # we jank it

            # real_state, real_vel = sim.get_4d_state()
            # predicted_state = traj.states[1, :].detach()
            # print("action", action)
            # print("real", real_state)
            # print("pred", predicted_state)

            traj.update_state(sim.get_currrent_state(), action)
            traj.learn_update()
            # traj.save_poses(???)
            traj.plot()
            print("sim step", step)

    #PARAM file to save the trajectory
    # traj.save_poses("paths/playground_testing.json")
    # traj.plot()

@typechecked
def rot_matrix_to_vec( R: TensorType["batch":..., 3, 3]) -> TensorType["batch":..., 3]:
    batch_dims = R.shape[:-2]

    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)

    def acos_safe(x, eps=1e-4):
        """https://github.com/pytorch/pytorch/issues/8069"""
        slope = np.arccos(1-eps) / eps
        # TODO: stop doing this allocation once sparse gradients with NaNs (like in
        # th.where) are handled differently.
        buf = torch.empty_like(x)
        good = abs(x) <= 1-eps
        bad = ~good
        sign = torch.sign(x[bad])
        buf[good] = torch.acos(x[good])
        buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
        return buf

    angle = acos_safe((trace - 1) / 2)[..., None]
    # print(trace, angle)

    vec = (
        1
        / (2 * torch.sin(angle + 1e-5))
        * torch.stack(
            [
                R[..., 2, 1] - R[..., 1, 2],
                R[..., 0, 2] - R[..., 2, 0],
                R[..., 1, 0] - R[..., 0, 1],
            ],
            dim=-1,
        )
    )

    # needed to overwrite nanes from dividing by zero
    vec[angle[..., 0] == 0] = torch.zeros(3, device=R.device)

    # eg TensorType["batch_size", "views", "max_objects", 3, 1]
    rot_vec = (angle * vec)[...]

    return rot_vec

@typechecked
def vec_to_rot_matrix(rot_vec: TensorType["batch":..., 3]) -> TensorType["batch":..., 3,3]:
    assert not torch.any(torch.isnan(rot_vec))

    angle = torch.norm(rot_vec, dim=-1, keepdim=True)
    axis = rot_vec / (1e-5 + angle)
    S = skew_matrix(axis)
    # print(S.shape)
    # print(angle.shape)
    angle = angle[...,None]
    rot_matrix = (
            torch.eye(3)
            + torch.sin(angle) * S
            + (1 - torch.cos(angle)) * S @ S
            )
    return rot_matrix

@typechecked
def skew_matrix(vec: TensorType["batch":..., 3]) -> TensorType["batch":..., 3,3]:
    batch_dims = vec.shape[:-1]
    S = torch.zeros(*batch_dims, 3, 3)
    S[..., 0, 1] = -vec[..., 2]
    S[..., 0, 2] =  vec[..., 1]
    S[..., 1, 0] =  vec[..., 2]
    S[..., 1, 2] = -vec[..., 0]
    S[..., 2, 0] = -vec[..., 1]
    S[..., 2, 1] =  vec[..., 0]
    return S



if __name__ == "__main__":
    main()
