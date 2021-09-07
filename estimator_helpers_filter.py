import numpy as np
from numpy.lib.function_base import vectorize
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import cv2
import skimage
import matplotlib.pyplot as plt
import time
import numpy.linalg as la

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

#Helper Functions
def find_POI(img_rgb, DEBUG=False): # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img_gray, None)

    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy # pixel coordinates

img2mse = lambda x, y : torch.mean((x - y) ** 2)

quad_loss = lambda x, y, M: torch.mean(((x - y).reshape((1, -1)) @ torch.inverse(M) @ (x - y).reshape((-1, 1))))

def vec2ss_matrix(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix

def state2pose(vector):
    pose = torch.zeros((4, 4))
    rot_flat = vector[6:15]
    rot = rot_flat.reshape((3, 3))
    trans = vector[:3]

    pose[:3, :3] = rot
    pose[:3, 3] = trans
    pose[3, 3] = 1.

    return pose


def convert_blender_to_sim_pose(pose):
    #Incoming pose converts body canonical frame to world canonical frame. We want a pose conversion from body
    #sim frame to world sim frame.
    world2sim = torch.tensor([[1., 0., 0.],
                        [0., 0., 1.],
                        [0., -1., 0.]])
    body2cam = world2sim
    rot = pose[:3, :3]          #Rotation from body to world canonical
    trans = pose[:3, 3]

    rot_c2s = world2sim @ rot @ body2cam.T
    trans_sim = world2sim @ trans

    c2w = torch.zeros((4, 4))
    c2w[:3, :3] = rot_c2s
    c2w[:3, 3] = trans_sim
    c2w[3, 3] = 1.

    return c2w

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

class state_transform(nn.Module):
    def __init__(self):
        super(state_transform, self).__init__()

        #All parameters defined as offsets
        self.psi = nn.Parameter(torch.normal(0., 1e-4, size=(1,)))
        self.phi = nn.Parameter(torch.normal(0., 1e-4, size=(1,)))
        self.v = nn.Parameter(torch.normal(0., 1e-4, size=(3,)))
        self.theta = nn.Parameter(torch.normal(0., 1e-4, size=(1,)))

    def forward(self, x):
        #x is initial estimate on the 18 dimensional state

        P = torch.zeros((4, 4))
        P[:3, :3] = x[6:15].reshape((3, 3))
        P[:3, 3] = x[:3]
        P[3, 3] = 1.

        theta = self.theta
        psi = self.psi
        phi = self.phi

        #convert w to spherical
        w = torch.cat((torch.cos(psi)*torch.sin(phi), torch.sin(psi)*torch.sin(phi), torch.cos(phi)))

        exp_i = torch.zeros((4,4))
        w_skewsym = vec2ss_matrix(w)

        #theta = self.theta
        #exp_i = torch.zeros((4,4))
        #w_skewsym = vec2ss_matrix(self.w)

        exp_i[:3, :3] = torch.eye(3) + torch.sin(theta) * w_skewsym + (1 - torch.cos(theta)) * torch.matmul(w_skewsym, w_skewsym)
        exp_i[:3, 3] = self.v #torch.matmul(torch.eye(3) * theta + (1 - torch.cos(theta)) * w_skewsym + (theta - torch.sin(theta)) * torch.matmul(w_skewsym, w_skewsym), self.v)
        exp_i[3, 3] = 1.

        T_i = torch.matmul(exp_i, P)
        #T_i = torch.matmul(P, exp_i)
        
        R_i = T_i[:3, :3]
        t_i = T_i[:3, 3]

        X = x.clone().detach()

        X[:3] = t_i
        X[6:15] = R_i.reshape(-1)

        return X

class Estimator():
    def __init__(self, N_iter, batch_size, sampling_strategy, renderer, agent, xt, sig, Q, dil_iter=3, kernel_size=5, lrate=.01, noise=None, sigma=0.01, amount=0.8, delta_brightness=0.) -> None:
    # Parameters
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.dil_iter = dil_iter

        self.lrate = lrate
        self.sampling_strategy = sampling_strategy
        #delta_phi, delta_theta, delta_psi, delta_t = args.delta_phi, args.delta_theta, args.delta_psi, args.delta_t
        self.noise, self.sigma, self.amount = noise, sigma, amount
        self.delta_brightness = delta_brightness

        self.renderer = renderer
        self.agent = agent

        #State initial estimate at time t=0
        self.xt = xt            #Size 18
        self.sig = sig          #State covariance 18x18
        self.Q = Q              #Process noise covariance

        self.iter = N_iter

        # create meshgrid from the observed image
        self.W, self.H, self.focal = self.renderer.hwf
        self.coords = np.asarray(np.stack(np.meshgrid(np.linspace(0, self.W - 1, self.W), np.linspace(0, self.H - 1, self.H)), -1),
                            dtype=int)

        #Storage for plots
        self.pixel_losses = []
        self.dyn_losses = []
        self.rot_errors = []
        self.trans_errors = []
        self.sig_det = []

    def optimize(self, start_state, sig, obs_img, obs_img_pose):

        obs_img = (np.array(obs_img) / 255.).astype(np.float32)

        # change brightness of the observed image
        if self.delta_brightness != 0:
            obs_img = (np.array(obs_img) / 255.).astype(np.float32)
            obs_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2HSV)
            if self.delta_brightness < 0:
                obs_img[..., 2][obs_img[..., 2] < abs(self.delta_brightness)] = 0.
                obs_img[..., 2][obs_img[..., 2] >= abs(self.delta_brightness)] += self.delta_brightness
            else:
                lim = 1. - self.delta_brightness
                obs_img[..., 2][obs_img[..., 2] > lim] = 1.
                obs_img[..., 2][obs_img[..., 2] <= lim] += self.delta_brightness
            obs_img = cv2.cvtColor(obs_img, cv2.COLOR_HSV2RGB)

        # apply noise to the observed image
        if self.noise == 'gaussian':
            obs_img_noised = skimage.util.random_noise(obs_img, mode='gaussian', var=self.sigma**2)
        elif self.noise == 's_and_p':
            obs_img_noised = skimage.util.random_noise(obs_img, mode='s&p', amount=self.amount)
        elif self.noise == 'pepper':
            obs_img_noised = skimage.util.random_noise(obs_img, mode='pepper', amount=self.amount)
        elif self.noise == 'salt':
            obs_img_noised = skimage.util.random_noise(obs_img, mode='salt', amount=self.amount)
        elif self.noise == 'poisson':
            obs_img_noised = skimage.util.random_noise(obs_img, mode='poisson')
        else:
            obs_img_noised = obs_img

        self.obs_img_noised = obs_img_noised

        obs_img_noised_POI = (np.array(obs_img_noised) * 255).astype(np.uint8)

        if self.sampling_strategy == 'interest_regions' or self.sampling_strategy == 'interest_points':
            # find points of interest of the observed image
            POI = find_POI(obs_img_noised_POI, False)  # xy pixel coordinates of points of interest (N x 2)

        #obs_img_noised = (np.array(obs_img_noised) / 255.).astype(np.float32)

        if self.sampling_strategy == 'interest_regions':
            # create sampling mask for interest region sampling strategy
            interest_regions = np.zeros((self.H, self.W, ), dtype=np.uint8)
            interest_regions[POI[:,1], POI[:,0]] = 1
            I = self.dil_iter
            interest_regions = cv2.dilate(interest_regions, np.ones((self.kernel_size, self.kernel_size), np.uint8), iterations=I)
            interest_regions = np.array(interest_regions, dtype=bool)
            interest_regions = self.coords[interest_regions]

        # not_POI contains all points except of POI
        coords = self.coords.reshape(self.H * self.W, 2)

        if self.sampling_strategy == 'interest_points':
            not_POI = set(tuple(point) for point in coords) - set(tuple(point) for point in POI)
            not_POI = np.array([list(point) for point in not_POI]).astype(int)

        # Create pose transformation model
        start_state = torch.Tensor(start_state).to(device)
        state_trans = state_transform().to(device)
        optimizer = torch.optim.Adam(params=state_trans.parameters(), lr=self.lrate, betas=(0.9, 0.999))

        # calculate angles and translation of the observed image's pose
        phi_ref = np.arctan2(obs_img_pose[1,0], obs_img_pose[0,0])*180/np.pi
        theta_ref = np.arctan2(-obs_img_pose[2, 0], np.sqrt(obs_img_pose[2, 1]**2 + obs_img_pose[2, 2]**2))*180/np.pi
        psi_ref = np.arctan2(obs_img_pose[2, 1], obs_img_pose[2, 2])*180/np.pi
        translation_ref = np.sqrt(obs_img_pose[0,3]**2 + obs_img_pose[1,3]**2 + obs_img_pose[2,3]**2)

        propagated_state = start_state.detach()

        new_lrate = self.lrate
        best_loss = 1e5
        best_state = start_state.detach()

        for k in range(self.iter):

            # Create pose transformation model
            #start_state = torch.Tensor(start_state).to(device)
            #state_trans = state_transform().to(device)
            #optimizer = torch.optim.Adam(params=state_trans.parameters(), lr=self.lrate, betas=(0.9, 0.999))

            if self.sampling_strategy == 'random':
                rand_inds = np.random.choice(coords.shape[0], size=self.batch_size, replace=False)
                batch = coords[rand_inds]

            elif self.sampling_strategy == 'interest_points':
                if POI.shape[0] >= self.batch_size:
                    rand_inds = np.random.choice(POI.shape[0], size=self.batch_size, replace=False)
                    batch = POI[rand_inds]
                else:
                    batch = np.zeros((self.batch_size, 2), dtype=np.int)
                    batch[:POI.shape[0]] = POI
                    rand_inds = np.random.choice(not_POI.shape[0], size=self.batch_size-POI.shape[0], replace=False)
                    batch[POI.shape[0]:] = not_POI[rand_inds]

            elif self.sampling_strategy == 'interest_regions':
                rand_inds = np.random.choice(interest_regions.shape[0], size=self.batch_size, replace=False)
                batch = interest_regions[rand_inds]

            else:
                print('Unknown sampling strategy')
                return

            target_s = obs_img_noised[batch[:, 1], batch[:, 0]]
            target_s = torch.Tensor(target_s).to(device)

            predict_state = state_trans(start_state)

            pose = state2pose(predict_state)

            sim_pose = convert_blender_to_sim_pose(pose)

            rgb = self.renderer.get_img_from_pix(batch, sim_pose, HW=False)

            #Process loss. 
            loss_dyn = quad_loss(predict_state, propagated_state, sig)

            optimizer.zero_grad()

            loss_rgb = img2mse(rgb, target_s)

            loss = loss_rgb #+ loss_dyn

            loss.backward()
            optimizer.step()

            if loss.cpu().detach().numpy() < best_loss and k > 0:
                best_loss = loss.cpu().detach().numpy()
                best_state = predict_state
                self.batch = batch

            #start_state = state_trans(start_state)
            #start_state = start_state.cpu().detach().numpy()

            new_lrate = self.lrate * (0.8 ** ((k + 1) / 100))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            with torch.no_grad():
                if (k + 1) % 20 == 0 or k == 0:
                    print('Step: ', k)
                    print('Loss: ', loss)
                    print('Pixel Loss', loss_rgb)
                    print('Dynamical Loss', loss_dyn)
                    print('Error between propagated and current', torch.norm(predict_state-propagated_state))

                    pose_dummy = sim_pose.cpu().detach().numpy()
                    # calculate angles and translation of the optimized pose
                    phi = np.arctan2(pose_dummy[1, 0], pose_dummy[0, 0]) * 180 / np.pi
                    theta = np.arctan2(-pose_dummy[2, 0], np.sqrt(pose_dummy[2, 1] ** 2 + pose_dummy[2, 2] ** 2)) * 180 / np.pi
                    psi = np.arctan2(pose_dummy[2, 1], pose_dummy[2, 2]) * 180 / np.pi
                    translation = np.sqrt(pose_dummy[0,3]**2 + pose_dummy[1,3]**2 + pose_dummy[2,3]**2)
                    #translation = pose_dummy[2, 3]
                    # calculate error between optimized and observed pose
                    phi_error = abs(phi_ref - phi) if abs(phi_ref - phi)<300 else abs(abs(phi_ref - phi)-360)
                    theta_error = abs(theta_ref - theta) if abs(theta_ref - theta)<300 else abs(abs(theta_ref - theta)-360)
                    psi_error = abs(psi_ref - psi) if abs(psi_ref - psi)<300 else abs(abs(psi_ref - psi)-360)
                    rot_error = phi_error + theta_error + psi_error
                    translation_error = abs(translation_ref - translation)

                    print('Rotation error: ', rot_error)
                    print('Translation error: ', translation_error)
                    print('-----------------------------------')

                #Store data
                self.pixel_losses.append(loss_rgb.cpu().detach().numpy().reshape(-1))
                self.dyn_losses.append(loss_dyn.cpu().detach().numpy().reshape(-1))
                self.rot_errors.append(rot_error)
                self.trans_errors.append(translation_error)
            
                if (k+1) % 300 == 0:
                    img_dummy = self.renderer.get_img_from_pose(sim_pose)
                    plt.figure()
                    plt.imsave('./paths/rendered_img.png', img_dummy.cpu().detach().numpy())
                    plt.close()

        return best_state.clone().detach()

    def measurement_function(self, state, start_state, sig):

        target_s = self.obs_img_noised[self.batch[:, 1], self.batch[:, 0]]
        target_s = torch.Tensor(target_s).to(device)

        pose = state2pose(state)

        sim_pose = convert_blender_to_sim_pose(pose)

        rgb = self.renderer.get_img_from_pix(self.batch, sim_pose, HW=False)

        #Process loss. 
        loss_dyn = quad_loss(state, start_state, sig)

        loss_rgb = img2mse(rgb, target_s)

        loss = loss_rgb + loss_dyn

        return loss

    def estimate_state(self, obs_img, obs_img_pose, action):
        # Computes Jacobian w.r.t dynamics are time t-1. Then update state covariance Sig_{t|t-1}.
        # Perform grad. descent on J = measurement loss + process loss
        # Compute state covariance Sig_{t} by hessian at state at time t.

        with torch.no_grad():
            #Propagated dynamics. x t|t-1
            start_state = self.agent.drone_dynamics(self.xt, action)

            #State estimate at t-1 is self.xt. Find jacobian wrt dynamics
            t1 = time.time()
            A = torch.autograd.functional.jacobian(lambda x: self.agent.drone_dynamics(x, action), self.xt)
            t2 = time.time()
            print('Elapsed time for Jacobian', t2-t1)
            print('Jacobian', A)

            #Propagate covariance
            sig_prop = A @ self.sig @ A.T + self.Q

        #Argmin of total cost. Encapsulate this argmin optimization as a function call
        xt = self.optimize(torch.tensor(start_state), torch.tensor(sig_prop), obs_img, obs_img_pose)

        with torch.no_grad():
            #Update state estimate
            self.xt = xt

            #Hessian to get updated covariance
            t3 = time.time()
            hess = torch.autograd.functional.hessian(lambda x: self.measurement_function(x, start_state, sig_prop), self.xt)

            #Turn covariance into positive definite
            hess_np = hess.cpu().detach().numpy()
            hess = nearestPD(hess_np)

            t4 = time.time()
            print('Elapsed time for hessian', t4-t3)
            #print('Hessian', sig)

            #self.sig_det.append(np.linalg.det(sig.cpu().numpy()))

            #Update state covariance
            self.sig = torch.inverse(torch.tensor(hess))

            print('Start state', start_state)

        return self.xt.clone().cpu().detach().numpy()
