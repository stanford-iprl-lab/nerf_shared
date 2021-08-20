import numpy as np
import torch

mseloss = lambda x, y : torch.mean((x - y) ** 2)

rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

trans_t = lambda t: np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]])

def vec2ss_matrix(vector):  # vector to skewsym. matrix

    ss_matrix = np.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix

def vec2ss_matrix_torch(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3,3))
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix


class camera_transf(nn.Module):
    def __init__(self):
        super(camera_transf, self).__init__()
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)))
        self.theta = nn.Parameter(torch.normal(0., 1e-6, size=()))

    def forward(self, x):
        exp_i = torch.zeros((4,4))
        w_skewsym = vec2ss_matrix(self.w)
        v_skewsym = vec2ss_matrix(self.v)
        exp_i[:3, :3] = torch.eye(3) + torch.sin(self.theta) * w_skewsym + (1 - torch.cos(self.theta)) * torch.matmul(w_skewsym, w_skewsym)
        exp_i[:3, 3] = torch.matmul(torch.eye(3) * self.theta + (1 - torch.cos(self.theta)) * w_skewsym + (self.theta - torch.sin(self.theta)) * torch.matmul(w_skewsym, w_skewsym), self.v)
        exp_i[3, 3] = 1.
        T_i = torch.matmul(exp_i, x)
        return T_i

class Estimator():
    def __init__(self, N_iter) -> None:

    batch_size = 512*4

    lrate = .01

    phi = 10.
    theta = 10.
    psi = 10.
    trans = 0.1

    I = 5

    kernel_size = 5

    sampling_strategy = 'interest_regions'

    H, W, focal = hwf

    coords = np.stack(np.meshgrid(np.linspace(0, H-1, H), np.linspace(0, W-1, W)), -1)  # (H, W, 2)

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
        print('HERE')

    t = time.time()

    obs_img_pose = render_poses[2, :3, :4]
    obs_img_pose = np.vstack((obs_img_pose, [0, 0, 0, 1.]))
    img_gt = gt_imgs[2]

    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img_gt,None)

    img_gt = torch.Tensor(img_gt).to(device)

    print(obs_img_pose)

    pose = rot_phi(phi/180.*np.pi) @ rot_theta(theta/180.*np.pi) @ rot_psi(psi/180.*np.pi) @ trans_t(trans) @ obs_img_pose
    #pose = trans_t(trans) @ obs_img_pose

    print(pose)

    pose = torch.Tensor(pose).to(device)

    #pose = torch.Tensor(render_poses[15]).to(device) #Take 2nd pose as guess
    #pose = pose[:3, :4]
    #pose = torch.row_stack((pose, torch.tensor([0, 0, 0, 1])))
    
    #img_gt = torch.Tensor(img_gt[::render_factor, ::render_factor, :]).to(device)     #Take 2nd image as GT

    cam_transf = camera_transf().to(device)
    optimizer = torch.optim.Adam(params=cam_transf.parameters(), lr=lrate, betas=(0.9, 0.999))

    # calculate angles and translation of the observed image's pose
    phi_ref = np.arctan2(obs_img_pose[1,0], obs_img_pose[0,0])*180/np.pi
    theta_ref = np.arctan2(-obs_img_pose[2, 0], np.sqrt(obs_img_pose[2, 1]**2 + obs_img_pose[2, 2]**2))*180/np.pi
    psi_ref = np.arctan2(obs_img_pose[2, 1], obs_img_pose[2, 2])*180/np.pi
    translation_ref = np.sqrt(obs_img_pose[0,3]**2 + obs_img_pose[1,3]**2 + obs_img_pose[2,3]**2)

    #Make an array of interest_points and round to nearest pixel
    interest_pts = [(round(point.pt[1]), round(point.pt[0])) for point in kp1]
    #Get rid of non-distinct pixels
    interest_pts = list(set(interest_pts))

    #Turn list of lists into array
    interest_pts = np.array([list(points) for points in interest_pts])

    print(interest_pts)

    interest_regions = np.zeros((W, H, ), dtype=np.uint8)
    interest_regions[interest_pts[:,1], interest_pts[:,0]] = 1
    interest_regions = cv.dilate(interest_regions, np.ones((kernel_size, kernel_size), np.uint8), iterations=I)
    interest_regions = np.array(interest_regions, dtype=bool)
    print(coords.shape, interest_regions.shape)
    interest_regions = coords[interest_regions]

    # not_POI contains all points except of POI
    coords = np.reshape(coords, [-1,2])  # (H * W, 2)
    not_POI = set(tuple(point) for point in coords) - set(tuple(point) for point in interest_pts)
    not_POI = np.array([list(point) for point in not_POI]).astype(int)

    for i in range(500):
        #print(i, time.time() - t)
        #t = time.time()

        pose_iter0 = cam_transf(pose)

        rays_o, rays_d = get_rays(H, W, K,pose_iter0)  # (H, W, 3), (H, W, 3)

        if sampling_strategy == 'random':
            rand_inds = np.random.choice(coords.shape[0], size=batch_size, replace=False)
            batch = coords[rand_inds]

        elif sampling_strategy == 'interest_regions':
            rand_inds = np.random.choice(interest_regions.shape[0], size=batch_size, replace=False)
            batch = interest_regions[rand_inds]

        else:
            print('Unknown sampling strategy')
            return

        rays_o = rays_o[batch[:, 0], batch[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[batch[:, 0], batch[:, 1]]
        batch_rays = torch.stack([rays_o, rays_d], 0)

        target_s = img_gt[batch[:, 0], batch[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, rays=batch_rays, **render_kwargs)

        #rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=pose_iter0[:3, :4], **render_kwargs)

        #print('RGB size', rgb.shape)

        #print('img shape', img_gt.shape)

        optimizer.zero_grad()

        output = mseloss(rgb, target_s)

        output.backward()
        optimizer.step()

        new_lrate = lrate * (0.8 ** ((i + 1) / 100))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if (i + 1) % 20 == 0 or i == 0:
            print('Step: ', i)
            print('Loss: ', output)

            with torch.no_grad():
                pose_dummy = pose_iter0.cpu().detach().numpy()
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

                #rgb0, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=pose_iter0[:3,:4], **render_kwargs)
                #plt.imshow(rgb0.detach().cpu().numpy()), plt.show()
