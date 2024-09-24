import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2

import sys
sys.path.insert(0, "../model")
from model.camera_model import ortho2rotation, rotation2orth, make_rand_axis, R_axis_angle


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, args=None, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(
            os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r'
        ) as fp:
            metas[s] = json.load(fp)

    # 存储所有的视场角
    all_focals = []
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        # 存储视场角
        focals = []
        if s=='train' or testskip==0:
            skip = 1
        elif s == 'test':
            skip = 1
        else:
            skip = 1

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            img = imageio.imread(fname)
            imgs.append(img)
            poses.append(np.array(frame['transform_matrix']))
            # 读取视场角并转换为焦距
            H, W = img.shape[:2]
            camera_angle_x = float(frame['camera_angle_x'])
            focal = .5 * W / np.tan(.5 * camera_angle_x)
            focals.append(focal)

            
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        # 存储所有焦距
        all_focals.append(focals)

    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    i_train, _ ,_ = i_split
    # imgs_shape:  (692, 260, 390, 4)
    imgs = np.concatenate(all_imgs, 0)
    print("imgs_shape: ", imgs.shape)
    poses = np.concatenate(all_poses, 0)
    focals = np.concatenate(all_focals, 0)

    # 创建深度信息张量，假设深度存储在 imgs[:, :, :, 3] 中
    # depth_info_tensor shape:  torch.Size([692, 260, 390])
    depth_info = (imgs[:, :, :, 3] > 0).astype(np.float32)  # 深度为0的存储为0，不为0的存储为1
    depth_info_tensor = torch.from_numpy(depth_info)  # 转换为torch张量

    print("depth_info_tensor shape: ", depth_info_tensor.shape)  # 检查形状

    # H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta['camera_angle_x'])
    # focal = .5 * W / np.tan(.5 * camera_angle_x)

    noisy_focals = focals
    if args.initial_noise_size_intrinsic != 0.0:
        noisy_focal = focal * (1 + args.initial_noise_size_intrinsic)
        print("Starting with noise in intrinsic parameters")
    
    poses_update = poses.copy()
    
    if args.initial_noise_size_rotation != 0.0:

        angle_noise = (
            np.random.rand(poses.shape[0], 1) - 0.5
        ) * 2 * args.initial_noise_size_rotation * np.pi / 180
        rotation_axis = make_rand_axis(poses.shape[0])
        rotmat = R_axis_angle(rotation_axis, angle_noise)

        poses_update[i_train, :3, :3] = rotmat[i_train] @ \
            poses_update[i_train, :3, :3]

        print("Starting with noise in rotation matrices")
        
    if args.initial_noise_size_translation != 0.0:
        
        individual_noise = (
            np.random.rand(poses.shape[0], 3) - 0.5
        ) * 2 * args.initial_noise_size_translation
        
        assert (
            np.all(
                np.abs(individual_noise) < args.initial_noise_size_translation
            )
        )
        
        poses_update[i_train, :3, 3] = poses_update[i_train, :3, 3] + \
            individual_noise[i_train]
        print("Starting with noise in translation parameters")
        
    if args.run_without_colmap != "none":
        if args.run_without_colmap in ["both", "rot"]: 
            print("Starting without colmap initialization in rotation")
            base_array = np.array([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]
            ])[None]
            poses_update[i_train, :3, :3] = base_array

        if args.run_without_colmap in ["both", "trans"]:
            print("Starting without colmap initialization in translation")
            poses_update[i_train, :3, 3] = 0.

    intrinsic_gt = torch.eye(4)
    intrinsic_gt = intrinsic_gt[None, :, :].repeat(focals.shape[0], 1, 1)
    for i in range(focals.shape[0]):
        intrinsic_gt[i][0][0] = focals[i]
        intrinsic_gt[i][1][1] = focals[i]
        intrinsic_gt[i][0][2] = W/2
        intrinsic_gt[i][1][2] = H/2
    intrinsic_gt = intrinsic_gt.cuda().float()
    noisy_intrinsic = intrinsic_gt
    # intrinsic_gt = torch.tensor([
    #     [focal, 0, W/2, 0],
    #     [0, focal, H/2, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1]
    # ]).cuda().float()

    print(f"Original focal length : {focals}\n")
    print(f"Initial noisy focal length : {noisy_focals}\n")

    render_poses = torch.stack(
        [
            pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(
                -180,180,40+1
            )[:-1]
        ], 
        dim=0
    )
    
    extrinsic_gt = torch.from_numpy(poses).cuda().float()

    return imgs, poses_update, render_poses, H, W, noisy_focals, i_split, \
        (intrinsic_gt, extrinsic_gt), depth_info_tensor


