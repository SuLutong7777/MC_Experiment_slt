import json
import torch
import logging
import os
import struct
import math
import apriltag
import cv2
import random
import sqlite3
import numpy as np

from PIL import Image
from pathlib import Path
from torchvision import transforms as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Real_Data(object):
    def __init__(self, train_json, test_json, val_json):
        self.train_json = train_json
        self.test_json = test_json
        self.val_json = val_json

        train_poses_tensor = self.load_poses_from_json(self.train_json)
        # 将训练集位姿中心化
        self.center_poses, colmap2world_transform = self.transform_poses_pca(train_poses_tensor)
        # 将中心化的位姿保存回 JSON 文件
        self.update_json_with_centered_poses(self.train_json, self.center_poses)
        # 生成渲染位姿
        render_poses = self.generate_ellipse_path(self.center_poses, n_frames=200)
        render_poses = torch.from_numpy(render_poses).to(torch.float32)
        # 将渲染位姿保存在test和val的json文件中
        self.save_dict_to_json(render_poses, test_json, mode='test')
        self.save_dict_to_json(render_poses, val_json, mode='val')
        # 画出中心化后的训练集位姿以及测试集位姿
        cam_intr_train = torch.full((train_poses_tensor.shape[0],), 0.0210)
        cam_intr_test = torch.full((200,), 0.0210)        
        self.show_camera_position(cam_intr_train, self.center_poses, pth='train_pca')
        self.show_camera_position(cam_intr_test, render_poses, pth='test_pca')    

    # 更新 JSON 文件中的 transform_matrix 为中心化后的位姿
    def update_json_with_centered_poses(self, json_path, center_poses):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 确保中心化后的位姿数量与原数据一致
        assert len(data["frames"]) == center_poses.shape[0], "中心化后的位姿数量与原数据不匹配！"

        # 更新每一帧的 transform_matrix
        for i, frame in enumerate(data["frames"]):
            # 将中心化后的位姿转换为列表并更新 JSON 中的 transform_matrix
            frame["transform_matrix"] = center_poses[i].cpu().numpy().tolist()

        # 将更新后的数据写回到 JSON 文件
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"成功更新 {len(data['frames'])} 帧的位姿到 JSON 文件：{json_path}")

    # 位姿中心化
    def transform_poses_pca(self, poses):
        # 获取所有相机的中心点
        trans = poses[:, :3, 3]
        # 取平均值
        trans_mean = torch.mean(trans, dim=0)
        # 中心化，相当于取所有点的平均中心为新坐标原点
        # 生成新的相机中心位置 [194, 3]
        trans = trans - trans_mean
        ############ 计算特征值eigval，和特征向量eigvec
        # 注意，这两个算出来是复数格式，有实部和虚部，即使虚部为0，也会保留
        # 所以这里要除去虚部(虚部全部算出来都是0)
        # trans.T @ trans: [3,3], 注意，这个过程在计算平移向量集合的协方差（正常有个除以n的系数，但是不影响特征向量）
        # eigval:[3], eigvec:[3,3]
        # eigval, eigvec = torch.linalg.eig(trans.T @ trans)
        # 转成Numpy做，pytorch版本的特征向量符号与Numpy不一致
        eigval, eigvec = np.linalg.eig(np.array(trans).T @ np.array(trans))
        eigval = torch.from_numpy(eigval)
        eigvec = torch.from_numpy(eigvec)
        # print(eigval, eigvec)
        # exit()
        # 对所有特征值进行从大到小的排序，获取排序的索引
        inds = torch.argsort(eigval.real, descending=True)
        # 同时排序特征向量
        # eigvec = eigvec[:, inds].real
        eigvec = eigvec[:, inds]
        # print(eigvec, "2222")
        # 将特征向量转置，构造投影矩阵，将所有坐标点投影到新的坐标系下
        # 这个新的坐标系的轴就是数据的主成分轴。
        # 这里eigvec为[3,3]，因为数据一共有三个主成分，分别为x,y,z，都需要保留，所以上面链接中的k值取3，就等同于不用筛选
        # eigvec中，每一列是特征向量，转置之后变成行，在进行投影的时候就是rot@trans，x,y,z维度能对应
        rot = eigvec.T
        # 保持坐标系变换后与原来规则相同
        # 在三维空间中，一个合法的旋转矩阵应该是正交的且行列式为1，这保证了坐标系变换保持了空间的右手规则。
        # 如果行列式小于0，表明旋转矩阵将导致坐标系翻转，违反了右手规则。
        # 一个矩阵的行列式（np.linalg.det(rot)）告诉我们这个矩阵是保持空间的定向（右手或左手）不变还是改变了空间的定向。具体来说：
        # 如果行列式大于0，说明变换后的坐标系保持原有的定向（即如果原坐标系是右手的，变换后仍然是右手的）。
        # 如果行列式小于0，说明变换后的坐标系改变了原有的定向（即从右手变为了左手，或从左手变为了右手）。
        if torch.linalg.det(rot) < 0:
            rot = torch.diag(torch.tensor([1.0, 1.0, -1.0])) @ rot

        # 构建完整的[R|T]变换矩阵，直接针对原始的pose信息，不再单纯考虑trans
        # 尺寸是[3, 4]
        transform_mat = torch.cat([rot, rot @ -trans_mean[:, None]], dim=-1)
        # 转为[4, 4]
        transform_mat = torch.cat([transform_mat, torch.tensor([[0, 0, 0, 1.]])], dim=0)
        # 整体RT矩阵转换[N, 4, 4]
        poses_recentered = transform_mat @ poses

        # 检查坐标轴方向
        # 检查在新坐标系中，相机指向的平均方向的y分量是否向下。如果是的话，这意味着变换后的位姿与常规的几何或物理约定（例如，通常期望的y轴向上）不符。
        if poses_recentered.mean(axis=0)[2, 1] < 0:
            poses_recentered = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0])) @ poses_recentered
            transform_mat = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0])) @ transform_mat
        # print(poses_recentered, "lll")
        # print(transform_mat, "eeee")

        # # 对数据进行归一化，收敛到[-1, 1]之间
        scale_factor = 1. / torch.max(torch.abs(poses_recentered[:, :3, 3]))
        print("scale_factor: ", scale_factor)
        poses_recentered[:, :3, 3] *= scale_factor
        poses_recentered[:, 3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(poses_recentered.shape[0], 1)
        transform_mat = torch.diag(torch.tensor([scale_factor] * 3 + [1])) @ transform_mat
        
        return poses_recentered, transform_mat

    # 将字典转换为json文件
    def save_dict_to_json(self, render_poses, output_json_path, mode='test'):
        # 构建要保存的JSON数据结构
        data = {"frames": []}
        num_frames = render_poses.shape[0]
        print('num_frames: ', num_frames)
        # 遍历 dict_data 中的每一个图像数据
        for i in range(render_poses.shape[0]):
            # 将相机角度和变换矩阵转换为适合JSON的格式
            camera_angle_x = float(0.9143)
            transform_matrix = render_poses[i].tolist()
            # 在 transform_matrix 中添加第四行 [0.0, 0.0, 0.0, 1.0]
            if len(transform_matrix) == 3:
                transform_matrix.append([0.0, 0.0, 0.0, 1.0])
            # 构建每一帧的数据
            frame_data = {
                "file_path": f"./{str(mode)}/{str(i+1).zfill(4)}",  # 使用四位数命名文件
                "camera_angle_x": camera_angle_x,  # 确保转换为 float 类型
                "transform_matrix": transform_matrix  # 将 Tensor 转换为列表并添加第四行
            }
            # 添加到 frames 列表中
            data["frames"].append(frame_data)
        # 将数据写入 json 文件
        with open(output_json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    # 读取json文件获取位姿
    def load_poses_from_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 假设数据存储为Nx4x4形式，读取出位姿
        poses = []
        for frame in data["frames"]:
            pose = np.array(frame["transform_matrix"])
            poses.append(torch.from_numpy(pose).float())
        
        return torch.stack(poses, 0)
    
    # 生成椭圆的轨迹路线，用来做前向测试和渲染
    def generate_ellipse_path(self, poses, n_frames=200, const_speed=True, z_variation=0., z_phase=0.):
        # 转为numpy,有个别函数pytorch中没有办法实现
        poses = np.array(poses)
        # 找到生成视角的中心点
        center = self.focus_point_fn(poses)
        # 计算路径中心的偏移量，将Z轴坐标设为0，这使得路径高度位于Z=0平面。
        offset = np.array([center[0], center[1], 0])
        # 计算相机位置相对于路径中心的偏移后的绝对值，并找到90%分位数的值，用于确定椭圆轴的缩放因子
        sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
        # 根据缩放因子sc和中心偏移offset，确定椭圆在X和Y方向的边界值。
        low = -sc + offset
        high = sc + offset
        # 确定Z轴方向上的边界值，根据相机位置的10%和90%分位数来获取。
        z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
        z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

        # 定义一个内部函数，根据角度theta计算在椭圆路径上的位置。位置计算包括X和Y轴上的椭圆插值，以及可选的Z轴上的高度变化。
        def get_positions(theta):
            return np.stack([
                low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
                low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
                z_variation * (z_low[2] + (z_high - z_low)[2] *
                            (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
            ], -1)
        # 生成一个从0到2*pi的线性分布的角度数组，用于椭圆路径的生成。
        theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
        # 调用get_positions函数，根据theta值计算椭圆路径上的位置。
        positions = get_positions(theta)
        # 如果要求相机运动的速度为常数，这涉及到计算相邻位置间的距离，
        # 并基于这些距离重新计算theta值，使得相机在路径上移动的速度尽可能保持一致。
        if const_speed:
            lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
            theta = self.sample_np(None, theta, np.log(lengths), n_frames + 1)
            positions = get_positions(theta)

        # 去掉重复的最后一个位置，因为起点和终点是相同的。
        positions = positions[:-1]

        # Set path's up vector to axis closest to average of input pose up vectors.
        avg_up = poses[:, :3, 1].mean(0)
        avg_up = avg_up / np.linalg.norm(avg_up)
        ind_up = np.argmax(np.abs(avg_up))
        up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

        def normalize(x):
            """Normalization helper function."""
            return x / np.linalg.norm(x)

        def viewmatrix(lookdir, up, position):
            """Construct lookat view matrix."""
            vec2 = normalize(lookdir)
            vec0 = normalize(np.cross(up, vec2))
            vec1 = normalize(np.cross(vec2, vec0))
            m = np.stack([vec0, vec1, vec2, position], axis=1)
            return m

        return np.stack([viewmatrix(p - center, up, p) for p in positions])

    # 计算所有相机位姿中，所有焦点轴（即每个相机位姿的朝向）最接近相交的点。
    def focus_point_fn(self, poses):
        # R和T矩阵
        directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
        m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
        mt_m = np.transpose(m, [0, 2, 1]) @ m
        focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
        return focus_pt

    def sample_np(self, rand,
                t,
                w_logits,
                num_samples,
                single_jitter=False,
                deterministic_center=False):
        """
        numpy version of sample()
    """
        eps = np.finfo(np.float32).eps

        # Draw uniform samples.
        if not rand:
            if deterministic_center:
                pad = 1 / (2 * num_samples)
                u = np.linspace(pad, 1. - pad - eps, num_samples)
            else:
                u = np.linspace(0, 1. - eps, num_samples)
            u = np.broadcast_to(u, t.shape[:-1] + (num_samples,))
        else:
            # `u` is in [0, 1) --- it can be zero, but it can never be 1.
            u_max = eps + (1 - eps) / num_samples
            max_jitter = (1 - u_max) / (num_samples - 1) - eps
            d = 1 if single_jitter else num_samples
            u = np.linspace(0, 1 - u_max, num_samples) + \
                np.random.rand(*t.shape[:-1], d) * max_jitter

        return self.invert_cdf_np(u, t, w_logits)

    def invert_cdf_np(self, u, t, w_logits):
        """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
        # Compute the PDF and CDF for each weight vector.
        w = np.exp(w_logits) / np.exp(w_logits).sum(axis=-1, keepdims=True)
        cw = np.minimum(1, np.cumsum(w[..., :-1], axis=-1))
        shape = cw.shape[:-1] + (1,)
        # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
        cw0 = np.concatenate([np.zeros(shape), cw,
                            np.ones(shape)], axis=-1)
        # Interpolate into the inverse CDF.
        interp_fn = np.interp
        t_new = interp_fn(u, cw0, t)
        return t_new

    def show_camera_position(self, intr_mat, RT_mat, cam_size=0.05, pth='pose'):
        # 初始化绘图底板
        self.init_show_figure(show_info=True)
        # 判断一下路径是否存在
        save_path = Path("/home/sulutong/mr2nerf-master/colmap2nerf")
        # os.makedirs(save_path, exist_ok=True)
        plt.cla()
        # color_gt = (0.7,0.2,0.7)
        color_pd = (0,0.6,0.7)
        self.ax.set_xlim(-1.0, 1.0)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_zlim(-1.0, 1.0)
        # self.ax.set_xlim(-0.5, 0.5)
        # self.ax.set_ylim(-0.5, 0.5)
        # self.ax.set_zlim(-0.5, 0.5)
        # [84, 3, 4]
        clip_pose = RT_mat[:, :3, :]
        self.draw_camera_shape(clip_pose, intr_mat, color_pd, cam_size)
        file_path = os.path.join(Path(save_path), Path(pth + '.png'))
        plt.savefig(file_path)
        # plt.show()
        # plt.pause(0)

    def init_show_figure(self, show_info=True):
        self.all_fig = plt.figure(figsize=(10,10))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.default'] = 'regular'
        self.ax = Axes3D(self.all_fig, auto_add_to_figure=False)
        self.all_fig.add_axes(self.ax)
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            
        if show_info:
            self.ax.set_xlabel("X Axis")
            self.ax.set_ylabel("Y Axis")
            self.ax.set_zlabel("Z Axis")
            # self.ax.set_xlim(-3.5, 3.5)
            # self.ax.set_ylim(-3.5, 3.5)
            # self.ax.set_zlim(-1.5, 3.5)
        else:
            self.ax.grid(False)
            self.ax.axis(False)

        plt.ion()
        plt.gca().set_box_aspect((1, 1, 1))

    def draw_camera_shape(self, extr_mat, intr_mat, color, cam_size=0.25):
        # extr_mat: [84, 3, 4]
        # intr_mat: [84, 3, 3]
        cam_line = cam_size
        # focal = intr_mat[:,0,0]*cam_line/self.train_w
        focal = intr_mat
        cam_pts_1 = torch.stack([-torch.ones_like(focal)*cam_line/2,
                                 -torch.ones_like(focal)*cam_line/2,
                                 -focal], -1)[:,None,:].to(extr_mat.device)
        cam_pts_2 = torch.stack([-torch.ones_like(focal)*cam_line/2,
                                  torch.ones_like(focal)*cam_line/2,
                                 -focal], -1)[:,None,:].to(extr_mat.device)
        cam_pts_3 = torch.stack([ torch.ones_like(focal)*cam_line/2,
                                  torch.ones_like(focal)*cam_line/2,
                                  -focal], -1)[:,None,:].to(extr_mat.device)
        cam_pts_4 = torch.stack([ torch.ones_like(focal)*cam_line/2,
                                 -torch.ones_like(focal)*cam_line/2,
                                 -focal], -1)[:,None,:].to(extr_mat.device)
        cam_pts_1 = cam_pts_1 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
        cam_pts_2 = cam_pts_2 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
        cam_pts_3 = cam_pts_3 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
        cam_pts_4 = cam_pts_4 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
        cam_pts = torch.cat([cam_pts_1, cam_pts_2, cam_pts_3, cam_pts_4, cam_pts_1], dim=-2)
        for i in range(4):
            # [84, 2, 3]
            cur_line_pts = torch.stack([cam_pts[:,i,:], cam_pts[:,i+1,:]], dim=-2).to('cpu')
            for each_cam in cur_line_pts:
                self.ax.plot(each_cam[:,0],each_cam[:,1],each_cam[:,2],color=color,linewidth=0.5)
        extr_T = extr_mat[:, :3, 3]
        for i in range(4):
            # [84, 2, 3]
            cur_line_pts = torch.stack([extr_T, cam_pts[:,i,:]], dim=-2).to('cpu')
            for each_cam in cur_line_pts:
                self.ax.plot(each_cam[:,0],each_cam[:,1],each_cam[:,2],color=color,linewidth=0.5)
        extr_T = extr_T.to('cpu')

        self.ax.scatter(extr_T[:,0],extr_T[:,1],extr_T[:,2],color=color,s=5)
