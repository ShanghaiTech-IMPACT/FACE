from skimage.measure import label, regionprops  
import json  
from tqdm import tqdm 
import pandas as pd 
import numpy as np  
import os 
import torch  
from glob import glob 
import torchio as tio 
from sympy import symbols, solve, Symbol  
from pymeshlab import Mesh, MeshSet 
import trimesh 
import argparse  
from skimage.measure import label, regionprops, marching_cubes

def voxel2mesh(tio_img, mesh_path,smooth_iters=5):

    img_data = tio_img.data.squeeze().numpy()  # Remove any extra dimensions and convert to numpy
    verts, faces, normals, values = marching_cubes(img_data, 0.5, spacing=tio_img.spacing)
    mesh = Mesh(verts, faces)
    ms = MeshSet()
    ms.add_mesh(mesh)
    ms.invert_faces_orientation()
    ms.laplacian_smooth(stepsmoothnum=smooth_iters)


    ms.save_current_mesh(mesh_path)

# 保存3D网格数据为文件
def save_mesh(verts, faces, path, invert_faces=False):
    """
    保存3D网格数据为.off文件
    输入:
        verts: 网格的顶点坐标
        faces: 网格的面（由顶点组成的面）
        path: 输出文件路径
        invert_faces: 如果为True,则反转面的朝向
    输出:
        无
    """
    mesh = Mesh(verts, faces) 
    ms = MeshSet()  
    ms.add_mesh(mesh) 
    if invert_faces:
        ms.invert_faces_orientation() 
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ms.save_current_mesh(path) 



# 解线性方程,求平面方程系数
def solve_linear_equation(ps, normal=None, vertical_plane=None):
    """
    根据给定的点集和条件求解平面方程系数。
    输入:
        ps: 点集,可以是3个点,1个点和法向量,或者2个点和垂直平面
        normal: 法向量,仅在1个点时使用
        vertical_plane: 垂直平面,仅在2个点时使用
    输出:
        返回平面方程系数 A, B, C, D
    """
    # 3个点确定一个平面
    if len(ps) == 3:
        a, b, c = symbols('a, b, c')
        res = solve([
            ps[0][0] * a + ps[0][1] * b + ps[0][2] * c + 1,
            ps[1][0] * a + ps[1][1] * b + ps[1][2] * c + 1,
            ps[2][0] * a + ps[2][1] * b + ps[2][2] * c + 1,
        ], (a, b, c))
        A, B, C, D = float(res[a]), float(res[b]), float(res[c]), 1

    # 1个点与法向量确定平面
    elif normal is not None:
        A, B, C = normal
        d = Symbol('d')
        res = solve([A * ps[0][0] + B * ps[0][1] + C * ps[0][2] + d], [d])
        D = float(res[d])

    # 2个点与垂直平面确定平面
    elif vertical_plane is not None:
        a, b, c = symbols('a, b, c')
        res = solve([
            vertical_plane[0] * a + vertical_plane[1] * b + vertical_plane[2] * c,
            a * ps[0][0] + b * ps[0][1] + c * ps[0][2] + 1,
            a * ps[1][0] + b * ps[1][1] + c * ps[1][2] + 1,
        ], (a, b, c))
        A, B, C, D = float(res[a]), float(res[b]), float(res[c]), 1

    return A, B, C, D

# 查询指定平面与空间的关系
def query_side(plane_params, space):
    """
    判断空间中的每个点是否在平面的某一侧
    输入:
        plane_params: 平面参数 (A, B, C, D)
        space: 空间中的每个点坐标
    输出:
        query_res: 平面一侧的点为正值，另一侧为负值
    """
    p = np.expand_dims(np.array(plane_params[:3]), axis=(1, 2, 3))
    query_res = np.sum(space * p, axis=0) + plane_params[-1]
    return query_res


# 计算 ROI 平面
def cal_roi_plane(points, region='R'):
    """
    计算ROI的平面方程参数
    输入:
        points: 存储关键点的字典
        region: 'R' 或 'L'，表示右侧或左侧
    输出:
        返回5个平面方程参数
    """
    # 计算HP平面
    A, B, C, D = solve_linear_equation([points['Or L'], points['Or R'], points['Po R']])
    A, B, C, D = solve_linear_equation([points['C-{}'.format(region)]], (A, B, C))

    A1, B1, C1, D1 = solve_linear_equation([points['Or L'], points['Or R'], points['Po R']])

    # 计算SP平面
    A2, B2, C2, D2 = solve_linear_equation([points['N'], points['S']], vertical_plane=(A1, B1, C1))

    # 计算ROI的三个平面
    A3, B3, C3, D3 = solve_linear_equation([points['Cornoid-{}'.format(region)], points['C-{}'.format(region)]], vertical_plane=(A2, B2, C2))
    A4, B4, C4, D4 = solve_linear_equation([points['Condylar-posterior-{}'.format(region)], points['Go-{}'.format(region)]], vertical_plane=(A2, B2, C2))
    A5, B5, C5, D5 = solve_linear_equation([points['Cornoid-{}'.format(region)], points['Go-{}'.format(region)]], vertical_plane=(A2, B2, C2))
    
    return (A, B, C, D), (A2, B2, C2, D2), (A3, B3, C3, D3), (A4, B4, C4, D4), (A5, B5, C5, D5)


# 计算交线
def cal_cross_line(line_planes, cross_planes, rx=10):
    """
    计算两条平面的交线
    输入:
        line_planes: 两个平面方程
        cross_planes: 与平面交叉的平面方程
        rx: 用于计算交点的X轴坐标
    输出:
        返回交线的两个点坐标
    """
    plane1, plane2 = line_planes 
    A1, B1, C1, D1 = plane1
    A2, B2, C2, D2 = plane2
    A3, B3, C3, D3 = cross_planes

    # 求解交线上的一个点
    sy, sz = symbols('sy, sz')
    res = solve([
        A1 * rx + B1 * sy + C1 * sz + D1,
        A2 * rx + B2 * sy + C2 * sz + D2,
    ], (sy, sz))
    ry, rz = float(res[sy]), float(res[sz])

    # 求解交线的交点
    a, b, c = symbols('a, b, c')
    res = solve([
        A1 * a + B1 * b + C1*c + D1,
        A2 * a + B2 * b + C2*c + D2,
        A3 * a + B3 * b + C3*c + D3,
    ], (a, b, c))

    x, y, z = float(res[a]), float(res[b]), float(res[c])
    return (x, y, z), (rx, ry, rz)



# 正常化法线
def normal_correct(planes):
    """
    确保法线朝向正确
    输入:
        planes: 平面方程
    输出:
        返回法线正确的平面参数
    """
    p, p1, p2, p3, p4 = planes
    A1, B1, C1, D1 = p1
    A2, B2, C2, D2 = p2
    A3, B3, C3, D3 = p3
    A4, B4, C4, D4 = p4

    A, B, C, D = p 
    if np.dot(np.array([A, B, C]), np.array([0, 0, 1])) < 0:
        A, B, C, D = -A, -B, -C, -D
    
    c1, l1 = cal_cross_line(((A1, B1, C1, D1), (A2, B2, C2, D2)), (A3, B3, C3, D3))
    c2, l2 = cal_cross_line(((A1, B1, C1, D1), (A3, B3, C3, D3)), (A4, B4, C4, D4))
    c3, l3 = cal_cross_line(((A1, B1, C1, D1), (A4, B4, C4, D4)), (A2, B2, C2, D2))
    
    v1 = np.array(c2) - np.array([l1])
    v2 = np.array(c3) - np.array([l2])
    v3 = np.array(c1) - np.array([l3])
    
    D2, D3, D4 = 1, 1, 1
    if np.dot(v1, np.array([A2, B2, C2])) > 0:
        A2 = -A2
        B2 = -B2
        C2 = -C2
        D2 = -1  

    if np.dot(v2, np.array([A3, B3, C3])) > 0:
        A3 = -A3
        B3 = -B3
        C3 = -C3
        D3 = -1  

    if np.dot(v3, np.array([A4, B4, C4])) > 0:
        A4 = -A4
        B4 = -B4
        C4 = -C4
        D4 = -1
    
    return (A, B, C, D), (A2, B2, C2, D2), (A3, B3, C3, D3), (A4, B4, C4, D4)


def segment_roi_ramus_from_voxels(lab, planes, points, nii_path, mesh_path, region='R'):
    lab_img = lab.data.squeeze().numpy()
    p, p1, p2, p3 = normal_correct(planes)

    x = np.arange(lab_img.shape[0])
    y = np.arange(lab_img.shape[1])
    z = np.arange(lab_img.shape[2])
    xyz = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=0)
    
    # ln = 1 if np.dot(np.array(p2[:-1]), np.array([0, 0, 1])) > 0 else -1

    c_res = np.where(query_side(p, xyz) < 0, 1, 0)
    uu_res = np.where(query_side(p1, xyz) >= 0, 1, 0)
    u_res = np.where(query_side(p1, xyz) < 0, 1, 0)
    b_res = np.where(query_side(p3, xyz) < 0, 1, 0)
        
    # labels = np.where(u_res * l_res * b_res * lab_img > 0, 1, 0)
    labels1 = np.where(u_res * b_res * lab_img > 0, 1, 0)
    labels2 = np.where(uu_res * c_res * lab_img > 0, 1, 0)
    labels = labels1 + labels2 
    # filter
    label_con = label(labels, connectivity=1)
    label_reg = regionprops(label_con)
    label_area_idxes = []
    label_area_idxes = [prop.label for prop in label_reg if prop.area > 100]
    label_area_centroids = [prop.centroid for prop in label_reg if prop.area > 100]

    centroids = np.stack(label_area_centroids, axis=0)
    target_centroid = np.mean(np.stack([points['Cornoid-{}'.format(region)], points['Condylar-posterior-{}'.format(region)], points['Go-{}'.format(region)]], axis=0), axis=0)[None, ...]
    centroids = np.sum(np.abs(centroids - target_centroid), axis=1)
    idx = np.argsort(centroids)[0]
    target = label_area_idxes[idx]
    label_out = np.where(label_con==target, 1, 0)
    
    if nii_path is not None:
        seg = tio.LabelMap(tensor=torch.from_numpy(label_out).unsqueeze(0).float(), affine=lab.affine)
        seg.save(nii_path)

    if mesh_path is not None:
        voxel2mesh(seg, mesh_path,smooth_iters=5)




def segment_roi_condyle_from_voxels(lab, plane_param, cpoint, nii_path,mesh_path):
    split_plane = (0, 1, 0, -cpoint[1])
    lab_img = lab.data.squeeze().numpy()

    x = np.arange(lab_img.shape[0])
    y = np.arange(lab_img.shape[1])
    z = np.arange(lab_img.shape[2])
    xyz = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=0)
    
    res1 = np.where(query_side(plane_param, xyz) > 0, 1, 0)
    res2 = np.where(query_side(split_plane, xyz) > 0, 1, 0)
    
    labels = np.where(res1 * res2 * lab_img > 0, 1, 0)
    
    # remove other parts
    label_lab = label(labels, connectivity=1)
    reg = regionprops(label_lab)
    reg = [item for item in reg if item.area > 50]
    reg = sorted(reg, key=lambda x: np.linalg.norm(np.array(x.centroid) - np.array(cpoint)))
    
    # centroid of reg is larger than y of cpoint 
    target_label = reg[0].label
    
    out = np.where(label_lab == target_label, 1, 0)
    
    if nii_path is not None:
        seg = tio.LabelMap(tensor=torch.from_numpy(out).unsqueeze(0).float(), affine=lab.affine)
        seg.save(nii_path)

    if mesh_path is not None:
        voxel2mesh(seg, mesh_path,smooth_iters=5)




def crop(bone_mask_path,points_path,out_nii_path,out_mesh_path,time_point,crop_region = 'ramus'):

    points = json.load(open(points_path))
    for t_c in ['R', 'L']:

        if crop_region == 'ramus':
            img = tio.ScalarImage(bone_mask_path)
            planes = cal_roi_plane(points, region=t_c)
            nii_path = os.path.join(out_nii_path, '{}_{}_{}.nii.gz'.format(time_point,t_c,crop_region))
            mesh_path = os.path.join(out_mesh_path, '{}_{}_{}.off'.format(time_point,t_c,crop_region))

            segment_roi_ramus_from_voxels(img , planes, points, nii_path, mesh_path, region=t_c)

        else:
            img = tio.ScalarImage(bone_mask_path)
            planes = cal_roi_plane(points, region=t_c)[0]
            if np.dot(np.array(planes[:3]), np.array([0, 0, 1])) < 0:
                planes = -np.array(planes)

            nii_path = os.path.join(out_nii_path, '{}_{}_{}.nii.gz'.format(time_point,t_c,crop_region))
            mesh_path = os.path.join(out_mesh_path, '{}_{}_{}.off'.format(time_point,t_c,crop_region))
            
            segment_roi_condyle_from_voxels(img, planes, points['C-{}'.format(t_c)], nii_path,mesh_path)


        
    
time_points = ['T1', 'T2']
dir_path = r'code\data'
IDs = os.listdir(dir_path)


for ID in IDs:
    for time_point in time_points:
        bone_mask_path = os.path.join(dir_path,ID,f'{time_point}_bone.nii.gz')
        points_path = os.path.join(dir_path,ID,f'{time_point}_keypoints.json')
        out_nii_path =  os.path.join(dir_path,ID,'crop_nii')
        out_mesh_path =  os.path.join(dir_path,ID,'crop_mesh')
        os.makedirs(out_nii_path, exist_ok=True)  
        os.makedirs(out_mesh_path, exist_ok=True)  

        try:
            crop(bone_mask_path,points_path,out_nii_path,out_mesh_path,time_point,crop_region = 'ramus')
            crop(bone_mask_path,points_path,out_nii_path,out_mesh_path,time_point,crop_region = 'condyle')
        except IndexError:
            print(f"IndexError in ID: {ID}. Skipping this file.")
            continue 

