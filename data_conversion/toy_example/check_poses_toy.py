import cv2
import os
import numpy as np
from glob import glob
import open3d as o3d
import imageio

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

############### loading poses
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def find_res(img):
    height, width, channels = imageio.imread(img).shape
    return [width,height]

def find_poses(instance_dir):
    image_dir = '{0}/image'.format(instance_dir)
    image_paths = sorted(glob_imgs(image_dir))
    n_images = len(image_paths)
    resolution = find_res(image_paths[0])

    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    i = 0
    intrinsics_all = []
    pose_all = []
    camera_dict = {}
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        camera_dict[i] = {}
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        pose_all.append(np.array(pose))
        intrinsics_all.append(np.array(intrinsics))
        camera_dict[i]["K"] = intrinsics
        camera_dict[i]["W2C"] = pose
        camera_dict[i]["img_size"] = resolution
        i+= 1

    return pose_all, intrinsics_all, camera_dict

################ visuliztion
points = {'[0, 1, 0]': [] , '[0, 0, 1]': []}
def get_camera_frustum(img_size, K, W2C, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    points[str(color)].append(K)
    hfov = np.rad2deg(np.arctan(W / 2. / K[0, 0]) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / K[1, 1]) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))
    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]
    # points[str(color)].append(frustum_points[0])

    return frustum_points, frustum_lines, frustum_colors


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def visualize_cameras(colored_camera_dicts, sphere_radius, camera_size=0.1, geometry_file=None, geometry_type='mesh'):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0., 0., 0.])
    things_to_draw = [sphere, coord_frame]

    idx = 0
    for color, camera_dict in colored_camera_dicts:
        idx += 1

        cnt = 0
        frustums = []
        for img_name in sorted(camera_dict.keys()):
            K = np.array(camera_dict[img_name]['K']).reshape((4, 4))
            W2C = np.array(camera_dict[img_name]['W2C']).reshape((4, 4))
            C2W = np.linalg.inv(W2C)
            points[str(color)].append(C2W)

            img_size = camera_dict[img_name]['img_size']
            frustums.append(get_camera_frustum(img_size, K, C2W, frustum_length=camera_size, color=color))
            cnt += 1
        cameras = frustums2lineset(frustums)
        things_to_draw.append(cameras)

    if geometry_file is not None:
        if geometry_type == 'mesh':
            geometry = o3d.io.read_triangle_mesh(geometry_file)
            geometry.compute_vertex_normals()
        elif geometry_type == 'pointcloud':
            geometry = o3d.io.read_point_cloud(geometry_file)
        else:
            raise Exception('Unknown geometry_type: ', geometry_type)

        things_to_draw.append(geometry)

    o3d.visualization.draw_geometries(things_to_draw)


instance_dir1 = 'toy_example/author_data/scan1'
instance_dir2 = 'toy_example/volsdf_data/scan1'
geometry_file = 'toy_example/surface_100.ply'
geometry_type = 'mesh'


pose_auth, intr_auth, cam_aut = find_poses(instance_dir1)
pose_col, intr_col, cam_col = find_poses(instance_dir2)

# for i in range(len(intr_auth)):
#     print(intr_col[i])

sphere_radius = 3.
# train_cam_dict = json.load(open(os.path.join(base_dir, 'train/cam_dict_norm.json')))
# test_cam_dict = json.load(open(os.path.join(base_dir, 'test/cam_dict_norm.json')))
# path_cam_dict = json.load(open(os.path.join(base_dir, 'camera_path/cam_dict_norm.json')))
train_cam_dict = cam_aut
test_cam_dict = cam_col
camera_size = 0.15
colored_camera_dicts = [([0, 1, 0], train_cam_dict),
                        ([0, 0, 1], test_cam_dict) ]

visualize_cameras(colored_camera_dicts, sphere_radius,
                  camera_size=camera_size, geometry_file=geometry_file, geometry_type=geometry_type)

# trans = icp(points['[0, 1, 0]'], points['[0, 0, 1]'])
# print(trans)
# print(points['[0, 1, 0]'][2])
# print("  ")
# print(points['[0, 0, 1]'][3])
# print(points)

