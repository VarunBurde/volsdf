# usage python colmap2volsdf_toy.py -i toy_example/colmap_data/scan1 -o toy_example/volsdf_data/scan1

import os
import collections
import numpy as np
import argparse
import cv2
import math
import open3d as o3d

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

    parser.add_argument("-i", "--colmap_data_dir", required = True, help="input directory ")
    parser.add_argument("-o", "--destination_data_dir", required = True ,  help='output directory')

    args = parser.parse_args()
    return args

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_cameras_text(path):

    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_images_text(path):

    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def get_center_point(num_cams,cameras):
    A = np.zeros((3 * num_cams, 3 + num_cams))
    b = np.zeros((3 * num_cams, 1))
    camera_centers=np.zeros((3,num_cams))
    for i in range(num_cams):
        P0 = cameras['world_mat_%d' % i][:3, :]

        K = cv2.decomposeProjectionMatrix(P0)[0]
        R = cv2.decomposeProjectionMatrix(P0)[1]
        c = cv2.decomposeProjectionMatrix(P0)[2]
        c = c / c[3]
        camera_centers[:,i]=c[:3].flatten()

        v = np.linalg.inv(K) @ np.array([800, 600, 1])
        v = v / np.linalg.norm(v)

        v=R[2,:]
        A[3 * i:(3 * i + 3), :3] = np.eye(3)
        A[3 * i:(3 * i + 3), 3 + i] = -v
        b[3 * i:(3 * i + 3)] = c[:3]

    soll= np.linalg.pinv(A) @ b

    return soll,camera_centers

def normalize_cameras(original_cameras_filename,output_cameras_filename,num_of_cameras):
    cameras = np.load(original_cameras_filename)
    if num_of_cameras==-1:
        all_files=cameras.files
        maximal_ind=0
        for field in all_files:
            maximal_ind=np.maximum(maximal_ind,int(field.split('_')[-1]))
        num_of_cameras=maximal_ind+1
    soll, camera_centers = get_center_point(num_of_cameras, cameras)

    center = soll[:3].flatten()

    max_radius = np.linalg.norm((center[:, np.newaxis] - camera_centers), axis=0).max() * 1.1

    normalization = np.eye(4).astype(np.float32)

    normalization[0, 3] = center[0]
    normalization[1, 3] = center[1]
    normalization[2, 3] = center[2]

    normalization[0, 0] = max_radius / 3.0
    normalization[1, 1] = max_radius / 3.0
    normalization[2, 2] = max_radius / 3.0

    cameras_new = {}
    for i in range(num_of_cameras):
        cameras_new['scale_mat_%d' % i] = normalization
        cameras_new['world_mat_%d' % i] = cameras['world_mat_%d' % i].copy()
    np.savez(output_cameras_filename, **cameras_new)

# ---------------------------------------------------------------------------------------#
def rotx(theta, unit="rad"):
    if unit == "deg":
        theta = theta * math.pi / 180
    ct = math.cos(theta)
    st = math.sin(theta)
    mat = np.matrix([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    mat = np.asmatrix(mat.round(15))
    return mat


# ---------------------------------------------------------------------------------------#
def roty(theta, unit="rad"):
    if unit == "deg":
        theta = theta * math.pi / 180
    ct = math.cos(theta)
    st = math.sin(theta)
    mat = np.matrix([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    mat = np.asmatrix(mat.round(15))
    return mat


# ---------------------------------------------------------------------------------------#
def rotz(theta, unit="rad"):
    if unit == "deg":
        theta = theta * math.pi / 180
    ct = math.cos(theta)
    st = math.sin(theta)
    mat = np.matrix([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    mat = np.asmatrix(mat.round(15))
    return mat


flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

if __name__ == '__main__':
    args = parse_args()
    Col_dir = args.colmap_data_dir
    Out_dir = args.destination_data_dir

    camera_file = os.path.join(Col_dir, 'cameras.txt')
    image_file = os.path.join(Col_dir, 'images.txt')
    img_dir = os.path.join(Out_dir, 'image')
    cameras = read_cameras_text(camera_file)
    images = read_images_text(image_file)


    K = np.eye(3)
    K[0, 0] = cameras[1].params[0]
    K[1, 1] = cameras[1].params[3]
    K[0, 2] = cameras[1].params[1]
    K[1, 2] = cameras[1].params[2]

    print(K)

    if not os.path.exists(Out_dir):
        os.mkdir(Out_dir)

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    cameras_npz_format = {}
    for ii in range(len(images)):
        cur_image = images[ii +1]
        up = np.zeros(3)
        M = np.zeros((3, 4))
        M[:, 3] = cur_image.tvec
        M[:3, :3] = qvec2rotmat(cur_image.qvec)

        ####### add extra rotation here
        # rot = rotx(65, unit= 'deg') @ roty(65, unit='deg')
        # M[:3, :3] =  M[:3, :3] @ rot

        ########## Projection matrix
        P = np.eye(4)
        P[:3, :] = K @ M

        ###### experiment flipping

        # P = P[[1, 0, 2, 3], :]  # swap y and z
        # P[0:3, 0] *= -1
        # P[0:3, 3] *= -1  # flip the y and z axis
        # P[0:3, 2] *= -1


        # up += P[0:3, 1]
        #
        # up = up / np.linalg.norm(up)
        # print("up vector was", up)
        # R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
        # R = np.pad(R, [0, 1])
        # R[-1, -1] = 1
        #
        # P = R @ P

        cameras_npz_format['world_mat_%d' % ii] = P

        img = cv2.imread(os.path.join(Col_dir, 'image', cur_image.name))


        img_path = os.path.join(img_dir,'{0:08}.jpg'.format(ii))
        cv2.imwrite(img_path, img)
        print("saving", '{0:08}.jpg'.format(ii))


    np.savez(os.path.join( Out_dir, "cameras_before_normalization.npz"), **cameras_npz_format)

    ############ Normalize camera
    print("Normalizing camera")

    camera_path_before_normalization = os.path.join( Out_dir, "cameras_before_normalization.npz")
    output_camera = os.path.join( Out_dir, "cameras.npz")
    normalize_cameras(camera_path_before_normalization, output_camera, -1)