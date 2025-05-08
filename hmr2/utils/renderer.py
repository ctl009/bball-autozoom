import os
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import pyrender
import trimesh
import cv2
from yacs.config import CfgNode
from typing import List, Optional
import pdb

def compute_bounding_box(vertices):
    """
    Compute the axis-aligned bounding box (AABB) for a given mesh.

    Parameters:
        vertices (numpy.ndarray): Array of shape (N, 3) containing the mesh vertices.

    Returns:
        dict: Bounding box with min and max coordinates for x, y, and z.
    """
    min_coords = np.min(vertices, axis=0)  # Minimum x, y, z
    max_coords = np.max(vertices, axis=0)  # Maximum x, y, z

    return {
        "min": min_coords,
        "max": max_coords
    }


# Function to create an arrow mesh along the X-axis
def create_arrow(scale=1.0, material=None, shaft_end = None, pose = None):
    
    # Shaft of the arrow (cylinder)
    shaft = trimesh.creation.cylinder(radius=0.02 * scale, height= shaft_end* scale, sections=20)
    shaft.apply_translation([0,0,shaft_end/2])  # Shift along the Z-axis

    # Tip of the arrow (cone)
    tip = trimesh.creation.cone(radius=0.04 * scale, height=0.2 * scale, sections=20)
    tip.apply_translation([0,0,shaft_end])  # Place at the end of the shaft

    # Combine shaft and tip
    arrow = trimesh.util.concatenate([shaft, tip])
    return pyrender.Mesh.from_trimesh(arrow, smooth=False, material=material, poses=pose)

def othogonal_coordinate(scale=1.0):
    
    x_rot = trimesh.transformations.rotation_matrix(
                np.radians(90), [0, 1, 0])
    y_rot = trimesh.transformations.rotation_matrix(
                np.radians(-90), [1, 0, 0])
    # Add arrow representing the Z-axis
    material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[1.0, 0.0, 0.0, 1.0], # red
                metallicFactor=0.0,
                roughnessFactor=1.0)
    
    z_arrow = create_arrow(scale=1.0, material=material, shaft_end=0.5, pose=np.eye(4))
    
    # Add arrow representing the X-axis
    material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.0, 1.0, 0.0, 1.0], # green
                metallicFactor=0.0,
                roughnessFactor=1.0)    
    # x_arrow = create_arrow(scale=1.0, material=material, shaft_end=[0.5,0,0])
    x_arrow = create_arrow(scale=1.0, material=material, shaft_end=0.5, pose=x_rot)
    # Add arrow representing the Y-axis
    material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.0, 0.0, 1.0, 1.0], # blue
                metallicFactor=0.0,
                roughnessFactor=1.0)
    # y_arrow = create_arrow(scale=1.0, material=material, shaft_end=[0,0.5,0])
    y_arrow = create_arrow(scale=1.0, material=material, shaft_end=0.5, pose=y_rot)
    return x_arrow, y_arrow, z_arrow

def arrow_from_chest(vertices):
    body_front_vert = vertices[5295]
    body_back_vert = vertices[4828]
    direction = -body_front_vert + body_back_vert
    arrow_len = np.linalg.norm(direction)+0.5

    shaft_radius=0.02; tip_radius=0.04; tip_length=0.2
    # Create a cylinder for the shaft
    shaft_length = arrow_len - tip_length
    shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_length)
    shaft.apply_translation([0, 0, shaft_length / 2])  # Move the shaft up to align with arrow direction

    # Create a cone for the tip
    tip = trimesh.creation.cone(radius=tip_radius, height=tip_length)
    tip.apply_translation([0, 0, shaft_length + tip_length / 2])  # Position the tip at the arrow's end

    # Combine shaft and tip into one mesh
    arrow = trimesh.util.concatenate([shaft, tip])

    # Rotate the arrow to align with the desired direction
    z_axis = np.array([0, 0, 1])  # Default direction of the arrow
    rotation_matrix = trimesh.geometry.align_vectors(z_axis, direction, False)
    arrow.apply_transform(rotation_matrix)

    # Translate the arrow to the starting point
    arrow.apply_translation(body_back_vert)

    return arrow
    


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam

def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses

def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)

def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))

def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    

def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

class Renderer:

    def __init__(self, cfg: CfgNode, faces: np.array):
        """
        Wrapper around the pyrender renderer to render SMPL meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """
        self.cfg = cfg
        self.focal_length = cfg.EXTRA.FOCAL_LENGTH
        self.img_res = cfg.MODEL.IMAGE_SIZE

        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces

    def __call__(self,
                vertices: np.array,
                camera_translation: np.array,
                image: torch.Tensor,
                full_frame: bool = False,
                imgname: Optional[str] = None,
                side_view=False, top_view=False,
                rot_angle=90,
                mesh_base_color=(1.0, 1.0, 0.9),
                scene_bg_color=(0,0,0),
                return_rgba=False,
                root_orientation = None
                ) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
        """
        
        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.
        else:
            image = image.clone() * torch.tensor(self.cfg.MODEL.IMAGE_STD, device=image.device).reshape(3,1,1)
            image = image + torch.tensor(self.cfg.MODEL.IMAGE_MEAN, device=image.device).reshape(3,1,1)
            image = image.permute(1, 2, 0).cpu().numpy()

        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(*mesh_base_color, 1.0))

        camera_translation[0] *= -1.


        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        elif top_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [1, 0, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))

        minY, maxY = np.min(vertices[:,1]), np.max(vertices[:,1])
        middleY = (0.7*minY + 0.3*maxY)
        cutting_verts = np.where(np.abs(vertices[:,1] - middleY) < 0.001)[0]

        material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[1.0, 0.0, 1.0, 1.0], # red
                    metallicFactor=0.0,
                    roughnessFactor=1.0)

        arrow = arrow_from_chest(vertices)
        if side_view:
            rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [0, 1, 0])
            arrow.apply_transform(rot)
        elif top_view:
            rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), [1, 0, 0])
            arrow.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        arrow.apply_transform(rot)
        scene.add(pyrender.Mesh.from_trimesh(arrow, smooth=False, material=material))



        
       
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)
        scene.add(camera, pose=camera_pose)


        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)


        # 3 coordinate
        x_arrow, y_arrow, z_arrow = othogonal_coordinate()
        # object orientation
        material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.0, 1.0, 1.0, 1.0], # cyan
        metallicFactor=0.0,
        roughnessFactor=1.0)
        object_arrow = create_arrow(scale=1.0, material=material, shaft_end=0.5, pose = np.eye(4))
        rotation = np.eye(4)  # Identity matrix for 4x4 transform
        object_rot = root_orientation[0,0].detach().cpu().numpy()  # Insert your 3x3 rotation matrix

        rotation[:3, :3] = object_rot  # Insert your 3x3 rotation matrix

        rotY = trimesh.transformations.rotation_matrix(
                np.radians(0), [0, 1, 0])

        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            scene.add(z_arrow, pose = rot.T)
            scene.add(x_arrow, pose = rot.T)
            scene.add(y_arrow, pose = rot.T)
            rotCorrect = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle*2), [1, 0, 0])
            scene.add(object_arrow, pose = (rotY@rot.T@rotCorrect@rotation))
            # create_bounding_box(scene, vertices @ (rot.T @ rotCorrect)[:3,:3].T)

        elif top_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [-1, 0, 0])
            scene.add(z_arrow, pose = rot.T)
            scene.add(x_arrow, pose = rot.T)
            scene.add(y_arrow, pose = rot.T)
            scene.add(object_arrow, pose = (rotY@rot@rotation))

            # create_bounding_box(scene, vertices @ rot[:3,:3].T)
        else:
            scene.add(z_arrow, "z_arrow")
            scene.add(x_arrow, "x_arrow")
            scene.add(y_arrow, "y_arrow")
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle*2), [1, 0, 0])
            
            scene.add(object_arrow, pose=(rotY@rot@rotation))



        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        output_img1 = color[:, :, :3]
        output_img1 = output_img1.astype(np.float32)
        cv2.imwrite("tmp1.jpg", 255*output_img1[:, :, ::-1])

        
        # pdb.set_trace()
        renderer.delete()

        if return_rgba:
            return color

        valid_mask = (color[:, :, -1])[:, :, np.newaxis]
        if not side_view and not top_view:
            output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]

        output_img = output_img.astype(np.float32)

        return output_img

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9), 
                            rot_axis=[1,0,0], rot_angle=0,):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])

        # if side_view:
        #     rot = trimesh.transformations.rotation_matrix(
        #         np.radians(rot_angle), [0, 1, 0])
        #     mesh.apply_transform(rot)

        mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        
        rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
            self,
            vertices: np.array,
            cam_t = None,
            rot=None,
            rot_axis=[1,0,0],
            rot_angle=0,
            camera_z=3,
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
        ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        if cam_t is not None:
            camera_translation = cam_t.copy()
            # camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * self.focal_length/render_res[1]])

        mesh = self.vertices_to_trimesh(vertices, camera_translation, mesh_base_color, rot_axis, rot_angle)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def direction_multiple(
            self,
            vertices: List[np.array],
            cam_t: List[np.array],
            rot_axis=[1,0,0],
            rot_angle=0,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
        ):

        mesh_list = [pyrender.Mesh.from_trimesh(self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle)) for vvv,ttt in zip(vertices, cam_t)]
       
        facingxz_list = []
        for vvv,ttt in zip(vertices, cam_t):
            body_front_vert = vvv[5295]
            body_back_vert = vvv[4828]
            direction = -body_front_vert + body_back_vert
            direction[1] = 0.
        
            front_pt = body_front_vert + ttt
            back_pt = body_back_vert + ttt

            rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
            front_pt = np.hstack([front_pt, np.ones(1)])
            back_pt = np.hstack([back_pt, np.ones(1)])
            
            front_pt = front_pt @ rot.T
            back_pt = back_pt @ rot.T

            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            front_pt = front_pt @ rot.T
            back_pt = back_pt @ rot.T
            
            front_pt = front_pt / front_pt[-1]
            back_pt = back_pt / back_pt[-1]

            facingxz_list.append([front_pt[:3], back_pt[:3]])

        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length

        # test proj
        focal = focal_length.detach().cpu().numpy()
        cam_x = camera_center[0].detach().cpu().numpy()
        cam_y = camera_center[1].detach().cpu().numpy()

        intrin = np.array([[focal, 0., cam_x],[0., focal, cam_y],[0., 0, 1.]])

        # for json saving
        facing_direction_2d = []
        facing_direction_3d = []
        position_2d = []
        for pairs in facingxz_list:
            front_pt, back_pt = pairs
            front_pt_calibrated = front_pt / front_pt[2]
            back_pt_calibrated = back_pt / back_pt[2]
            # transform the left-hand coordinate to right-hand coordinate
            front_pt_calibrated[0] *= -1
            back_pt_calibrated[0] *= -1
            front_proj = (front_pt_calibrated @ intrin.T)
            back_proj = (back_pt_calibrated @ intrin.T)

            direction_2d = front_proj - back_proj
            direction_3d = front_pt - back_pt
            direction_3d[0] *= -1

            facing_direction_2d.append(direction_2d[:2].tolist())
            facing_direction_3d.append(direction_3d[:3].tolist())
        for i,mesh in enumerate(mesh_list):
            center = mesh.centroid
            center_pt_calibrated = center / center[2]
            center_pt_calibrated[0] *= -1
            center_proj = (center_pt_calibrated @ intrin.T)
            position_2d.append(center_proj[:2].tolist())


        return facing_direction_2d, facing_direction_3d, position_2d


    def render_rgba_multiple(
            self,
            vertices: List[np.array],
            cam_t: List[np.array],
            rot_axis=[1,0,0],
            rot_angle=0,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
        ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        mesh_list = [pyrender.Mesh.from_trimesh(self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle)) for vvv,ttt in zip(vertices, cam_t)]

        material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[1.0, 0.0, 1.0, 1.0], # red
                    metallicFactor=0.0,
                    roughnessFactor=1.0)
       
        orientation_list = []
        facingxz_list = []
        for vvv,ttt in zip(vertices, cam_t):
            body_front_vert = vvv[5295]
            body_back_vert = vvv[4828]
            direction = -body_front_vert + body_back_vert
            direction[1] = 0.
            arrow_len = np.linalg.norm(direction)

            shaft_radius=0.02; tip_radius=0.04; tip_length=0.2
            # Create a cylinder for the shaft
            shaft_length = arrow_len - tip_length
            shaft = trimesh.creation.cylinder(radius=shaft_radius, height=shaft_length)
            shaft.apply_translation([0, 0, shaft_length / 2])  # Move the shaft up to align with arrow direction

            # Create a cone for the tip
            tip = trimesh.creation.cone(radius=tip_radius, height=tip_length)
            tip.apply_translation([0, 0, shaft_length + tip_length / 2])  # Position the tip at the arrow's end

            # Combine shaft and tip into one mesh
            arrow = trimesh.util.concatenate([shaft, tip])

            # Rotate the arrow to align with the desired direction
            z_axis = np.array([0, 0, 1])  # Default direction of the arrow
            rotation_matrix = trimesh.geometry.align_vectors(z_axis, direction, False)
            arrow.apply_transform(rotation_matrix)

            # Translate the arrow to the starting point
            arrow.apply_translation(body_back_vert)
            arrow.apply_translation(ttt)

            front_pt = body_front_vert + ttt
            back_pt = body_back_vert + ttt

            rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
            arrow.apply_transform(rot)
            
            front_pt = np.hstack([front_pt, np.ones(1)])
            back_pt = np.hstack([back_pt, np.ones(1)])
            
            front_pt = front_pt @ rot.T
            back_pt = back_pt @ rot.T

            rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
            arrow.apply_transform(rot)

            front_pt = front_pt @ rot.T
            back_pt = back_pt @ rot.T

            orientation_list.append(pyrender.Mesh.from_trimesh(arrow,smooth=False, material=material))
            
            front_pt = front_pt / front_pt[-1]
            back_pt = back_pt / back_pt[-1]
            # pdb.set_trace()

            facingxz_list.append([front_pt[:3], back_pt[:3]])
        


        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        for i,mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')
        for i,arrow in enumerate(orientation_list):
            scene.add(arrow, f'ori_{i}')


        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        # test proj
        focal = focal_length.detach().cpu().numpy()
        cam_x = camera_center[0].detach().cpu().numpy()
        cam_y = camera_center[1].detach().cpu().numpy()

        intrin = np.array([[focal, 0., cam_x],[0., focal, cam_y],[0., 0, 1.]])
        


        renderer.delete()


        # for json saving
        facing_direction_2d = []
        facing_direction_3d = []
        position_2d = []
        for pairs in facingxz_list:
            front_pt, back_pt = pairs
            front_pt_calibrated = front_pt / front_pt[2]
            back_pt_calibrated = back_pt / back_pt[2]
            # transform the left-hand coordinate to right-hand coordinate
            front_pt_calibrated[0] *= -1
            back_pt_calibrated[0] *= -1
            front_proj = (front_pt_calibrated @ intrin.T)
            back_proj = (back_pt_calibrated @ intrin.T)

            direction_2d = front_proj - back_proj
            direction_3d = front_pt - back_pt
            direction_3d[0] *= -1

            facing_direction_2d.append(direction_2d[:2].tolist())
            facing_direction_3d.append(direction_3d[:3].tolist())
        for i,mesh in enumerate(mesh_list):
            center = mesh.centroid
            center_pt_calibrated = center / center[2]
            center_pt_calibrated[0] *= -1
            center_proj = (center_pt_calibrated @ intrin.T)
            position_2d.append(center_proj[:2].tolist())



        return color, facing_direction_2d, facing_direction_3d, position_2d

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)
