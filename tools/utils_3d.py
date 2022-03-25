import trimesh
import numpy as np
from PIL import Image
import pyrender
import imageio

# os.environ["PYOPENGL_PLATFORM"] = "egl"


def save_gif(
    f_out,
    data,
    **kwargs,
):
    """Save a list of images as a gif"""
    data = [Image.fromarray(data[:, :, :, i]) for i in range(data.shape[3])]
    data[0].save(f_out, save_all=True, append_images=data[1:], **kwargs)


def load_gif(
    f_in,
    return_first=None,
):
    gif = imageio.get_reader(f_in)
    frames = []
    for idx, f in enumerate(gif):
        if (return_first is not None) and idx >= return_first:
            break
        frames.append(np.array(f))
    if len(frames[0].shape) == 2:
        frames = [np.expand_dims(f, axis=2) for f in frames]
    frames = [np.expand_dims(f, axis=3) for f in frames]
    return np.concatenate(frames, axis=3)



def normalize_unit_cube(mesh, scale=True):
    """Normalize a mesh so that it occupies a unit cube"""

    # Get the overall size of the object
    mesh = mesh.copy()
    mesh_min, mesh_max = np.min(mesh.vertices, axis=0), np.max(mesh.vertices, axis=0)
    size = mesh_max - mesh_min

    # Center the object
    mesh.vertices = mesh.vertices - ((size / 2.0) + mesh_min)

    # Normalize scale of the object
    if scale:
        mesh.vertices = mesh.vertices * (1.0 / np.max(size))
    mesh.fix_normals()
    return mesh


def render_mesh(
    mesh,
    objects=None,
    mode="RGB",
    remove_texture=False,
    yfov=(np.pi / 4.0),
    resolution=(1280, 720),
    xtrans=0.0,
    ytrans=0.0,
    ztrans=2.0,
    xrot=-25.0,
    yrot=45.0,
    zrot=0.0,
    spotlight_intensity=8.0,
    bg_color=255,
):
    assert len(resolution) == 2

    def _force_trimesh(mesh, remove_texture=False):
        """
        Forces a mesh or list of meshes to be a single trimesh object.
        """

        if isinstance(mesh, list):
            return [_force_trimesh(m) for m in mesh]

        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                mesh = trimesh.Trimesh()
            else:
                mesh = trimesh.util.concatenate(
                    tuple(
                        trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                        for g in mesh.geometry.values()
                    )
                )
        if remove_texture:
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        return mesh

    mesh = _force_trimesh(mesh, remove_texture)

    # Create a pyrender scene with ambient light
    scene = pyrender.Scene(ambient_light=np.ones(3), bg_color=bg_color)

    if objects is not None:
        for o in objects:
            o = o.subdivide_to_size(max_edge=0.05)
            n = o.vertices.shape[0]
            o.visual = trimesh.visual.create_visual(
                vertex_colors=np.hstack(
                    (
                        np.ones((n, 1)) * 0,
                        np.ones((n, 1)) * 0,
                        np.ones((n, 1)) * 255,
                        np.ones((n, 1)) * 50,
                    )
                )
            )
            scene.add(pyrender.Mesh.from_trimesh(o, wireframe=True))

    if not isinstance(mesh, list):
        mesh = [mesh]
    for m in mesh:
        if isinstance(m, trimesh.points.PointCloud):
            scene.add(
                pyrender.Mesh.from_points(m.vertices, colors=m.colors)
            )
        else:
            scene.add(pyrender.Mesh.from_trimesh(m))

    camera = pyrender.PerspectiveCamera(
        yfov=yfov, aspectRatio=resolution[0] / resolution[1]
    )

    # Apply translations
    trans = np.array(
        [
            [1.0, 0.0, 0.0, xtrans],
            [0.0, 1.0, 0.0, ytrans],
            [0.0, 0.0, 1.0, ztrans],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Apply rotations
    xrotmat = trimesh.transformations.rotation_matrix(
        angle=np.radians(xrot), direction=[1, 0, 0], point=(0, 0, 0)
    )
    camera_pose = np.dot(xrotmat, trans)
    yrotmat = trimesh.transformations.rotation_matrix(
        angle=np.radians(yrot), direction=[0, 1, 0], point=(0, 0, 0)
    )
    camera_pose = np.dot(yrotmat, camera_pose)
    zrotmat = trimesh.transformations.rotation_matrix(
        angle=np.radians(zrot), direction=[0, 0, 1], point=(0, 0, 0)
    )
    camera_pose = np.dot(zrotmat, camera_pose)

    # Insert the camera
    scene.add(camera, pose=camera_pose)

    # Insert a splotlight to give contrast
    spot_light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=spotlight_intensity,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 6.0,
    )
    scene.add(spot_light, pose=camera_pose)

    # Render!
    r = pyrender.OffscreenRenderer(*resolution)
    color, _ = r.render(scene)
    return np.array(Image.fromarray(color).convert(mode))
