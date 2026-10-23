"""Render the actual MJCF; optional annotations are not physical geometry."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import json
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def main():
    m = mujoco.MjModel.from_xml_path(str(ROOT/'mjcf/rover.xml'))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    refs = json.loads((ROOT/'config/cad_reference.json').read_text())
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
    expected = np.array(refs['zero_pose_com_ros_m']) + d.xpos[base]
    np.testing.assert_allclose(d.subtree_com[base], expected, atol=1e-10)
    m.vis.global_.offwidth = 960
    m.vis.global_.offheight = 640
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    with mujoco.Renderer(m, height=640, width=960) as renderer:
        view = mujoco.MjvCamera()
        view.lookat[:] = [0, 0, .09]
        view.distance = .65
        view.azimuth = 135
        view.elevation = -25
        renderer.update_scene(d, camera=view, scene_option=opt)
        Image.fromarray(renderer.render()).save(ROOT/'docs/mujoco_preview.png')
        Image.fromarray(renderer.render()).save(ROOT/'docs/assembly_preview.png')
        # Transparent chassis reveals the aggregate COM; red arrow = optical ray.
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'base_link_visual')
        m.geom_rgba[gid, 3] = .18
        view.azimuth = 90
        view.elevation = -8
        view.distance = .55
        renderer.update_scene(d, camera=view, scene_option=opt)
        sphere = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(sphere, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.005]*3),
                           d.subtree_com[base], np.eye(3).ravel(), np.array([1.,0.,1.,1.]))
        renderer.scene.ngeom += 1
        frame = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'camera_stereo_center_optical_frame')
        start = d.xpos[frame]
        ray = d.xmat[frame].reshape(3,3)[:,2]
        np.testing.assert_allclose(ray, [.819152044289,0,-.573576436351], atol=1e-10)
        arrow = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                           np.zeros(3), np.eye(3).ravel(), np.array([1.,.1,.1,1.]))
        mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, .003, start, start+.075*ray)
        renderer.scene.ngeom += 1
        Image.fromarray(renderer.render()).save(ROOT/'docs/camera_35deg_com.png')
    print('Rendered 35-degree camera and verified MuJoCo aggregate COM against URDF-derived reference.')

if __name__ == '__main__':
    main()
