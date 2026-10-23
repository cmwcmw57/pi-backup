"""Check delivered URDF behavior against measured masses and sensor placements."""
import sys
import json
import math
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(sys.argv.pop(1)) if __name__ == '__main__' and len(sys.argv) > 1 else Path(__file__).resolve().parents[1]

class RoverRequirements(unittest.TestCase):
    def setUp(self):
        self.robot = ET.parse(ROOT / 'rover_portable.urdf').getroot()
        self.links = {x.get('name'): x for x in self.robot.findall('link')}
        self.joints = {x.get('name'): x for x in self.robot.findall('joint')}
        self.frames = {'base_link': np.eye(4)}
        pending = list(self.joints.values())
        while pending:
            before = len(pending)
            for j in pending[:]:
                parent = j.find('parent').get('link')
                if parent not in self.frames:
                    continue
                o = j.find('origin')
                T = np.eye(4)
                T[:3, 3] = np.fromstring(o.get('xyz', '0 0 0'), sep=' ')
                T[:3, :3] = Rotation.from_euler('xyz', np.fromstring(o.get('rpy', '0 0 0'), sep=' ')).as_matrix()
                self.frames[j.find('child').get('link')] = self.frames[parent] @ T
                pending.remove(j)
            self.assertLess(len(pending), before, 'URDF must be a connected acyclic tree')

    def test_total_measured_mass_without_double_counting(self):
        mass = sum(float(v.find('inertial/mass').get('value')) for v in self.links.values() if v.find('inertial/mass') is not None)
        self.assertAlmostEqual(mass, 2.3263, places=8)

    def test_sensor_mounts_use_updated_global_positions(self):
        self.assertIn('camera_link', self.frames)
        self.assertIn('imu_link', self.frames)
        # Hand-derived CAD-to-ROS: (X,Y,Z) -> (Y,-X,Z-161.813) mm.
        np.testing.assert_allclose(self.frames['camera_link'][:3, 3], [0.116619, 0, -0.004122], atol=2e-8)
        np.testing.assert_allclose(self.frames['imu_link'][:3, 3], [0, 0, -0.024600], atol=2e-8)

    def test_stereo_baseline_and_35_degree_downward_optical_axes(self):
        self.assertIn('camera_infra1_optical_frame', self.frames)
        self.assertIn('camera_infra2_optical_frame', self.frames)
        left = self.frames['camera_infra1_optical_frame']
        right = self.frames['camera_infra2_optical_frame']
        np.testing.assert_allclose(left[:3, 3] - right[:3, 3], [0, 0.050, 0], atol=1e-10)
        # Independent geometry: ROS +X forward, +Z up; downward ray is -Z.
        c, s = math.cos(math.radians(35)), math.sin(math.radians(35))
        np.testing.assert_allclose(left[:3, 2], [c, 0, -s], atol=1e-10)
        np.testing.assert_allclose(right[:3, 2], [c, 0, -s], atol=1e-10)
        np.testing.assert_allclose(left[:3, 3], [0.116619 - .0043*c, 0.025, -0.004122 + .0043*s], atol=2e-8)

    def test_reference_com_matches_actual_urdf_mass_distribution(self):
        refs = json.loads((ROOT / 'config/cad_reference.json').read_text())
        moment = np.zeros(3)
        mass = 0.
        for name, link in self.links.items():
            inertial = link.find('inertial')
            if inertial is None:
                continue
            m = float(inertial.find('mass').get('value'))
            p = np.fromstring(inertial.find('origin').get('xyz'), sep=' ')
            T = self.frames[name]
            moment += m * (T[:3, :3] @ p + T[:3, 3])
            mass += m
        np.testing.assert_allclose(moment/mass, refs['zero_pose_com_ros_m'], atol=1e-10)
        # g -> kg and mass-weighted aggregate are explicit, never STL-density defaults.
        self.assertAlmostEqual(float(self.links['camera_link'].find('inertial/mass').get('value')), .072)
        self.assertAlmostEqual(float(self.links['imu_link'].find('inertial/mass').get('value')), .0024)

    def test_chassis_com_follows_hollow_cad_geometry(self):
        components = json.loads((ROOT/'config/components.json').read_text())
        chassis = next(c for c in components if c.get('occurrence_id') == 203)
        # Independently audited native CAD volume centroid, transformed to CAD world mm.
        # A filled envelope box wrongly places this at (0,-5,137.813).
        np.testing.assert_allclose(chassis['global_com_cad_mm'],
                                   [.21315529, -7.99903009, 127.93272087], atol=1e-4)

    def test_all_available_components_have_stl_inertial_sources(self):
        components=json.loads((ROOT/'config/components.json').read_text())
        available=[c for c in components if not c['source_model'].startswith('omitted_')]
        self.assertEqual(len(available),28)
        for c in available:
            self.assertIn('inertial_mesh',c,c['source_model'])
            self.assertTrue((ROOT/c['inertial_mesh']).is_file())
        self.assertEqual(len([c for c in components if c['link']=='camera_link']),1)
        self.assertEqual(len([c for c in components if c['link']=='imu_link']),1)

    def test_wheel_local_z_and_opposed_passive_rockers(self):
        for c in ['fl', 'fr', 'rl', 'rr']:
            j = self.joints[c + '_wheel_joint']
            self.assertEqual(j.get('type'), 'continuous')
            np.testing.assert_array_equal(np.fromstring(j.find('axis').get('xyz'), sep=' '), [0, 0, 1])
            self.assertEqual(j.find('parent').get('link'), c + '_steering_link')
        m = self.joints['right_rocker_joint'].find('mimic')
        self.assertEqual(m.get('joint'), 'left_rocker_joint')
        self.assertEqual(float(m.get('multiplier')), -1)

    def test_mesh_paths_and_inertia_are_usable(self):
        for link in self.links.values():
            for mesh in link.findall('.//mesh'):
                self.assertTrue((ROOT / mesh.get('filename')).is_file(), mesh.get('filename'))
            inertia = link.find('inertial/inertia')
            if inertia is None:
                continue
            vals = {k: float(v) for k, v in inertia.attrib.items()}
            I = np.array([[vals['ixx'], vals['ixy'], vals['ixz']], [vals['ixy'], vals['iyy'], vals['iyz']], [vals['ixz'], vals['iyz'], vals['izz']]])
            eigenvalues = np.linalg.eigvalsh(I)
            self.assertGreater(eigenvalues[0], 0)
            self.assertLessEqual(eigenvalues[2], eigenvalues[0] + eigenvalues[1] + 1e-10)

if __name__ == '__main__':
    unittest.main()
