"""Analytic unequal-volume fixture checks COM and measured-mass inertia scaling."""
import unittest
import numpy as np
import trimesh

class MeshInertia(unittest.TestCase):
    def properties(self, mesh, mass):
        try:
            from refine_mesh_inertia import measured_mesh_properties
        except ImportError:
            self.fail('STL-based measured-mass property calculation is not implemented')
        return measured_mesh_properties(mesh, mass)

    def test_unequal_solids_use_volume_centroid_and_measured_mass(self):
        a = trimesh.creation.box(extents=[1,1,1])
        b = trimesh.creation.box(extents=[2,1,1])
        b.apply_translation([3,0,0])
        p = self.properties(trimesh.util.concatenate([a,b]), 6.)
        # Volumes 1 and 2, uniform density 2, COM x=2. Bbox center is x=1.75.
        np.testing.assert_allclose(p['com_m'], [2,0,0], atol=1e-12)
        np.testing.assert_allclose(p['inertia_kg_m2'], np.diag([1,14,14]), atol=1e-12)
        self.assertAlmostEqual(p['mass_kg'], 6.)
        self.assertAlmostEqual(p['volume_m3'], 3.)

    def test_open_mesh_is_rejected(self):
        mesh = trimesh.creation.box()
        mesh.update_faces(np.arange(len(mesh.faces)-1))
        with self.assertRaises(ValueError):
            self.properties(mesh, 1.)

    def test_nonpositive_mass_is_rejected(self):
        with self.assertRaises(ValueError):
            self.properties(trimesh.creation.box(), 0.)

if __name__ == '__main__':
    unittest.main()
