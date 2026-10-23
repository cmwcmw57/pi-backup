"""Geometry preparation must preserve volume without double-counting overlap."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import trimesh

class MeshPreparation(unittest.TestCase):
    def prepare(self, mesh):
        try:
            from prepare_component_meshes import prepare_inertial_mesh
        except ImportError:
            self.fail('Component STL preparation is not implemented')
        return prepare_inertial_mesh(mesh)

    def test_overlapping_boxes_are_unioned_before_density_scaling(self):
        a=trimesh.creation.box()
        b=trimesh.creation.box();b.apply_translation([.5,0,0])
        result, audit=self.prepare(trimesh.util.concatenate([a,b]))
        self.assertTrue(result.is_watertight)
        self.assertAlmostEqual(result.volume,1.5,places=7)
        np.testing.assert_allclose(result.center_mass,[.25,0,0],atol=1e-7)
        np.testing.assert_allclose(result.bounds,[[-.5,-.5,-.5],[1,.5,.5]],atol=1e-7)

    def test_one_missing_triangle_is_closed_without_bbox_growth(self):
        mesh=trimesh.creation.box()
        mesh.update_faces(np.arange(len(mesh.faces)-1))
        result,audit=self.prepare(mesh)
        self.assertTrue(result.is_watertight)
        self.assertAlmostEqual(result.volume,1.,places=7)
        np.testing.assert_allclose(result.bounds,[[-.5]*3,[.5]*3],atol=1e-7)

    def test_valid_single_solid_is_preserved(self):
        mesh=trimesh.creation.box(extents=[1,2,3])
        result,audit=self.prepare(mesh)
        self.assertAlmostEqual(result.volume,6.)
        np.testing.assert_allclose(result.moment_inertia,mesh.moment_inertia,atol=1e-12)

    def test_edge_touching_solids_survive_stl_round_trip(self):
        from prepare_component_meshes import serialize_inertial_mesh
        a=trimesh.creation.box(extents=[.01,.01,.01])
        b=a.copy();b.apply_translation([.01,.01,0])
        prepared,_=self.prepare(trimesh.util.concatenate([a,b]))
        with tempfile.TemporaryDirectory() as directory:
            result,audit=serialize_inertial_mesh(prepared,Path(directory)/'touching.stl')
        self.assertTrue(result.is_volume)
        self.assertAlmostEqual(result.volume,2e-6,delta=2e-10)
        self.assertLess(audit['maximum_vertex_shift_mm'],.002)

if __name__=='__main__':
    unittest.main()
