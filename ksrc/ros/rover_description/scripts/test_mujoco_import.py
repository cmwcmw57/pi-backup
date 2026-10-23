import sys
from pathlib import Path
import unittest
import mujoco

ROOT = Path(sys.argv.pop(1)) if __name__ == '__main__' and len(sys.argv)>1 else Path(__file__).resolve().parents[1]

class MuJoCoImport(unittest.TestCase):
    def test_full_robot_is_floating_and_coupled_with_measured_mass(self):
        model = mujoco.MjModel.from_xml_path(str(ROOT/'mjcf/rover.xml'))
        self.assertAlmostEqual(float(sum(model.body_mass)),2.3263,places=8)
        self.assertEqual(model.nq,17)
        self.assertEqual(model.nv,16)
        self.assertEqual(model.nu,8)
        self.assertEqual(model.neq,1)
        self.assertEqual(model.ncam,2)

if __name__=='__main__':
    unittest.main()
