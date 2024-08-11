import unittest
import pyiron_nodes


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = pyiron_nodes.__version__
        print(version)
        self.assertTrue(version.startswith("0"))
