import unittest
from pyiron_nodes.development import hash_based_storage
from pyiron_workflow import Workflow


@Workflow.wrap.as_function_node("result")
def add(x, z):
    return x + z

@Workflow.wrap.as_function_node("result")
def multiply(z, y):
    return z * y


class TestHashBasedStorage(unittest.TestCase):
    def test_extract_node_input(self):
        a = add(1, 2)
        m = multiply(a, 3)
        d = hash_based_storage.extract_node_input(m)
        self.assertTrue("z" in d)
        self.assertTrue("y" in d)
        self.assertEqual(d["y"], "3")
        self.assertTrue(d["z"].startswith("hash"))


if __name__ == '__main__':
    unittest.main()
