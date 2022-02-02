import videopipeline as vpl
import unittest
import numpy as np


class TestPipeline(unittest.TestCase):

    def test_1(self):
        a1 = vpl.nodes.generators.Flatten(range(6))
        b1 = vpl.core.Function(lambda n: n * 3)(a1)
        c1 = vpl.core.Function(lambda n: n + 1)(b1)
        d1 = vpl.core.Function(lambda n: n * 2, aggregate=True)(c1)

        result = d1()
        for t, p in zip([2, 8, 14, 20, 26, 32], result):
            self.assertEqual(t, p)

    def test_2(self):
        a1 = vpl.nodes.generators.Flatten(range(6))
        b1 = vpl.core.Function(lambda n: n * 3)(a1)
        c1 = vpl.core.Function(lambda n: n * 2, aggregate=True)(b1)

        result = c1()
        for t, p in zip([0, 6, 12, 18, 24, 30], result):
            self.assertEqual(t, p)

    def test_3(self):
        a1 = vpl.nodes.generators.Flatten(range(6))
        b1 = vpl.core.Function(lambda n: n * 3)(a1)

        a2 = vpl.nodes.generators.Flatten(['0', '1', '2', '3', '4', '5'])
        b2 = vpl.core.Function(lambda n: f' {n} ')(a2)

        c1 = vpl.core.Function(lambda *args: args[0], aggregate=True)([b1, b2])

        result = c1()
        for t, p in zip([[0, ' 0 '], [3, ' 1 '], [6, ' 2 '], [9, ' 3 '], [12, ' 4 '], [15, ' 5 ']], result):
            for t_, p_ in zip(t, p):
                self.assertEqual(t_, p_)

    def test_4(self):
        a1 = vpl.nodes.generators.Flatten(range(10))
        b1 = vpl.core.Filter(lambda n: n < 5, aggregate=True)(a1)

        result = b1()
        for t, p in zip([0, 1, 2, 3, 4], result):
            self.assertEqual(t, p)

    def test_5(self):
                    
        a1 = vpl.nodes.generators.Flatten(range(10))
        b1 = vpl.core.Function(lambda n: n + 1)(a1)
        b2 = vpl.core.Function(lambda n: n - 1)(a1)
        c1 = vpl.core.Function(lambda both: np.mean(both, dtype=int), aggregate=True)([b1, b2])

        result = c1()

        vpl.core.Pipeline(c1).render_model()

        for t, p in zip(range(10), result):
            self.assertEqual(t, p)


if __name__ == '__main__':
    unittest.main()
