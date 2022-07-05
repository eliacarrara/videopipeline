import unittest

import numpy as np

import videopypeline


class TestPipeline(unittest.TestCase):

    def test_linear(self):
        a1 = videopypeline.generators.Flatten(range(6))
        b1 = videopypeline.core.Function(lambda n: n * 3)(a1)
        c1 = videopypeline.core.Function(lambda n: n + 1)(b1)
        d1 = videopypeline.core.Function(lambda n: n * 2, aggregate=True)(c1)
        result = d1()

        self.assertListEqual([2, 8, 14, 20, 26, 32], result)

    def test_no_collect(self):
        tmp = []
        a1 = videopypeline.generators.Flatten(range(6))
        b1 = videopypeline.core.Function(lambda n: n * 3)(a1)
        c1 = videopypeline.core.Action(tmp.append, aggregate=True, collect=False)(b1)
        result = c1()

        self.assertEqual(len(result), 0)
        self.assertListEqual([0, 3, 6, 9, 12, 15], tmp)

    def test_linear_with_action(self):
        a1 = videopypeline.generators.Flatten(range(6))
        b1 = videopypeline.core.Function(lambda n: n * 3)(a1)
        c1 = videopypeline.core.Action(lambda *args: "some value")(b1)
        d1 = videopypeline.core.Function(lambda n: n * 2, aggregate=True)(c1)
        result = d1()

        self.assertListEqual([0, 6, 12, 18, 24, 30], result)

    def test_tree(self):
        a1 = videopypeline.generators.Flatten(range(70))  # First generator is intentionally larger than second one
        b1 = videopypeline.core.Function(lambda n: n * 3)(a1)

        a2 = videopypeline.generators.Flatten(['0', '1', '2', '3', '4', '5'])
        b2 = videopypeline.core.Function(lambda n: f' {n} ')(a2)

        c1 = videopypeline.core.Function(lambda *args: args, aggregate=True)([b1, b2])

        result = c1()
        true = [(0, ' 0 '), (3, ' 1 '), (6, ' 2 '), (9, ' 3 '), (12, ' 4 '), (15, ' 5 ')]

        self.assertEqual(len(true), len(result))
        for t, p in zip(true, result):
            self.assertTupleEqual(t, p)

    def test_filter_linear(self):
        a1 = videopypeline.generators.Flatten(range(10))
        b1 = videopypeline.core.Filter(lambda n: n < 5, aggregate=True)(a1)
        p = videopypeline.core.Pipeline(b1)
        result = p()

        self.assertListEqual([0, 1, 2, 3, 4], result)

    def test_graph(self):
        a1 = videopypeline.generators.Flatten(range(10))
        b1 = videopypeline.core.Function(lambda n: n + 1)(a1)
        b2 = videopypeline.core.Function(lambda n: n - 1)(a1)
        c1 = videopypeline.core.Function(lambda *a: np.mean(a, dtype=int), aggregate=True)([b1, b2])
        result = c1()

        self.assertListEqual(list(range(10)), result)

    def test_arg_select(self):
        tmp_a = []
        tmp_b = []

        a1 = videopypeline.generators.Flatten(range(5))
        b1 = videopypeline.core.Function(lambda n: (n, chr(ord('a') + n)))(a1)
        c1 = videopypeline.core.Action(tmp_a.append)(b1[0])
        c2 = videopypeline.core.Action(tmp_b.append)(b1[1])
        d1 = videopypeline.core.Action(lambda *a: None, aggregate=True)([c1, c2])
        result = d1()

        self.assertListEqual(list(range(5)), tmp_a)
        self.assertListEqual(['a', 'b', 'c', 'd', 'e'], tmp_b)

        true = [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e')]
        self.assertEqual(len(true), len(result))
        for t, p in zip(true, result):
            self.assertTupleEqual(t, p)

    def test_constant(self):
        a1 = videopypeline.generators.Constant(42)
        a2 = videopypeline.generators.Flatten(range(10))
        b1 = videopypeline.core.Function(lambda *a: chr(a[0]), aggregate=True, collect=True)([a1, a2])
        result = b1()

        self.assertEqual('*' * 10, ''.join(result))


if __name__ == '__main__':
    unittest.main()
