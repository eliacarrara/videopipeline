#!/usr/bin/env python

"""
"""

from __future__ import annotations
import graphviz
import random


class AbortPipeline(Exception):
    pass


class AbstractNode:

    def __init__(self, process_fn, **kwargs):
        assert callable(process_fn)
        self.previous = []

        self.process_fn = process_fn
        self.verbose: bool = bool(kwargs["verbose"]) if "verbose" in kwargs else False
        self.aggregate: bool = bool(kwargs["aggregate"]) if "aggregate" in kwargs else False
        self.collect: bool = bool(kwargs["collect"]) if "collect" in kwargs else True
        
        self.cache = None

    def __call__(self, *args):
        # TODO wrap function passed with abstract node
        is_modelling = self.is_modelling(*args)
        if is_modelling:
            return self.model(args[0])
        elif not is_modelling and self.aggregate:
            return self.infer_aggregate(*args)
        elif not is_modelling and not self.aggregate:
            return self.infer(*args)
        else:
            assert False

    @staticmethod
    def is_modelling(*args):
        if len(args) != 1:
            return False

        node = args[0]
        one_parent = isinstance(node, AbstractNode)
        many_parents = isinstance(node, list) and all(map(lambda n: isinstance(n, AbstractNode), node))
        return one_parent or many_parents

    def infer(self, *args):
        assert isinstance(self.previous, list)
        assert all(isinstance(p, AbstractNode) for p in self.previous)
        
        if self.cache is not None:
            return self.cache
        
        if self.aggregate:
            self.cache = self.infer_aggregate(*args)
            return self.cache

        if len(self.previous) == 0:
            self.cache = self.process_fn(*args)
            return self.cache
        elif len(self.previous) == 1:
            previous_output = self.previous[0].infer(*args)
            self.cache = self.process_fn(previous_output)
            return self.cache
        elif len(self.previous) > 1 and len(args) == 0:
            previous_output = [prev.infer() for prev in self.previous]
            return self.process_fn(previous_output)
        elif len(self.previous) > 1 and len(args[0]) > 1:
            assert len(self.previous) == len(args[0])

            previous_output = [prev.infer(arg) for prev, arg in zip(self.previous, args[0])]
            self.cache = self.process_fn(previous_output)

            return self.cache
        else:
            assert False, f'len(self.previous)={len(self.previous)}, len(args)={len(args)}'

    def infer_aggregate(self, *args):
        assert self.aggregate
        
        if self.cache is not None:
            return self.cache
        
        collection = []
        iteration = 0
        run = True

        while run:
            try:
                if self.verbose:
                    print(f"Aggregating {iteration}")

                if len(self.previous) == 0:
                    output = self.process_fn(*args)
                elif len(self.previous) == 1:
                    prev = self.previous[0].infer(*args)
                    output = self.process_fn(prev)
                else:
                    c_prev = []
                    for p, arg in zip(self.previous, args):
                        c_prev.append(p.infer(arg))
                    output = self.process_fn(c_prev)
                
                if self.collect:
                    collection.append(output)
                
            except AbortPipeline:
                pass
            except StopIteration:
                run = False

            iteration += 1
        
        self.cache = collection
        return self.cache

    def model(self, node):
        assert self.is_modelling(node)
        # TODO check if type hints match
        if isinstance(node, list):
            assert all(isinstance(n, AbstractNode) for n in node)
            self.previous.extend(node)
        else:
            assert isinstance(node, AbstractNode)
            self.previous.append(node)
        return self

    def start(self):
        for p in self.previous:
            p.start()
        self.start_callback()

    def end(self):
        for p in self.previous:
            p.end()
        self.end_callback()

    def start_callback(self):
        pass

    def end_callback(self):
        pass


class Function(AbstractNode):

    def __init__(self, process_fn, **kwargs):
        super().__init__(process_fn, **kwargs)


class Generator(Function):

    def __init__(self, generator_fn, **kwargs):
        super().__init__(self.generate, **kwargs)
        self.generator_fn = generator_fn
        self.generator = None

    def generate(self, *args):
        self.generator = self.generator_fn(*args) if self.generator is None else self.generator
        return next(self.generator)


class Action(Function):

    def __init__(self, action_fn, **kwargs):
        super().__init__(self.action, **kwargs)
        self.action_fn = action_fn

    def action(self, *args):
        self.action_fn(*args)
        assert len(args) > 0
        if len(args) == 1:
            return args[0]
        else:
            return args


class Filter(Action):

    def __init__(self, filter_fn, **kwargs):
        super().__init__(self.filter, **kwargs)
        self.filter_fn = filter_fn

    def filter(self, *args):
        if not self.filter_fn(*args):
            raise AbortPipeline()


class Pipeline(Function):

    def __init__(self, end_node, **kwargs):
        super().__init__(self.pipeline, **kwargs)
        # TODO use kwargs
        if isinstance(end_node, list):
            assert len(end_node[0].previous) == 0
            for i in range(len(end_node) - 1):
                end_node[i + 1](end_node[i])

            self.end_node = end_node[-1]
        elif isinstance(end_node, AbstractNode):
            self.end_node = end_node
        else:
            assert False

    def pipeline(self, *args):
        self.end_node.start()
        ret = self.end_node(*args)
        self.end_node.end()
        return ret

    def render_model(self):
        def rnd_hex():
            return ''.join([random.choice(hex_chars) for n in range(5)])

        dot = graphviz.Digraph('pipeline-graph', format='png')
        tree = self.traverse_tree(self.end_node)
        hex_chars = [c for c in "ABCDEF0123456789"]
        tr = {node: node.__class__.__name__ + "-" + rnd_hex() for node in tree.keys()}
        for node, prevs in tree.items():
            name = tr[node]
            dot.node(name)

            for prev in prevs:
                prev_name = tr[prev]
                dot.edge(prev_name, name)

        return dot

    @staticmethod
    def traverse_tree(node):
        nodes = {}
        stack = [node]

        while stack:
            current = stack.pop()

            assert current not in nodes
            nodes[current] = []

            # if previous is empty then current must be a generator
            assert current.previous or isinstance(current, Generator)

            for prev in current.previous:
                stack.append(prev)
                nodes[current].append(prev)

        return nodes
