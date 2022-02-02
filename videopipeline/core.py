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
        self.debug_verbose: bool = bool(kwargs["debug_verbose"]) if "debug_verbose" in kwargs else False  # TODO use
        self.aggregate: bool = bool(kwargs["aggregate"]) if "aggregate" in kwargs else False
        self.collect: bool = bool(kwargs["collect"]) if "collect" in kwargs else True
        
        self.cache = None

    def __call__(self, *args):
        is_modelling = self.is_modelling(*args)

        if is_modelling:  # model
            return self.model(args[0])

        elif not is_modelling and not self.aggregate:  # infer once
            return self.infer()

        elif not is_modelling and self.aggregate:  # infer until generator exhausts
            collection, iteration, run = [], 0, True
            
            while run:
                try:
                    if self.verbose:
                        print(f"Aggregating {iteration}")

                    output = self.infer()
                    self.clear_cache()
                    
                    if self.collect:
                        collection.append(output)                    
                except AbortPipeline:
                    self.clear_cache()
                except StopIteration:
                    run = False

                iteration += 1

            return collection
            
        else:
            assert False

    def __getitem__(self, n):
        # TODO return n-th element of output tuple, only applicable for multiple outputs
        pass

    def infer(self):
        assert isinstance(self.previous, list)
        assert all(isinstance(p, AbstractNode) for p in self.previous)

        # Return cached result if available
        if self.cache is not None:
            return self.cache

        # Infer previous nodes
        previous_output = [prev.infer() for prev in self.previous]

        # Infer current node
        if len(self.previous) == 0:
            self.cache = self.process_fn()
        elif len(self.previous) == 1:
            self.cache = self.process_fn(previous_output[0])
        elif len(self.previous) > 1:
            self.cache = self.process_fn(previous_output)

        return self.cache

    def clear_cache(self):
        for p in self.previous:
            p.clear_cache()
        self.cache = None

    def model(self, node):
        assert self.is_modelling(node)

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

    @staticmethod
    def is_modelling(*args):
        if len(args) != 1:
            return False

        node = args[0]
        one_parent = isinstance(node, AbstractNode)
        many_parents = isinstance(node, list) and all(map(lambda n: isinstance(n, AbstractNode), node))
        return one_parent or many_parents


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
            return ''.join([random.choice(hex_chars) for _ in range(5)])

        dot = graphviz.Digraph('pipeline-graph', format='png')
        tree = self.traverse_dfs(self.end_node)
        hex_chars = [c for c in "ABCDEF0123456789"]
        tr = {node: node.__class__.__name__ + "-" + rnd_hex() for node in tree.keys()}
        for node, previous in tree.items():
            name = tr[node]
            dot.node(name)

            for prev in previous:
                prev_name = tr[prev]
                dot.edge(prev_name, name)

        # dot.render(outfile="filename.png", cleanup=True)
        return dot

    @staticmethod
    def traverse_dfs(node):
        nodes = {}
        stack = [node]

        while stack:
            current = stack.pop()

            if current not in nodes:
                nodes[current] = []

                # if previous is empty then current must be a generator
                assert current.previous or isinstance(current, Generator)

                for prev in current.previous:
                    stack.append(prev)
                    nodes[current].append(prev)

        return nodes
