r"""
This module contains the core functions and classes for the videopypeline package.
"""

from __future__ import annotations

import random
import typing
import time

import graphviz
import numpy as np


class AbortPipeline(Exception):
    """ Exception used to interrupt the pipeline execution.

    This Exception is used by :py:class:`core.Function`-Nodes. When the
    :py:attr:`core.Function.filter_fn`-Method returns False this exception is thrown.
    """
    pass


class AbstractNode:
    """Base node class.

    :param process_fn: The operation which will be wrapped by the node
    :type process_fn: callable
    :param name: The name of the node
    :type name: str, optional
    :param verbose: If true, prints information, defaults to False
    :type verbose: bool, optional
    :param debug_verbose: If true, prints debug information, defaults to False
    :type debug_verbose: bool, optional
    :param aggregate: If true, invokes the previous nodes until their generator exhausts. The output of every invocation
        is stored in a list before passing it to subsequent nodes (See ``collect``). Defaults to False.
    :type aggregate: bool, optional
    :param collect: If true, when aggregating, the nodes output will be stored, otherwise discarded (See ``aggregate``).
        Defaults to False.
    :type collect: bool, optional
    :param timeit: If true, the execution time for invoking this node is timed, defaults to False
    :type timeit: bool, optional
    """

    def __init__(self, process_fn: typing.Callable, name: str = "", verbose: bool = False, debug_verbose: bool = False,
                 aggregate: bool = False, collect: bool = True, timeit: bool = False):
        assert callable(process_fn)
        self.previous: typing.List[AbstractNode] = []  #: The previous linked nodes
        self.process_fn: typing.Callable = process_fn  #: The wrapped callable
        self.cache_data = None  #: The cached output of this node
        self.time_data = []  #: The collected timer data

        self.name: str = name  #: The name of the node
        self.verbose: bool = bool(verbose)  #: If true, console output is more verbose
        self.debug_verbose: bool = bool(debug_verbose)  #: If true, debug information will be printed
        self.aggregate: bool = bool(aggregate)  #: If true, :py:meth:`__call__` invokes :py:meth:`invoke` until the generator exhausts, otherwise only once
        self.collect: bool = bool(collect)  #: Applicable if :py:attr:`aggregate` is true. If true, the outputs will be collected into a list, otherwise discared
        self.timeit: bool = bool(timeit)  #: If true, the execution time is timed.

    def __call__(self, *args):
        """ Wrapper for :py:meth:`infer` and :py:meth:`model`

        This method is a convenience wrapper for two functions :py:meth:`infer` and :py:meth:`model`.
        If ``args[0]`` is of type ``AbstractNode`` or ``List[AbstractNode]``, :py:meth:`model` is invoked, otherwise
        :py:meth:`infer`.

        There are two types of inferring. This is controlled by :py:attr:`aggregate`. The two types are defined as
        follows:

        1. (``aggregate == False``) :py:meth:`infer` is called once and the output is returned.
        2. (``aggregate == True``) :py:meth:`infer` is called until the underlying generator exhausts.
           If :py:attr:`collect` is ``True`` the output for every iteration is stored in a list which is returned.
           If :py:attr:`collect` is ``True`` the output is discarded.

        :param args: asdf
        :return: asdf
        """
        is_modelling = self.is_modelling(*args)

        if is_modelling:  # model
            return self.model(args[0])

        elif not is_modelling and not self.aggregate:  # infer once
            return self.infer()

        elif not is_modelling and self.aggregate:  # infer until generator exhausts
            collection, iteration, run = [], 0, True

            try:
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
            except KeyboardInterrupt:
                if self.verbose:
                    print("Interrupted...")

            return collection
            
        else:
            assert False

    def __getitem__(self, n: int) -> AbstractNode:
        """

        :param n:
        :return:
        """
        assert isinstance(n, int)
        return AbstractNode(lambda *args: args[0][n], name=f"ArgSelect{n}")(self)

    def infer(self):
        """ Infer

        :return: asdf
        """
        assert isinstance(self.previous, list)
        assert all(isinstance(p, AbstractNode) for p in self.previous)

        # Return cached result if available
        if self.cache_data is not None:
            return self.cache_data

        # Infer previous nodes
        previous_output = [prev.infer() for prev in self.previous]

        if self.debug_verbose:
            size = previous_output.shape if isinstance(previous_output, np.ndarray) else ''
            print(f'Input: {type(previous_output)}{size}')

        t0 = None
        if self.timeit:
            t0 = time.perf_counter()

        # Infer current node
        self.cache_data = self.process_fn(*previous_output)

        if self.timeit:
            t1 = time.perf_counter()
            self.time_data.append(t1 - t0)

        if self.debug_verbose:
            size = self.cache_data.shape if isinstance(self.cache_data, np.ndarray) else ''
            print(f'Output: {type(self.cache_data)}{size}')

        return self.cache_data

    def clear_cache(self):
        """

        :return:
        """
        for p in self.previous:
            p.clear_cache()
        self.cache_data = None

    def model(self, node: AbstractNode) -> AbstractNode:
        """

        :param node:
        :return:
        """
        assert self.is_modelling(node)

        if isinstance(node, list):
            assert all(isinstance(n, AbstractNode) for n in node)
            self.previous.extend(node)
        else:
            assert isinstance(node, AbstractNode)
            self.previous.append(node)
        return self

    def start(self):
        """

        :return:
        """
        for p in self.previous:
            p.start()
        self.start_callback()

    def end(self):
        """

        :return:
        """
        for p in self.previous:
            p.end()
        self.end_callback()

    def start_callback(self):
        """

        :return:
        """
        pass

    def end_callback(self):
        """

        :return:
        """
        pass

    @staticmethod
    def is_modelling(*args) -> bool:
        """

        :param args:
        :return:
        """
        if len(args) != 1:
            return False

        node = args[0]
        one_parent = isinstance(node, AbstractNode)
        many_parents = isinstance(node, list) and all(isinstance(n, AbstractNode) for n in node)
        return one_parent or many_parents


class Function(AbstractNode):
    """

    """

    def __init__(self, process_fn, **kwargs):
        super().__init__(process_fn, **kwargs)


class Generator(Function):
    """

    """

    def __init__(self, generator_fn: typing.Callable[[], typing.Generator], **kwargs):
        super().__init__(self.generate, **kwargs)
        self.generator = generator_fn()

    def generate(self):
        return next(self.generator)


class Action(Function):
    """

    """

    def __init__(self, action_fn: typing.Callable, **kwargs):
        super().__init__(self.action, **kwargs)
        self.action_fn = action_fn

    def action(self, *args):
        self.action_fn(*args)
        return args[0] if len(args) == 1 else args


class Filter(Action):
    """

    """

    def __init__(self, filter_fn: typing.Callable[..., bool], **kwargs):
        super().__init__(self.filter, **kwargs)
        self.filter_fn = filter_fn

    def filter(self, *args):
        if not self.filter_fn(*args):
            raise AbortPipeline()


class Pipeline(Function):
    """

    """

    def __init__(self, end_node: AbstractNode | typing.List[AbstractNode], **kwargs):
        super().__init__(self.pipeline, **kwargs)
        if isinstance(end_node, list):
            assert all(isinstance(n, AbstractNode) for n in end_node)
            assert len(end_node[0].previous) == 0

            for i in range(len(end_node) - 1):
                end_node[i + 1](end_node[i])

            self.end_node: AbstractNode = end_node[-1]
        elif isinstance(end_node, AbstractNode):
            self.end_node: AbstractNode = end_node
        else:
            assert False, type(end_node)

        self.previous = [end_node]

    def pipeline(self, *args):
        self.end_node.start()
        ret = self.end_node(*args)
        self.end_node.end()
        return ret

    def render_model(self):
        def get_name(n):
            def rnd_hex():
                hex_chars = [c for c in "ABCDEF0123456789"]
                return ''.join([random.choice(hex_chars) for _ in range(5)])

            return (n.__class__.__name__ if n.name == "" else n.name) + "-" + rnd_hex()

        dot = graphviz.Digraph('pipeline' if self.name == "" else self.name, format='png')

        graph = self.traverse_dfs(self.end_node)
        tr = {n: get_name(n) for n in graph.keys()}

        for node, previous in self.traverse_dfs(self.end_node).items():
            name = tr[node]
            dot.node(name)

            for prev in previous:
                prev_name = tr[prev]
                dot.edge(prev_name, name)

        # dot.render(outfile="filename.png", cleanup=True)
        return dot

    @staticmethod
    def traverse_dfs(node: AbstractNode) -> typing.Dict[AbstractNode, typing.List[AbstractNode]]:
        nodes = {}
        stack = [node]

        while stack:
            current = stack.pop()

            if current not in nodes:
                nodes[current] = []

                # if previous is empty then current must be a generator
                assert current.previous or isinstance(current, Generator) or isinstance(current, Pipeline)

                for prev in current.previous:
                    stack.append(prev)
                    nodes[current].append(prev)

        return nodes
