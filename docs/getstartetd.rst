###########
Get Started
###########

************
Installation
************

Use the following line to install Videopypeline:

.. code-block::

   pip install videopypeline

************
Fundamentals
************

This section introduces fundamental concepts of the videopypeline package.

.. _node:

Node
=====

A node is simply wrapper for a python function. This allows the function to be used in the videopypeline framework.
A node allows to link other nodes and create a dependency. By linking node A to node B, node A becomes the previous node
to node B. When called, node A will now forward its output to node B which takes it as input.

.. literalinclude:: code_snippets/node.py
    :language: python
    :linenos:

Nodes are not restricted to a linear dependency. Every node can depend on one or more previous nodes and a node can be
dependent on by one or more nodes. This means that our dependency structure can be a graph.

TODO generator at the beginning
TODO refer to pipeline

:py:class:`videopypeline.core.AbstractNode` describes the base class for all node types in the videopypeline framework.
There are four basic node types:

1. Generators - ``() -> yield a`` - Used to generate data.
2. Functions - ``a -> b`` - Used to apply an operation on an input and return the output.
3. Actions - ``a -> a`` - Used to perform an operation on an input and return the input unchanged.
4. Filters - ``a -> if pred(a) then halt()`` - Skips the current iteration of a pipeline execution.

The following sections describe all types in depth.

Generator
---------
    
Function
--------

Action
------

Filter
------

.. _pipeline:

Pipeline
========

A pipeline is a collection of :ref:`Nodes <node>` which dependent on one another.

Modelling
---------

Inferring
---------

*****
Usage
*****

When adding custom Nodes, which inherit from :code:`vpl.core.Function`, make sure the output object is different from the
input object, as this could lead to unexpected behaviour.