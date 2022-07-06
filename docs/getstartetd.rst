###########
Get Started
###########

************
Installation
************

.. code-block::

   pip install videopypeline

********
Concepts
********

Nodes
=====

Generator
---------

Function
--------


Action
------

Pipeline
========


*****
Usage
*****

When adding custom Nodes, which inherit from :code:`vpl.core.Function`, make sure the output object is different from the
input object, as this could lead to unexpected behaviour.