import videopypeline as vpl

# Define nodes
node_gen = vpl.generators.Value(1)
node_a = vpl.core.AbstractNode(lambda a: a + 5)
node_b = vpl.core.AbstractNode(lambda b: b * 2)

# Link node_gen as previous from node_a
node_a.model(node_gen)

# Link node_a as previous from node_b
node_b.model(node_a)

# Invoke node_b - broken down:
# gen = node_gen()
# a = node_a(gen)
# b = node_b(a)
print(node_b())  # 12
