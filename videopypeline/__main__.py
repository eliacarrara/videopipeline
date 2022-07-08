import videopypeline as vpl


def asdf(arg):
    print(arg)
    return arg


a = vpl.generators.Iteration(range(10))
b = vpl.core.Function(lambda n: n * 2)(a)
c = vpl.core.Function(lambda n: n + 1, aggregate=True, collect=True)(b)
d = vpl.core.Function(asdf)(c)

print("cunt")
print(d())
