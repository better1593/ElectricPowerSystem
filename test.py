x1 = "x1"
x2 = "x2"
x3 = "X3"
x4 = "x1"
a = []
a.append(x1)
a.append(x2)
a.append(x3)
a.append(x4)
a = set(a)
print(a)

b = {"x1":"dfd","x2":"dfgh"}
print(a,b.keys(),set(b.keys()).intersection(a))