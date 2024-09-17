

a = 50
b = a
c = 50

print("Initial id values:")
print(id(a))
print(id(b))
print(id(c))

print("comparisons between a, b, and c:")
print(a is b)
print(b is c)
print(c is a)

a = 60

print("id values after changing a to 60:")
print(id(a))
print(id(b))
print(id(c))

a = None
del(c)
print("id values after setting a to None:")
print(id(a))
print(id(b))
print(id(c))