

x = 3

try:
    print(x)
    print(x+3)
    print("3" + x)
except TypeError as e:
    print(f"TypeError: {e}")
finally:
    print("This will always execute")