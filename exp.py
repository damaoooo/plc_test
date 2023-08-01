s = ["1", "2", "3"]

try:
    s.index("4")
except ValueError as e:
    print(str(e))