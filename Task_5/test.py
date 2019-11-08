from __future__ import print_function
from task_6 import substitutive


@substitutive
def f(x, y, z):
    print(x, y, z)


try:
    a = f(1, 2)
    a(3)
except Exception as e:
    print(e)

