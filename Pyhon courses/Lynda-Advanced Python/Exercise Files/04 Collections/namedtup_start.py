# Demonstrate the usage of namdtuple objects

import collections


def main():
    # TODO: create a Point namedtuple
    Point = collections.namedtuple("Point", "a b")
    point1 = Point(3, 4)
    point2 = Point("alon", "bar")
    print(point1, point2)
    print(point2.b)

    # TODO: use _replace to create a new instance
    point1 = point1._replace(a = 30)
    print(point1)

if __name__ == "__main__":
    main()
