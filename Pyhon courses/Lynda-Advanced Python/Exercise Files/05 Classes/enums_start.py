# define enumerations using the Enum base class

from enum  import Enum, unique, auto

#If two keys have the same value, unique will raise an error
@unique
class Fruit(Enum):
    APPLE = 1
    BANANA = 2
    Orange = 3
    Pear = auto()

def main():
    pass
    # TODO: enums have human-readable values and types
    print(Fruit.APPLE)
    print(type(Fruit.APPLE))
    print(repr(Fruit.APPLE))

    # TODO: enums have name and value properties
    print(Fruit.APPLE.name, Fruit.APPLE.value)

    # TODO: print the auto-generated value
    print(Fruit.Pear.value)

    # TODO: enums are hashable - can be used as keys
    myFruit = {}
    myFruit[Fruit.BANANA] = "Alon Moses"
    print(myFruit[Fruit.BANANA])

if __name__ == "__main__":
    main()
