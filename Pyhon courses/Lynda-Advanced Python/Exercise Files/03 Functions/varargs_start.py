# Demonstrate the use of variable argument lists


# TODO: define a function that takes variable arguments
def addition(*args):
    result = 0
    for arg in args:
        result += arg
    return result


def main():
    # TODO: pass different arguments
    print(addition(1,2,3,4,5))
    print(addition(1,2,3,4,5,54))

    # TODO: pass an existing list
    nums = [2,3,4,5]
    print(addition(*nums))


if __name__ == "__main__":
    main()
