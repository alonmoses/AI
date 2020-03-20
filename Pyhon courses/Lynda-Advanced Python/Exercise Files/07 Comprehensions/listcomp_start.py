# Demonstrate how to use list comprehensions


def main():
    # define two lists of numbers
    evens = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    odds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    # TODO: Perform a mapping and filter function on a list
    evenSquare = list(
        map(lambda s: s**2, filter(lambda f: f>4 and f<17, evens))
    )
    print(evenSquare)
    # TODO: Derive a new list of numbers frm a given list
    evenSquare3 = [e**3 for e in evens]
    print(evenSquare3)

    # TODO: Limit the items operated on with a predicate condition
    oddSquare = [e**2 for e in odds if e>3 and e<15]
    print(oddSquare)

if __name__ == "__main__":
    main()
