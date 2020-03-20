# Demonstrate the usage of Counter objects

from collections import Counter


def main():
    # list of students in class 1
    class1 = ["Bob", "Becky", "Chad", "Darcy", "Frank", "Hannah"
              "Kevin", "James", "James", "Melanie", "Penny", "Steve"]

    # list of students in class 2
    class2 = ["Bill", "Barry", "Cindy", "Debbie", "Frank",
              "Gabby", "Kelly", "James", "Joe", "Sam", "Tara", "Ziggy"]

    # TODO: Create a Counter for class1 and class2
    count1 = Counter(class1)
    count2 = Counter(class2)

    # TODO: How many students in class 1 named James?
    print(count1["James"])

    # TODO: How many students are in class 1?
    print(sum(count1.values()))

    # TODO: Combine the two classes
    count1.update(count2)
    print(sum(count1.values()))

    # TODO: What's the most common name in the two classes?
    print(count1.most_common(4))

    # TODO: Separate the classes again
    count1.subtract(count2)
    print(count1.most_common(4))

    # TODO: What's common between the two classes?
    print(count2 & count1)

if __name__ == "__main__":
    main()
