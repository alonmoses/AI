# deque objects are like double-ended queues

import collections
import string


def main():
    
    # TODO: initialize a deque with lowercase letters
    deque = collections.deque(string.ascii_lowercase)

    # TODO: deques support the len() function
    print("lowercase letters number:", len(deque))

    # TODO: deques can be iterated over
    # for element in deque:
    #     print(element.upper(), end = ",")

    # TODO: manipulate items from either end
    deque.pop()
    deque.popleft()
    deque.append(3)
    deque.appendleft(5)
    print(deque)

    # TODO: rotate the deque
    deque.rotate(2)
    print(deque)

if __name__ == "__main__":
    main()
