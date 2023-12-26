import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Rect

# https://github.com/adviksinghania/quadtree-python/blob/main/quadtree.py
class Point:
    def __init__(self, x, y, data=None):
        self.x = x
        self.y = y
        self.data = data

    def __iter__(self):
        yield (self.x, self.y, self.data)
class Rectangle:
    """Creating a Rectangle."""

    def __init__(self, x, y, w, h):
        """Properties of the Rectangle."""
        self.x = x  # coordinates of corner
        self.y = y
        self.w = w  # width
        self.h = h  # height
        self.points = set()  # list to store the contained points

    def intersects(self, other: 'Rectangle') -> bool:

        return not ((other.x - other.w  >= self.x + self.w) or
                    (other.x + other.w  <= self.x - self.w) or
                    (other.y - other.h  >= self.y + self.h) or
                    (other.y + other.h  <= self.y - self.h) )


    def box(self):
        return [self.x, self.x + self.w, self.y, self.y + self.h]

    # To print boundary of this function
    def __repr__(self):
        return f'({self.x}, {self.y}, {self.w}, {self.h})'

    def get_rect(self, color='k'):
        return Rect((self.x, self.y), self.w, self.h, fill=False,
                    color=color)
    def area(self):
        return self.w * self.h

    def __lt__(self, nxt):
        # return self.area() < nxt.area()
        return len(self.points) < len(nxt.points)
    def contains(self, point):
        check_x = self.x <= point.x <= self.x + self.w
        check_y = self.y <= point.y <= self.y + self.h
        return check_x and check_y

    def insert(self, point):
        if not self.contains(point):
            return False

        self.points.add(point)
        return True

class QuadTree:
    """Creating a quadtree."""

    def __init__(self, boundary, capacity):
        """Properties for a quadtree."""
        self.boundary = boundary  # object of class Rectangle
        self.capacity = capacity  # 4
        self.divided = False  # to check if the tree is divided or not
        self.northeast = None
        self.southeast = None
        self.northwest = None
        self.southwest = None


    def subdivide(self):
        """Dividing the quadtree into four sections."""
        x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.w, self.boundary.h

        north_east = Rectangle(x + w / 2, y, w / 2, h / 2)
        self.northeast = QuadTree(north_east, self.capacity)

        south_east = Rectangle(x + w / 2, y + h / 2, w / 2, h / 2)
        self.southeast = QuadTree(south_east, self.capacity)

        south_west = Rectangle(x, y + h / 2, w / 2, h / 2)
        self.southwest = QuadTree(south_west, self.capacity)

        north_west = Rectangle(x, y, w / 2, h / 2)
        self.northwest = QuadTree(north_west, self.capacity)
        self.divided = True

        for i in self.boundary.points:
            self.northeast.insert(i)
            self.southeast.insert(i)
            self.northwest.insert(i)
            self.southwest.insert(i)

    def insert(self, point):
        # If this major rectangle does not contain the point no need to check subdivided rectangle
        if not self.boundary.contains(point):
            return

        if len(self.boundary.points) < self.capacity:
            self.boundary.insert(point)  # add the point to the list if the length is less than capacity
        else:
            if not self.divided:
                self.subdivide()

            self.northeast.insert(point)
            self.southeast.insert(point)
            self.southwest.insert(point)
            self.northwest.insert(point)

    def query(self, range:Rectangle, found:list=[]):
        if not self.boundary.intersects(range):
            return found
        else:
            for p in self.boundary.points:
                if range.contains(p):
                    found.append(p)
        if self.divided:
            self.northeast.query(range, found)
            self.southeast.query(range, found)
            self.southwest.query(range, found)
            self.northwest.query(range, found)
        return found

    def sortedRect(self, rect:list=[]):
        if(self.divided):
            self.northeast.sortedRect(rect)
            self.southeast.sortedRect(rect)
            self.southwest.sortedRect(rect)
            self.northwest.sortedRect(rect)
        else:
            heapq.heappush(rect, self.boundary)
        return rect





if __name__ == '__main__':
    xy = np.random.uniform(low=-10, high=10, size=(500, 2))
    txy = np.random.uniform(low=-5, high=5, size=(1, 2))
    boundary = Rectangle(-10, -10, 20, 20)
    test_boundary = Rectangle(txy[0, 0], txy[0, 1], 5, 5)
    tree = QuadTree(boundary, 8)
    for point in xy:
        tree.insert(Point(point[0], point[1]))


    plt.cla()
    ax = plt.gca()
    plt.scatter(xy[:,0], xy[:, 1], c='g', s=10)
    ax.add_patch(test_boundary.get_rect('r'))
    points = tree.query(test_boundary)
    X = [p.x for p in points]
    Y = [p.y for p in points]
    plt.scatter(X, Y, c='r', s=15)
    # for rect in tree.sortedRect():
    #     ax.add_patch(rect.get_rect())

    print('finish')
    plt.show()