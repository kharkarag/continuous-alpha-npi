import time
import math
import numpy as np
from PIL import Image
from skimage.draw import line, circle_perimeter

# import gym
# import gym_drawobjects

# env = gym.make("drawobjects-v0")


# square_actions = [np.array([0,-10])] * 5 + \
#                  [np.array([10,0])] * 5 + \
#                  [np.array([0,10])] * 5 + \
#                  [np.array([-10,0])] * 5

# triangle_actions = [np.array([5,-5])] * 5 + \
#                    [np.array([5,5])] * 5 + \
#                    [np.array([-10,0])] * 5

# c_size = 4
# c_rate = 4
# circle_actions = [np.array([c_size*c_rate*math.sin(x), c_size*c_rate*math.cos(x)]) for x in np.arange(0, 6.28, 0.1*c_rate)]


# square_vertices = [(100,100), (50,100), (50, 150), (100, 150), (100,100)]
square_vertices = [(100,100), (50,100), (50, 50), (100, 50), (100,100)]
triangle_vertices = [(100,100), (125,75), (150, 100), (100,100)]


def draw_figure(vertices, name):
    img = np.ones((200, 200), dtype=np.uint8)*255

    for i, vertex in enumerate(vertices[1:], start=1):
        rr, cc = line(vertices[i-1][0], vertices[i-1][1], vertex[0], vertex[1])
        img[rr, cc] = 0
    im = Image.fromarray(img)
    im.save(name)


draw_figure(square_vertices, 'ref_img/square.jpg')
draw_figure(triangle_vertices, 'ref_img/triangle.jpg')


img = np.ones((200, 200), dtype=np.uint8)*255
rr, cc = circle_perimeter(100, 125, 25)
img[rr, cc] = 0

im = Image.fromarray(img)
im.save('ref_img/circle.jpg')


img = np.ones((200, 200), dtype=np.uint8)*255

for i, vertex in enumerate(square_vertices[1:], start=1):
    rr, cc = line(square_vertices[i-1][0], square_vertices[i-1][1], vertex[0], vertex[1])
    img[rr, cc] = 0


for i, vertex in enumerate(triangle_vertices[1:], start=1):
    rr, cc = line(triangle_vertices[i-1][0], triangle_vertices[i-1][1], vertex[0], vertex[1])
    img[rr, cc] = 0

rr, cc = circle_perimeter(100, 125, 25)
img[rr, cc] = 0

im = Image.fromarray(img)
im.save('ref_img/total_figure.jpg')