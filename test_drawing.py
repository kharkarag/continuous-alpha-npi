import math
import numpy as np

cur = np.array([100.5,100.5])
action = np.array([-50, 100])
target = cur + action

# print(f"Current: {cur}")

# while True:
#     nex = np.copy(cur)

#     for i in range(cur.shape[0]):
#         if action[i] >= 0:
#             nex[i] = np.floor(cur[i] + np.sign(action[i]))
#         else:
#             nex[i] = np.ceil(cur[i] + np.sign(action[i]))

#     deltas = (nex - cur) / action
#     min_idx = np.argmin(np.abs(deltas))

#     maj_length = (deltas*action)[min_idx]
#     min_length = action[1-min_idx]/action[min_idx] * maj_length

#     dist = math.sqrt(maj_length ** 2 + min_length ** 2)
#     dist_color = int((1 - dist / math.sqrt(2)) * 255)

#     cur[min_idx] += maj_length
#     cur[1-min_idx] += min_length

#     print(f"Current: {str(cur):<16} | {maj_length:>5.2f}, {min_length:>5.2f} | {dist:.2f} | {dist_color}")

#     if (np.floor(cur) + 0.5 == target).all():
#         break


c = (50, 50)
action = np.array([5, 10])


print(np.minimum(c + action, [200, 200]))

target = np.maximum(np.minimum(c + action, [200, 200]), [0,0])

print(target)