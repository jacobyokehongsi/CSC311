import numpy as np
import matplotlib.pyplot as plt


# from scipy.spatial import distance
# a = np.random.rand(1,2)
# b = np.random.rand(1,2)
# dst = distance.euclidean(a, b)

# q1 a.
# arr = []
# for i in range(100):
#     x_y = np.random.rand(2) # 100, (dimension)
#     print(np.ndim(x_y))
#     arr.append(x_y)
# print(arr)

def cube(dim):
    cube = np.random.rand(100, dim)
    return cube


def distance(dim_array):
    dst_arr = []
    m_arr = []
    sd_arr = []
    for d in dim_array:
        dst = cube(d)
        # dst_arr.append(dst)
        lst = []
        for i in range(len(dst)):
            for j in range(len(dst)):
                if not np.array_equal(i, j):
                    if np.linalg.norm(dst[i] - dst[j]) not in lst:
                        lst.append(np.linalg.norm(dst[i] - dst[j]))
        m = np.mean(lst)
        sd = np.std(lst)

        m_arr.append(m)
        sd_arr.append(sd)

        # print(m)
        # print(sd)

    return m_arr, sd_arr


arr = [2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10]
m_arr, sd_arr = distance(arr)

plt.plot(arr, m_arr)
plt.plot(arr, sd_arr)
plt.xlabel("Dimensions")
plt.ylabel("Mean/SD")
# plt.savefig("q1.jpg")

# need to figure out matplotlib


# np.linalg.norm(x - y)
#
# a = np.random.rand(2)
# print(a)
# print(a.ndim)

# array = np.arange(1)
# print(array)
# print(array.shape)
#
# array = np.arange(2).reshape(2, 1)
# print(array)
# print(array.shape)

# array = np.arange(8).reshape(2, 2, 2)
# print(array)
# print(array.ndim)

# array = np.arange(2**4).reshape(2, 2, 2, 2)
# print(array)
# print(array.ndim)
#
# array = np.arange(2**5).reshape(2, 2, 2, 2, 2)
# print(array)
# print(array.shape)
#
# array = np.arange(2**6).reshape(2, 2, 2, 2, 2, 2)
# print(array)
# print(array.shape)
#
# array = np.arange(2**7).reshape(2, 2, 2, 2, 2, 2, 2)
# print(array)
# print(array.shape)
#
# array = np.arange(2**8).reshape(2, 2, 2, 2, 2, 2, 2, 2)
# print(array)
# print(array.shape)
#
# array = np.arange(2**9).reshape(2, 2, 2, 2, 2, 2, 2, 2, 2)
# print(array)
# print(array.shape)
#
# array = np.arange(2**10).reshape(2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
# print(array)
# print(array.shape)
