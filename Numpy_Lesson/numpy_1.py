import numpy as np

# ========================================================================
# array, size and basic syntax and operations
# matrix_2d = np.array([[1, 2, 3], [2, 3, 4]])
# print(matrix_2d)
# print(f"shape: {matrix_2d.shape}")

# matrix_3d = np.array([[[2, 3, 3], [2, 3, 1]], [[4, 5, 5], [5, 6, 6]]])

# print("\nmatrix_3d: ")
# print(matrix_3d)
# print(f"shape of 3d matrix: ", {matrix_3d.shape})


# print(f"size: {matrix_3d.size}")
# print(f"dimension: {matrix_3d.ndim}")
# print(f"dtype: {matrix_3d.dtype}")
# print(f"item size : {matrix_3d.itemsize} bytes")
# print(f"nbytes: {matrix_3d.nbytes} bytes")
# print(f"transpose: \n {matrix_3d.T}")


# matrix_transposed_3d = matrix_3d.transpose(0, 2, 1)

# print(f"transposed 3d: \n{matrix_transposed_3d}")

# =========================================================================
# indexing and slicing

# array_1d = np.array([1, 2, 3, 4, 5])

# print("access alement:", array_1d[0])
# print("Last element:", array_1d[-1])
# print("element NO.1 to NO.3 : ", array_1d[1:3])
# print("every 2 element: ", array_1d[::2])  # slicing: [start:stop:stop]

# array_2d = np.array([[2, 3, 4], [2, 2, 1], [4, 5, 5], [5, 6, 6]])
# print("\n2D array:")
# print(array_2d)
# print("element at row 1 column 2: ", array_2d[1, 2])
# print("first row:", array_2d[0, :])
# print("second column:", array_2d[:, 1])
# print("submatrix(first two rows and columns):")
# print(array_2d[0:2, 0:2])


# ============================================================================
# conditional element grabbing

# print("elements greater than 2 :", array_2d[array_2d > 2])

# ============================================================================
# array basic operations

# arr1 = np.array([1, 2, 3, 4])
# arr2 = np.array([4, 4, 5, 3])

# print("Array a: ", arr1)
# print("Array b: ", arr2)

# print("addition: ", arr2 + arr1)
# print("subtraction: ", arr2 - arr1)
# print("multiplication: ", arr1 * arr2)
# print("division: ", arr2 / arr1)
# print("power: ", arr1 ** arr2)
# print("square root : ", np.sqrt(arr1))

# print("+10 : ", arr1 + 10)
# print(" *2 :", arr1 * 2)
# print(" **2: ", arr1 ** 2)

# print("a > 2 :", arr1 > 2)
# =================================================================================
# array manipulation
# array1 = np.arange(12)
# print("original 1d array: ", array1)

# reshaped = array1.reshape(3, 4)
# print("\n reshaped to 3 by 4 matrix: ")
# print(reshaped)

# flattened = reshaped.flatten()
# print("\nflattened: ", flattened)

# auto_reshaped = array1.reshape(2, -1)
# print("\n auto reshaped(2 row, auto columns): ")
# print(auto_reshaped)

# a = np.array([[1, 2], [3, 4]])
# b = np.array([[2, 3], [4, 5]])
# concatenated_vertically = np.vstack((a, b))
# concatenated_horizontally = np.hstack((a, b))

# print(concatenated_vertically)
# print(concatenated_horizontally)

# ====================================================================================
# array axis
# matrix = np.array([[2, 3, 2], [2, 3, 2], [4, 4, 5]])
# matrix_3d = np.array([[[2, 3, 4], [2, 3, 1], [2, 2, 1]],
#                      [[1, 2, 9], [2, 2, 7], [0, 0, 3]]])

# print(matrix)
# print(matrix_3d)

# sum_axis0 = np.sum(matrix, axis=0)
# print(f"axis=0(down rows): {sum_axis0} -> shape: {sum_axis0.shape}")

# sum_axis1 = np.sum(matrix, axis=1)

# print(f"axis = 1 (across columns): {sum_axis1} -> shape: {sum_axis1.shape}")

# sum_axis2 = np.sum(matrix_3d, axis=2)
# print(
#     f"axis = 2 (collapse columns across each rows): {sum_axis2} -> shape: {sum_axis2.shape}")

# ========================================================================================
# matrix multiplication

# matrix_1 = np.array([[1, 2, 2], [2, 3, 5], [6, 5, 9]])
# matrix_2 = np.array([[2, 2, 1], [3, 3, 4], [5, 5, 2]])

# print(matrix_1 * matrix_2)  # element wise multiply

# print(matrix_2.dot(matrix_1))

# print(np.linalg.det(matrix_1))

# print(np.linalg.inv(matrix_1))

# eigenvalues, eigenvectors = np.linalg.eig(matrix_1)

# print(f"{eigenvalues}")
# print(f"{eigenvectors}")
# ===================================================================================
# arrray methods
array1 = np.array([[2, 1, 2], [3, 3, 2], [3, 4, 5]])

print("sorted array: ", np.sort(array1))

# array1.sort(axis=1)
# print(array1)

location = np.where(array1 < 2)
print(location)
