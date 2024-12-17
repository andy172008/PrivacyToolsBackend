import math
import random
import numpy as np
import pandas as pd






def read_csv_to_dict(csv_file):
    df = pd.read_csv(csv_file, header=None, names=['x', 'y'])
    records = df.to_dict('records')
    return records


def select_random_rows_as_dict(records, num_rows):
    # 确保不会选取超过列表长度的行数
    num_rows = min(num_rows, len(records))
    random_rows = random.sample(records, num_rows)
    return random_rows

# 扰动
def generate_points_with_density(coordinates, k, epsilon):
    def generate_random_circle_points(center, radius, num_points):
        points = []
        for _ in range(num_points):
            angle = random.uniform(0, 2 * math.pi)
            r = radius * math.sqrt(random.uniform(0, 1))
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            points.append((x, y))
        return points

    result = []

    # 直接遍历坐标列表
    for center in coordinates:
        # 基于概率密度生成一个随机半径
        radius = np.random.gamma(2 * k + 1, 1 / epsilon, size=1)[0]
        # 在此半径的圆内生成k个随机点
        points = generate_random_circle_points(center, radius, k)
        result.extend(points)

    return result


def count_points_in_rectangles(points, rectangles):
    # 初始化一个列表来存储每个矩形内点的数量
    rectangle_counts = [0] * len(rectangles)

    # 遍历所有点
    for point in points:
        # 遍历所有矩形
        for i, (bottom_left, top_right) in enumerate(rectangles):
            # 检查点是否在矩形内（包含边界）
            if bottom_left[0] <= point[0] <= top_right[0] and bottom_left[1] <= point[1] <= top_right[1]:
                # 如果是，则增加该矩形中的点的计数
                rectangle_counts[i] += 1

    # 计算所有矩形中点的总数
    total_points = sum(rectangle_counts)

    # 如果总数为零，则所有矩形的点的比例都是零
    if total_points == 0:
        return rectangle_counts

    # 计算每个矩形中的点的比例
    rectangle_proportions = [count / total_points for count in rectangle_counts]

    return rectangle_proportions

#mse
def mean_squared_error(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length.")

    mse = sum(((p - q) ** 2)**0.5 for p, q in zip(list1, list2)) / len(list1)
    return mse


def find_factors_closest_to_sqrt(n):
    for i in range(int(n ** 0.5), 0, -1):
        if n % i == 0:
            return i, n // i
    return 1, n

# 长方形分成正方形
def divide_rectangle_into_squares(rectangle, num_squares):
    # 计算矩形的宽度和高度
    length = abs(rectangle[2][0] - rectangle[0][0])
    width = abs(rectangle[1][1] - rectangle[0][1])

    num_rows = math.ceil(math.sqrt(num_squares * width / length))
    num_cols = math.ceil(math.sqrt(num_squares * length / width))

    #print(num_cols, num_rows)

    square_size = width / math.sqrt(num_squares * width / length)

    # 生成正方形中心点坐标
    squares_centers = []
    for row in range(num_rows):
        for col in range(num_cols):
            center_x = rectangle[0][0] + (col + 0.5) * square_size
            center_y = rectangle[0][1] + (row + 0.5) * square_size

            squares_centers.append((center_x, center_y))

    return square_size, squares_centers, len(squares_centers), num_cols, num_rows

def plot_divided_grid_with_complete_coverage(grid_centers, grid_length, n):
    # Calculate the bounding rectangle
    x_coords = [x for x, _ in grid_centers]
    y_coords = [y for _, y in grid_centers]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    rect_width = max_x - min_x + grid_length
    rect_height = max_y - min_y + grid_length

    # Find the division factors closest to the square root of n
    regions_across, regions_down = find_factors_closest_to_sqrt(n)
    region_width = rect_width / regions_down
    region_height = rect_height / regions_across


    rectangles_coordinates = []
    for i in range(regions_across):
        for j in range(regions_down):
            x1 = min_x + j * region_width - grid_length / 2
            y1 = min_y + i * region_height - grid_length / 2
            x2 = x1 + region_width
            y2 = y1 + region_height
            rectangles_coordinates.append(((x1, y1), (x2, y2)))

    return rectangles_coordinates
# 聚合
def distribution_estimation(eps, num, input_csv):
    d = 50
    r = 1/10
    k = 1
    df = pd.read_csv(input_csv, header=None, names=['x', 'y'])
    # 计算最小矩形的边界
    min_x, max_x = df['x'].min(), df['x'].max()
    min_y, max_y = df['y'].min(), df['y'].max()

    # 生成矩形的四个顶点坐标
    rectangle_points = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
    square_size, squares_centers, squares_nums, num_cols, num_rows = divide_rectangle_into_squares(rectangle_points, d)
    rectangles_coordinates = plot_divided_grid_with_complete_coverage(squares_centers, square_size, int(1 / r))

    records = read_csv_to_dict(input_csv)
    random_ori_data = select_random_rows_as_dict(records, num)
    points = [(point['x'], point['y']) for point in random_ori_data]
    # print(random_ori_data)
    ori_proportion = count_points_in_rectangles(points, rectangles_coordinates)
    # print(ori_proportion)
    # mse = mean_squared_error(new_proportion, ori_proportion)
    mses = []
    for i in range(1):
        new_two_dimensiom = generate_points_with_density(points, k, eps)
        # new_points = [(point['x'], point['y']) for point in new_two_dimensiom]
        new_proportion = count_points_in_rectangles(new_two_dimensiom, rectangles_coordinates)
        # print(new_proportion)
        mses.append(mean_squared_error(new_proportion, ori_proportion))
        print('real', ori_proportion)
        print('new', new_proportion)
    return np.mean(mses)

input_csv = 'local.csv'
eps = 1.0
num = 100000
mse = distribution_estimation(eps, num, input_csv)
print(mse)

