import math
import random
import numpy as np




#计算扰动概率
def perturb_probability(d, m, eps):
    omega = math.comb(d-m, m) + (math.comb(d, m) - math.comb(d-m, m)) * pow(math.e, eps)
    pt = (math.comb(d-1, m-1) * pow(math.e, eps)) / omega
    pf = math.comb(d-m-1, m-1) / omega + (math.comb(d-1, m-1) - math.comb(d-m-1, m-1)) * pow(math.e, eps) / omega
    P = []
    P.append(math.comb(d-m, m) / omega)
    for i in range(1, m+1):
        prob = P[i-1] + math.comb(m, i) * math.comb(d-m, m-i) * pow(math.e, eps) / omega
        P.append(prob)
    return P, pt, pf

#随机数在数组中位置
def find_position(P, r):
    if not P:
        return None  # 如果数组为空，返回None
    if r < P[0]:
        return 0  # 如果r小于数组的第一个元素，返回0
    if r >= P[-1]:
        return None  # 如果r大于等于数组的最后一个元素，无合适位置

    low, high = 0, len(P) - 1
    while low < high:
        mid = (low + high) // 2
        if P[mid] <= r:
            low = mid + 1
        else:
            high = mid

    # 此时low是第一个大于r的元素的索引，检查这个索引的前一个元素
    if P[low] > r and P[low - 1] < r:
        return low
    return None

#集合中随机抽样
def random_subset(original_set, n):
    if n > len(original_set):
        raise ValueError("n is larger than the set size")
    sampled_list = random.sample(list(original_set), n)  # 将集合转换为列表，并从中随机抽取n个元素
    new_set = set(sampled_list)  # 将列表转换成集合
    return new_set

# 扰动
def perturbation(P, set_input, d, m):
    d_set = set(range(d))
    set_2 = d_set - set_input
    # print(set_2)
    random_number = np.random.rand()
    # print(random_number)
    c = find_position(P, random_number)
    # print(c)
    new_set_1 = random_subset(set_input, c)
    # print(new_set_1)
    new_set_2 = random_subset(set_2, m-c)
    # print(new_set_2)
    set_output = new_set_1 | new_set_2
    return set_output


#分布正则化归一化
def project_probability_simplex(p_estimate):
    k = len(p_estimate)  # Infer the size of the alphabet.
    p_estimate_sorted = np.sort(p_estimate)
    p_estimate_sorted[:] = p_estimate_sorted[::-1]
    p_sorted_cumsum = np.cumsum(p_estimate_sorted)
    i = 1
    while i < k:
        if p_estimate_sorted[i] + (1.0 / (i + 1)) * (1 - p_sorted_cumsum[i]) < 0:
            break
        i += 1
    lmd = (1.0 / i) * (1 - p_sorted_cumsum[i - 1])
    return np.maximum(p_estimate + lmd, 0)

#MSE,MAE
def mse_mae(distribution1, distribution2):
    # 将输入转换为numpy数组
    dist1 = np.array(distribution1)
    dist2 = np.array(distribution2)

    # 计算MSE
    mse = np.sum((dist1 - dist2) ** 2)

    # 计算MAE
    mae = np.sum(np.abs(dist1 - dist2))

    return mse, mae


#数据集提取
def merge_first_n_sets_as_set_of_sets(filename, n):
    with open(filename, 'r') as file:
        lines = file.readlines()
        big_set = set()

        for i, line in enumerate(lines[:n]):  # 仅处理前n行
            numbers = set(map(int, line.strip().split()))
            big_set.add(frozenset(numbers))  # 使用frozenset使集合可哈希，能被添加到另一个集合中

        return big_set


def fill_sets(big_set, m, d):
    """
    调整 big_set 中每个集合的元素数量至少为 m，如果元素数量少于 m，则从 0 到 d-1 的自然数中随机抽取填充。

    :param big_set: 集合的集合，例如 {frozenset([1, 2]), frozenset([3, 4, 5])}
    :param m: 每个集合应有的最小元素数量
    :param d: 可抽取的最大自然数范围（不包括d本身）
    """
    new_big_set = set()
    for small_set in big_set:
        # 复制原始的frozenset到一个新的可变set
        modified_set = set(small_set)
        # 如果当前集合元素少于m，进行填充
        while len(modified_set) < m:
            # 生成一个随机数并添加到集合中，集合会自动处理重复元素的问题
            new_element = random.randint(0, d - 1)
            modified_set.add(new_element)
        new_big_set.add(frozenset(modified_set))  # 使用不可变集合frozenset以便将其加入另一个集合
    return new_big_set


#计数
def count_elements_below_threshold(arr, threshold):
    # 生成一个新数组，其长度为threshold，每个位置初始化为0
    counts = [0] * threshold

    # 遍历数组中的每个元素
    for num in arr:
        # 如果元素小于阈值，则增加相应位置的计数
        if num < threshold:
            counts[num] += 1

    return counts

# 分布估计 聚合
def distribution_estimation(input_dat, n, eps, d, m):

    #d:数据域大小、m:集合大小
    threshold = d
    mses = []
    P, pt, pf = perturb_probability(d, m, eps)

    big_set = merge_first_n_sets_as_set_of_sets(input_dat, n)
    big_set = fill_sets(big_set, m, d)
    for i in range(1):
        original_arr = []
        perturbed_arr = []
        estimated_counts = []
        ori_distribution = []
        estimated_distribution = []
        for set_input in big_set:
            for things in set_input:
                original_arr.append(things)
            set_output = perturbation(P, set_input, d, m)
            for things in set_output:
                perturbed_arr.append(things)
        ori_counts = count_elements_below_threshold(original_arr, threshold)
        perturb_counts = count_elements_below_threshold(perturbed_arr, threshold)
        for counts in perturb_counts:
            estimated_counts.append((counts - n * pf) / (pt - pf))
        for counts in ori_counts:
            ori_distribution.append(counts / n)
        for counts in estimated_counts:
            estimated_distribution.append(counts / n)
        ori_distribution = project_probability_simplex(ori_distribution)
        estimated_distribution = project_probability_simplex(estimated_distribution)
        mse, mae = mse_mae(ori_distribution, estimated_distribution)
        # print('real', ori_distribution)
        # print('new', estimated_distribution)
        mses.append(mse)

    return np.mean(mses)



n = 100000
eps = 1.0
d = 256
m = 2
input_dat = 'setdata.dat'

mse = distribution_estimation(input_dat, n, eps, d, m)
print(mse)



