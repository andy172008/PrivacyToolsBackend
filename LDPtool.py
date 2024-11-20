# -*- coding: utf-8 -*-
# @Time : 2022/5/11 10:09 上午
# @Author : 贺星宇
# @File : LDPtool.py
# @Software: PyCharm

import copy
import math
import random
import xxhash
import numpy as np
from random import choice
from collections import Counter


class Data(object):
    def __init__(self, file_address: str, data_type: str):
        # 储存文件地址
        self.file_address = ''
        # 储存所有数据
        # 数据集文件中每行代表一个用户，如果一行只有一个数据，那么self.data的格式为[用户1,用户2,...,用户n]
        # 如果一行有多个数据，那么self.data的格式为[[用户1数据1，用户1数据2，用户1数据3],[用户2数据1，用户2数据2，用户2数据3],...,[用户3数据1，用户3数据2，用户3数据3]]
        self.data = []
        # 储存定义域
        self.domain = []
        # 储存数据的真实频率
        self.true_p = []
        # 储存数据的真实均值
        self.true_mean = 0
        # 记录共有多少条数据
        self.dataNum = 0
        # 记录数据类型
        self.data_type = data_type

        self.file_address = file_address

        if data_type == 'categorical':
            self.read_categorical_dataset()
        elif data_type == 'numeric':
            self.read_numeric_dataset()
        elif data_type == 'set':
            self.read_set_dataset()
        else:
            print('读取数据集时输入的数据类型参数不对')

        if data_type == 'categorical' or data_type == 'set':
            self.data_statistics_p()
        elif data_type == 'numeric':
            self.data_statistics_mean()

        # 打印数据相关信息
        # self.show_data_information()

    def read_categorical_dataset(self):
        with open(self.file_address, 'r') as f:
            for line in f.readlines():
                # 移除头尾换行符
                line = line.strip()
                self.data.append(int(float(line)))

    def read_numeric_dataset(self):
        with open(self.file_address, 'r') as f:
            for line in f.readlines():
                # 移除头尾换行符
                line = line.strip()
                self.data.append(float(line))

    def read_set_dataset(self):
        with open(self.file_address, 'r') as f:
            for line in f.readlines():
                # 移除头尾换行符
                line = line.strip()
                # 将一行中的数字划分开
                line_list = line.split(' ')
                line_list = [int(float(x)) for x in line_list]
                self.data.append(line_list)

    # 对所有数据做一个简要的统计
    def data_statistics_p(self):
        temp_data = []
        if self.data_type == 'categorical':
            temp_data = self.data
        elif self.data_type == 'set':
            for i in self.data:
                for j in i:
                    temp_data.append(j)
        count = Counter(temp_data)
        count_dict = dict(count)
        for x in count_dict:
            self.domain.append(x)
        self.domain.sort()
        self.dataNum = len(self.data)
        for x in self.domain:
            self.true_p.append(count_dict.get(x, 0) / self.dataNum)

    def data_statistics_mean(self):
        for x in self.data:
            self.true_mean += x
        self.dataNum = len(self.data)
        self.true_mean /= self.dataNum

    # 展示统计出的信息
    def show_data_information(self):
        feature_type = len(self.domain)
        print('共有数据%d条' % self.dataNum)
        print('属性种类共有%d个' % feature_type)
        for i in range(feature_type):
            print('第%d种属性，为%d，频率为%f' % (i + 1, self.domain[i], self.true_p[i]))


# 类别型数据
class GRR_USER(object):
    def __init__(self, epsilon: float, domain: list, data: int):
        super(GRR_USER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 原始数据定义域的长度
        self.d = len(domain)
        # 用户的原始数据
        self.data = data
        # 扰动数据
        self.per_data = -1

        # 为使用方便，定义e^\epsilon
        e_epsilon = np.exp(self.epsilon)

        # 协议中的扰动概率
        self.p = e_epsilon / (e_epsilon + self.d - 1)
        self.q = 1 / (e_epsilon + self.d - 1)

    def run(self):
        encode_x = self.encode(self.data)
        perturb_x = self.perturb(encode_x)
        self.per_data = perturb_x

    # 此时传入的是用户的原始数据，并不是该数据在domain中的位置
    def encode(self, x: int) -> int:
        return x

    # 返回的是扰动后的数据，并不是在domain中的
    def perturb(self, x: int) -> int:
        if np.random.uniform(0, 1) < self.p:
            return x
        else:
            per_x = choice(self.domain)
            # 当随机选择的元素与之前的x一致时，再进行随机选择，直到不一致为止
            while per_x == x:
                per_x = choice(self.domain)
            return per_x

    def get_per_data(self):
        return self.per_data


class GRR_SERVER(object):
    def __init__(self, epsilon: float, domain: list, per_datalist: list):
        super(GRR_SERVER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 原始数据定义域的长度
        self.d = len(domain)
        # 所有用户的扰动数据
        self.per_datalist = per_datalist
        # 用户数量
        self.n = len(per_datalist)
        # 频率估计结果
        self.es_data = []

        # 为使用方便，定义e^\epsilon
        e_epsilon = np.exp(self.epsilon)

        # 协议中的扰动概率
        self.p = e_epsilon / (e_epsilon + self.d - 1)
        self.q = 1 / (e_epsilon + self.d - 1)

    def estimate(self):
        per_data = self.per_datalist
        # 在获取扰动数据中元素频率时，一定要用字典，可以大大节省运行时间
        count = Counter(per_data)
        count_dict = dict(count)

        for x in self.domain:
            x_count = count_dict.get(x, 0)
            rs = (x_count - self.n * self.q) / (self.n * (self.p - self.q))
            self.es_data.append(rs)

    def get_es_data(self):
        return self.es_data


class SUE_USER(object):
    def __init__(self, epsilon, domain, data):
        super(SUE_USER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 原始数据定义域的长度
        self.d = len(domain)
        # 用户的原始数据
        self.data = data
        # 扰动数据
        self.per_data = -1

        # 为使用方便，定义e^\epsilon2
        e_epsilon2 = np.exp(self.epsilon / 2)

        # 协议中的扰动概率
        self.p = e_epsilon2 / (e_epsilon2 + 1)
        self.q = 1 / (e_epsilon2 + 1)

    def run(self):
        encode_x = self.encode(self.data)
        perturb_x = self.perturb(encode_x)
        self.per_data = perturb_x

    # 此时传入的是用户的原始数据，并不是该数据在domain中的位置
    def encode(self, x):
        l = list()
        user_data_index = self.domain.index(self.data)
        for i in range(self.d):
            if i == user_data_index:
                l.append(1)
            else:
                l.append(0)
        return l

    # 返回的是扰动后的数据，并不是在domain中的
    def perturb(self, x: list) -> list:
        for i in range(self.d):
            if x[i] == 1:
                if np.random.uniform() > self.p:
                    x[i] = 0
            elif x[i] == 0:
                if np.random.uniform() < self.q:
                    x[i] = 1
        return x

    def get_per_data(self):
        return self.per_data


class SUE_SERVER(object):
    def __init__(self, epsilon, domain, per_datalist):
        super(SUE_SERVER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 原始数据定义域的长度
        self.d = len(domain)
        # 所有用户的扰动数据
        self.per_datalist = per_datalist
        # 用户数量
        self.n = len(per_datalist)
        # 频率估计结果
        self.es_data = []

        # 为使用方便，定义e^\(epsilon/2)
        e_epsilon2 = np.exp(self.epsilon / 2)

        # 协议中的扰动概率
        self.p = e_epsilon2 / (e_epsilon2 + 1)
        self.q = 1 / (e_epsilon2 + 1)

    def estimate(self):
        array = np.sum(self.per_datalist, axis=0)
        count_dict = dict(zip(self.domain, array))

        for x in self.domain:
            x_count = count_dict.get(x, 0)
            rs = (x_count - self.n * self.q) / (self.n * (self.p - self.q))
            self.es_data.append(rs)

    def get_es_data(self):
        return self.es_data


class OUE_USER(object):
    def __init__(self, epsilon: float, domain: list, data: int):
        super(OUE_USER, self).__init__()
        self.epsilon = epsilon  # 隐私预算
        self.domain = domain  # 原始数据定义域
        self.d = len(domain)  # 原始数据定义域的长度
        self.data = data  # 用户的原始数据
        self.per_data = []  # 扰动数据

        e_epsilon = np.exp(self.epsilon)  # 为使用方便，定义e^\epsilon

        # 协议中的扰动概率
        self.p = 1 / 2
        self.q = 1 / (e_epsilon + 1)

    def run(self):
        encode_x = self.encode(self.data)
        perturb_x = self.perturb(encode_x)
        self.per_data = perturb_x

    def encode(self, x) -> list:
        l = list()
        user_data_index = self.domain.index(self.data)
        for i in range(self.d):
            if i == user_data_index:
                l.append(1)
            else:
                l.append(0)
        return l

    def perturb(self, x: list) -> list:
        for i in range(self.d):
            if x[i] == 1:
                if np.random.uniform() > self.p:
                    x[i] = 0
            elif x[i] == 0:
                if np.random.uniform() < self.q:
                    x[i] = 1
        return x

    def get_per_data(self):
        return self.per_data


class OUE_SERVER(object):
    def __init__(self, epsilon: float, domain: list, per_datalist: list):
        super(OUE_SERVER, self).__init__()
        self.epsilon = epsilon  # 隐私预算
        self.domain = domain  # 原始数据定义域
        self.d = len(domain)  # 原始数据定义域的长度
        self.per_datalist = per_datalist  # 所有用户的扰动数据
        self.n = len(per_datalist)  # 用户数量
        self.es_data = []  # 频率估计结果

        e_epsilon = np.exp(self.epsilon)  # 为使用方便，定义e^\epsilon

        # 协议中的扰动概率
        self.p = 1 / 2
        self.q = 1 / (e_epsilon + 1)

    def estimate(self):
        per_data = self.per_datalist

        array = np.sum(per_data, axis=0)
        count_dict = dict(zip(self.domain, array))

        for x in self.domain:
            x_count = count_dict.get(x, 0)
            rs = (x_count - self.n * self.q) / (self.n * (self.p - self.q))
            self.es_data.append(rs)

    def get_es_data(self):
        return self.es_data


class OLH_USER(object):
    def __init__(self, epsilon, domain, data):
        super(OLH_USER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义区间
        self.domain = domain
        # 用户的原始数据
        self.data = data
        # 扰动数据
        self.per_data = []

        # 为使用方便，定义e^\epsilon
        e_epsilon = np.exp(self.epsilon)

        # 设置g为最佳本地哈希的值
        self.g = int(round(e_epsilon)) + 1

        # 协议中的扰动概率
        self.p = e_epsilon / (e_epsilon + self.g - 1)
        self.q = 1.0 / (e_epsilon + self.g - 1)

    def run(self):
        encode_x = self.encode(self.data)
        perturb_x = self.perturb(encode_x[1])
        self.per_data = [encode_x[0], perturb_x]

    def encode(self, v: int):
        seed = random.randint(0, 100000)
        hash = (xxhash.xxh32(str(v), seed=seed).intdigest() % self.g)
        return seed, hash

    def perturb(self, x: int):
        new_domain = [i for i in range(self.g)]
        if np.random.uniform(0, 1) < self.p:
            return x
        else:
            per_x = choice(new_domain)
            # 当随机选择的元素与之前的x一致时，再进行随机选择，直到不一致为止
            while per_x == x:
                per_x = choice(new_domain)
            return per_x

    def get_per_data(self):
        return self.per_data


class OLH_SERVER(object):
    def __init__(self, epsilon: float, domain: list, per_datalist: list):
        super(OLH_SERVER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 扰动数据列表
        self.per_datalist = per_datalist
        # 用户数量
        self.n = len(per_datalist)
        # 频率估计结果
        self.es_data = []
        # 值域
        self.domain = domain

        e_epsilon = np.exp(self.epsilon)

        # 设置g为最佳本地哈希的值
        self.g = int(round(e_epsilon)) + 1

        # 协议中的扰动概率
        self.p = e_epsilon / (e_epsilon + self.g - 1)
        self.q = 1 / self.g

    def estimate(self):
        for x in self.domain:
            count = 0
            for data in self.per_datalist:
                if xxhash.xxh32(str(x), seed=data[0]).intdigest() % self.g == data[1]:
                    count = count + 1
            rs = (count - self.n * self.q) / (self.n * (self.p - self.q))
            self.es_data.append(rs)

    def get_es_data(self):
        return self.es_data


class EFM_USER(object):
    def __init__(self, epsilon: float, domain: list, data: int):
        super(EFM_USER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 原始数据定义域的长度
        self.domain_length = len(domain)
        # 用户的原始数据
        self.data = data
        # 扰动数据
        self.per_data = 0
        # 论文方案中使用的两个参数，m和d
        # m是hash后的定义域大小
        e = math.exp(self.epsilon)
        k = len(domain)
        theta = math.sqrt(e * (e * k - e + 1) / (k + e - 1))
        self.m = int(round(1 + theta))
        # d是用户扰动后数据中1的个数
        self.d = 1

        # 为使用方便，定义e^\epsilon
        e_epsilon = np.exp(self.epsilon)
        # 方案中所用到的概率
        self.q = (e_epsilon * self.d) / (e_epsilon * self.d + self.m - self.d)

    def run(self):
        # 用户数据被hash到0到m-1中的一个数字
        # 种子在100000内随机取一个，这个范围其实不是很重要
        seed = random.randint(0, 100000)
        hash_data = xxhash.xxh32(str(self.data), seed=seed).intdigest() % self.m
        # 目前的定义域，0到m-1
        hash_domain = [x for x in range(self.m)]
        # 用户数据位于定义域中的索引
        user_index = hash_domain.index(hash_data)
        # 去除了用户数据后的定义域索引
        hash_domain_without_user = copy.deepcopy(hash_domain)
        hash_domain_without_user.remove(user_index)

        # 生成长度为m的二进制向量
        y = [0 for _ in range(self.m)]
        temp = np.random.uniform()
        # 用户位设置为1，再挑选d-1个不同的值
        if temp < self.q:
            y[user_index] = 1
            temp_list = random.sample(hash_domain_without_user, self.d - 1)
            for i in temp_list:
                y[i] = 1
        # 用户位设置为0，再挑选d个不同的值
        else:
            temp_list = random.sample(hash_domain_without_user, self.d)
            for i in temp_list:
                y[i] = 1

        self.per_data = [seed, y]

    def get_per_data(self):
        return self.per_data


class EFM_SERVER(object):
    def __init__(self, epsilon: float, domain: list, per_datalist: list):
        super(EFM_SERVER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 原始数据定义域的长度
        self.domain_length = len(domain)
        # 所有用户的扰动数据
        self.per_datalist = per_datalist
        # 用户数量
        self.n = len(per_datalist)
        # 频率估计结果
        self.es_data = []
        # 论文方案中使用的两个参数，m和d
        # m是hash后的定义域大小
        e = math.exp(self.epsilon)
        k = len(domain)
        theta = math.sqrt(e * (e * k - e + 1) / (k + e - 1))
        self.m = int(round(1 + theta))
        # d是用户扰动后数据中1的个数
        self.d = 1

        # 为使用方便，定义e^\epsilon
        e_epsilon = np.exp(self.epsilon)

        # 方案中所用到的概率
        self.q = (e_epsilon * self.d) / (e_epsilon * self.d + self.m - self.d)

    def estimate(self):
        for x in self.domain:
            count = 0
            for user_seed_and_data in self.per_datalist:
                hash_x = xxhash.xxh32(str(x), seed=user_seed_and_data[0]).intdigest() % self.m
                if user_seed_and_data[1][hash_x] == 1:
                    count += 1
            rs = (self.m * count - self.n * self.d) / (self.n * (self.m * self.q - self.d))
            self.es_data.append(rs)

    def get_es_data(self):
        return self.es_data


class SS_USER(object):
    def __init__(self, epsilon: float, domain: list, data: int):
        super(SS_USER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 原始数据定义域的长度
        self.domain_length = len(domain)
        # 为了与论文中保持一致，这里将原始数据定义域的长度重新命名为k
        self.k = self.domain_length
        # 用户扰动后数据中1的个数
        k = self.k
        e = math.exp(self.epsilon)
        self.d = int(round(k / (e + 1)))
        # 用户的原始数据
        self.data = data
        # 扰动数据
        self.per_data = 0

        # 为使用方便，定义e^\epsilon
        e_epsilon = np.exp(self.epsilon)

        # 方案中所用到的概率
        self.q = (e_epsilon * self.d) / (e_epsilon * self.d + self.k - self.d)

    def run(self):
        # 为使用方便，定义e^\epsilon
        e_epsilon = np.exp(self.epsilon)

        # 子集选择比较特殊，编码和扰动过程无法分开，因此直接在run函数中写

        # 用户数据位于定义域中的索引
        user_index = self.domain.index(self.data)
        # 去除了用户数据后的定义域索引
        domain_without_user = [x for x in range(self.domain_length)]
        domain_without_user.remove(user_index)

        # 生成长度为k的二进制向量
        y = [0 for _ in range(self.k)]
        temp = np.random.uniform()
        # 用户位设置为1，再挑选d-1个不同的值
        if temp < self.q:
            y[user_index] = 1
            temp_list = random.sample(domain_without_user, self.d - 1)
            for i in temp_list:
                y[i] = 1
        # 用户位设置为0，再挑选d个不同的值
        else:
            temp_list = random.sample(domain_without_user, self.d)
            for i in temp_list:
                y[i] = 1

        self.per_data = y

    def get_per_data(self):
        return self.per_data


class SS_SERVER(object):
    def __init__(self, epsilon: float, domain: list, per_datalist: list):
        super(SS_SERVER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 原始数据定义域的长度
        self.domain_length = len(domain)
        # 为了与论文中保持一致，这里将原始数据定义域的长度重新命名为k
        self.k = self.domain_length
        # 用户扰动后数据中1的个数
        k = self.k
        e = math.exp(self.epsilon)
        self.d = int(round(k / (e + 1)))
        # 所有用户的扰动数据
        self.per_datalist = per_datalist
        # 用户数量
        self.n = len(per_datalist)
        # 频率估计结果
        self.es_data = []

        # 为使用方便，定义e^\epsilon
        e_epsilon = np.exp(self.epsilon)

        # 方案中所用到的概率
        self.q = (e_epsilon * self.d) / (e_epsilon * self.d + self.k - self.d)

    def estimate(self):
        per_data = self.per_datalist
        array = np.sum(per_data, axis=0)
        count_dict = dict(zip(self.domain, array))

        for x in self.domain:
            x_count = count_dict.get(x, 0)
            rs = ((self.k - 1) * x_count - self.n * self.d + self.n * self.q) / (self.n * (self.k * self.q - self.d))
            self.es_data.append(rs)

    def get_es_data(self):
        return self.es_data


# 数值型数据
class Duchi_USER(object):
    def __init__(self, epsilon: float, data: float):
        super(Duchi_USER).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 用户的原始数据
        self.data = data
        # 扰动数据
        self.per_data = 0.0

        # 为使用方便，定义e^\epsilon
        e_epsilon = np.exp(self.epsilon)

        self.p = (e_epsilon - 1) / (2 * e_epsilon + 2) * self.data + 0.5

    def run(self):
        temp = np.random.uniform()
        e_epsilon = np.exp(self.epsilon)

        if temp < self.p:
            self.per_data = (e_epsilon + 1) / (e_epsilon - 1)
        else:
            self.per_data = -(e_epsilon + 1) / (e_epsilon - 1)

    def get_per_data(self):
        return self.per_data


class Duchi_SERVER(object):
    def __init__(self, epsilon: float, per_datalist: list):
        super(Duchi_SERVER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 所有用户的扰动数据
        self.per_datalist = per_datalist
        # 估计均值结果
        self.es_mean = 0

    def estimate(self):
        # 均值估计的聚合方法很简单，直接将用户数据相加即可
        for x in self.per_datalist:
            self.es_mean += x
        self.es_mean /= len(self.per_datalist)

    def get_es_mean(self):
        return self.es_mean


class PM_USER(object):
    def __init__(self, epsilon: float, data: float):
        super(PM_USER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 用户的原始数据
        self.data = data
        # 扰动数据
        self.per_data = 0.0

        # 为使用方便，定义e^{\epsilon/2}
        e_epsilon_half = np.exp(self.epsilon / 2)

        # 下列参数都是PM机制中所使用的参数
        self.c = (e_epsilon_half + 1) / (e_epsilon_half - 1)
        self.p = (e_epsilon_half * e_epsilon_half - e_epsilon_half) / (2 * e_epsilon_half + 2)
        self.l = (self.c + 1) / 2 * self.data - (self.c - 1) / 2
        self.r = self.l + self.c - 1

    def run(self):
        # 在PM方案中，概率密度函数可以看成三个矩形，这三个矩形面积之和为1。
        # 因此我们可以先随机抽取一个0到1的数，找到对应的矩形，再在该矩形中随机挑选一个数当作用户的扰动数据。

        # 最左侧图形的面积
        p1 = (self.l + self.c) * self.p / np.exp(self.epsilon)
        # 中间图形的面积
        p2 = (self.r - self.l) * self.p

        temp = np.random.uniform()

        if temp < p1:
            # 在最左侧图形中随机选一个
            rs = np.random.uniform(-self.c, self.l)
        elif temp < p1 + p2:
            # 在中间图形随机选一个
            rs = np.random.uniform(self.l, self.r)
        else:
            # 在右侧图形随机选一个
            rs = np.random.uniform(self.r, self.c)

        self.per_data = rs

    def get_per_data(self):
        return self.per_data


class PM_SERVER(object):
    def __init__(self, epsilon: float, per_datalist: list):
        super(PM_SERVER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 所有用户的扰动数据
        self.per_datalist = per_datalist
        # 估计均值结果
        self.es_mean = 0

    def estimate(self):
        # 均值估计的聚合方法很简单，直接将用户数据相加即可
        for x in self.per_datalist:
            self.es_mean += x
        self.es_mean /= len(self.per_datalist)

    def get_es_mean(self):
        return self.es_mean


# 集合数据
class Wheel_USER(object):
    def __init__(self, epsilon: float, domain: list, data_list: list):
        super(Wheel_USER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 用户的原始数据
        self.data_list = data_list
        # 用户手中数据的数目
        self.c = len(data_list)
        # 扰动数据
        self.per_data = 0

    # N是用户数量
    # c是数据域大小,也就是用户手中数据的条数
    # 扰动过程, X为用户的真实数据, 用户输入值的域，epsilon
    def run(self):
        # 为了直接将之前的代码拿过来用，特意设置下列值
        epsilon = self.epsilon
        c = self.c
        X = [self.data_list]
        N = 1
        # 手动抽取用户所用的hash种子
        seed = random.randint(0, 100000)

        max_int_32 = (1 << 32) - 1
        Y = [0 for col in range(N)]
        s = math.exp(epsilon)
        temp_p = 1 / (2 * c - 1 + c * s)
        omega = c * temp_p * s + (1 - c * temp_p)
        for i in range(N):
            V = [0 for col in range(c)]
            # hash
            for j in range(c):
                V[j] = xxhash.xxh32_intdigest(str(X[i][j]), seed=seed) / max_int_32
            # 区间合并的准备工作
            # 详见论文算法3
            bSize = math.ceil(1 / temp_p)
            lef = [0 for col in range(bSize)]
            rig = [0 for col in range(bSize)]
            for b in range(bSize):
                lef[b] = min((b + 1) * temp_p, 1.0)
                rig[b] = b * temp_p
            # 算法3 第6行开始
            for v in V:
                temp_b = math.ceil(v / temp_p) - 1
                lef[temp_b] = min(v, lef[temp_b])
                if temp_b < math.ceil(1 / temp_p) - 1:
                    rig[temp_b + 1] = max(v + temp_p, rig[temp_b + 1])
                else:
                    rig[0] = max(v + temp_p - 1, rig[0])
            temp_rig0 = rig[0]
            for b in range(bSize - 1):
                lef[b] = max(lef[b], rig[b])
                rig[b] = rig[b + 1]
            lef[bSize - 1] = max(lef[bSize - 1], rig[bSize - 1])
            rig[bSize - 1] = temp_rig0 + 1.0
            # 算法3 21行结束
            # ll为总长度
            ll = 0.0
            for b in range(bSize):
                ll = ll + rig[b] - lef[b]
            r = np.random.random_sample()
            a = 0.0
            for b in range(bSize):
                a = a + s * (rig[b] - lef[b]) / omega
                if a > r:
                    z = rig[b] - (a - r) * omega / s
                    break
                a = a + (omega - ll * s) * (lef[(b + 1) % round(bSize)] + math.floor((b + 1) * temp_p) - rig[b]) / (
                        (1 - ll) * omega)
                if a > r:
                    z = lef[(b + 1) % bSize] - (a - r) * (1 - ll) * omega / (omega - ll * s)
                    break
            z = z % 1.0
            Y[i] = z
        self.per_data = [seed, Y[0]]

    def get_per_data(self):
        return self.per_data


class Wheel_SERVER(object):
    def __init__(self, epsilon: float, domain: list, per_datalist: list, c: int):
        super(Wheel_SERVER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 用户数量
        self.n = len(per_datalist)
        # 用户手中数据数目
        self.c = c
        # 频率估计结果
        self.es_data = []

        # 将用户扰动数据和用户种子拆分
        # 用户扰动数据集合
        self.per_datalist = []
        self.seed = []
        for x in per_datalist:
            self.seed.append(x[0])
            self.per_datalist.append(x[1])

    # Y是所有用户的扰动数据，N是行，c是列，D是原始数据域
    def estimate(self):
        Y = self.per_datalist
        N = self.n
        c = self.c
        epsilon = self.epsilon
        D = self.domain

        max_int_32 = (1 << 32) - 1
        k = len(D)
        # Estimate_Dist = [0] * k
        Estimate_Dist = [0 for col in range(k)]
        # Estimate_Dist = np.zeros(k, dtype=float)
        s = math.exp(epsilon)
        temp_p = 1 / (2 * c - 1 + c * s)
        for i in range(N):
            z = Y[i]
            for j in range(k):
                x = D[j]
                v = xxhash.xxh32_intdigest(str(x), seed=self.seed[i]) / max_int_32
                if z - temp_p < v <= z or z - temp_p + 1 < v < 1:
                    Estimate_Dist[j] += 1
        # 矫正过程
        pt = temp_p * s / (c * temp_p * s + (1 - c * temp_p))
        pf = temp_p
        for i in range(k):
            Estimate_Dist[i] = 1 / N * (Estimate_Dist[i] - N * pf) / (pt - pf)
        self.es_data = Estimate_Dist

    def get_es_data(self):
        return self.es_data
