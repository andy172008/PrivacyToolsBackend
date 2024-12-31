# -*- coding: utf-8 -*-
# @Time : 2022/5/11 10:09 上午
# @Author : 贺星宇
# @File : LDPtool.py
# @Software: PyCharm

import copy
import math
import random
from collections import Counter
from random import choice

import numpy as np
import xxhash
from scipy import stats
from scipy.stats import gamma


# TODO 待添加：位置
class Data(object):
    def __init__(self, file_address: str, data_type: str):
        # 储存文件地址
        self.file_address = ''
        # 储存所有数据
        # 数据集文件中每行代表一个用户，如果一行只有一个数据，那么self.data的格式为[用户1,用户2,...,用户n]
        # 如果一行有多个数据，那么self.data的格式为[[用户1的数据1，用户1的数据2，用户1的数据3],[用户2的数据1，用户2的数据2，用户2的数据3],...,[用户3的数据1，用户3的数据2，用户3的数据3]]
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
        # 记录用户每个集合中有多少个元素
        self.set_size = 1

        # 对于键值对，我们专门设计两个变量储存真实频率和真实
        self.true_key_p = []
        self.true_value_mean = []

        self.file_address = file_address

        if data_type == 'categorical' or data_type == 'order':
            self.read_categorical_dataset()
        elif data_type == 'numerical':
            self.read_numeric_dataset()
        elif data_type == 'set':
            self.read_set_dataset()
        elif data_type == 'key-value':
            self.read_key_value_dataset()
        else:
            print('读取数据集时输入的数据类型参数不对')

        if data_type == 'categorical' or data_type == 'set' or data_type == 'order':
            self.data_statistics_p()
        elif data_type == 'numeric':
            self.data_statistics_mean()
        elif data_type == 'key-value':
            self.data_statistics_key_value()

    # 读取类别型数据集，将其添加到self.data
    def read_categorical_dataset(self):
        with open(self.file_address, 'r') as f:
            for line in f.readlines():
                # 移除头尾换行符
                line = line.strip()
                self.data.append(int(float(line)))

    # 读取数值型数据集，将其添加到self.data
    def read_numeric_dataset(self):
        with open(self.file_address, 'r') as f:
            for line in f.readlines():
                # 移除头尾换行符
                line = line.strip()
                self.data.append(float(line))

    # 读取集合型数据集，将其添加到self.data
    def read_set_dataset(self):
        with open(self.file_address, 'r') as f:
            for line in f.readlines():
                # 移除头尾换行符
                line = line.strip()
                # 将一行中的数字划分开
                line_list = line.split(' ')
                line_list = [int(float(x)) for x in line_list]
                self.data.append(line_list)
        # 记录每个集合中的元素数量
        self.set_size = len(self.data[0])

    def read_key_value_dataset(self):
        with open(self.file_address, 'r') as f:
            for line in f.readlines():
                # 移除头尾换行符
                line = line.strip()
                # 将一行中的数字划分开
                line_list = line.split(' ')
                one_user_data = []
                for i in range(len(line_list)):
                    if i % 2 == 0:
                        key = int(float(line_list[i]))
                        value = float(line_list[i + 1])
                        one_user_data.append([key, value])
                self.data.append(one_user_data)
        self.set_size = len(self.data[0])

    # TODO
    def read_location_dataset(self):
        pass

    # 对self，data中的数据统计真实频率
    def data_statistics_p(self):
        temp_data = []
        if self.data_type == 'categorical' or self.data_type == 'order':
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
            if self.data_type == 'categorical' or self.data_type == 'order':
                self.true_p.append(count_dict.get(x, 0) / self.dataNum)
            elif self.data_type == 'set':
                self.true_p.append(count_dict.get(x, 0) / (self.dataNum * self.set_size))

    # 对self.data中的数据统计真实均值
    def data_statistics_mean(self):
        for i in self.data:
            self.true_mean += i
        self.dataNum = len(self.data)
        self.true_mean /= self.dataNum

    # 对self.data中的键值对分别统计键的频率和值的均值
    def data_statistics_key_value(self):
        # 所有键的出现次数
        key_count = {}
        # 每一个键的值的和
        key_sum = {}

        for user_data in self.data:
            for key, value in user_data:
                if key not in key_count:
                    key_count[key] = 0
                    key_sum[key] = 0
                key_count[key] += 1
                key_sum[key] += value

        # 求每个键对应值的均值
        key_mean = {key: key_sum[key] / key_count[key] for key in key_count}
        for key in key_count:
            self.domain.append(key)
        self.domain.sort()
        self.dataNum = len(self.data)

        total_count = sum(key_count.values())
        for x in self.domain:
            self.true_key_p.append(key_count.get(x, 0) / total_count)

            self.true_value_mean.append(key_mean.get(x, 0))

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
        hash_value = (xxhash.xxh32(str(v), seed=seed).intdigest() % self.g)
        return seed, hash_value

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
        Y = [0 for _ in range(N)]
        s = math.exp(epsilon)
        temp_p = 1 / (2 * c - 1 + c * s)
        omega = c * temp_p * s + (1 - c * temp_p)
        for i in range(N):
            V = [0 for _ in range(c)]
            # hash
            for j in range(c):
                V[j] = xxhash.xxh32_intdigest(str(X[i][j]), seed=seed) / max_int_32
            # 区间合并的准备工作
            # 详见论文算法3
            bSize = math.ceil(1 / temp_p)
            lef = [0 for _ in range(bSize)]
            rig = [0 for _ in range(bSize)]
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
        Estimate_Dist = [0 for _ in range(k)]
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
            # Estimate_Dist[i] = 1 / N * (Estimate_Dist[i] - N * pf) / (pt - pf)
            # 上面的代码貌似少除以了了一个集合长度，因此我手动添加一个，应该是正确的
            Estimate_Dist[i] = 1 / (N * c) * (Estimate_Dist[i] - N * pf) / (pt - pf)
        self.es_data = Estimate_Dist

    def get_es_data(self):
        return self.es_data


# ------------------------下方为唐聪新加入代码，不一定正确
class PrivSet_USER(object):
    def __init__(self, epsilon: float, domain: list, data_list: list):
        super(PrivSet_USER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 用户的原始数据
        self.data_list = data_list
        # 输入集合大小
        self.m = len(data_list)
        # 数据域大小
        self.d = len(domain)
        # 扰动数据
        self.per_data = []

    def run(self):
        P = self.perturb_probability()
        perturbed_data = self.perturbation(P)
        self.per_data = perturbed_data

    # 计算扰动概率
    def perturb_probability(self):
        d = self.d
        m = self.m
        eps = self.epsilon
        omega = math.comb(d - m, m) + (math.comb(d, m) - math.comb(d - m, m)) * pow(math.e, eps)
        P = []
        P.append(math.comb(d - m, m) / omega)
        for i in range(1, m + 1):
            prob = P[i - 1] + math.comb(m, i) * math.comb(d - m, m - i) * pow(math.e, eps) / omega
            P.append(prob)
        return P

    # 确认随机数在数组中的位置
    def find_position(self, P, r):
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

    # 在任意一个集合中随机抽样函数
    def random_subset(self, original_set, n):
        if n > len(original_set):
            raise ValueError("n is larger than the set size")
        sampled_list = random.sample(list(original_set), n)  # 将集合转换为列表，并从中随机抽取n个元素
        new_set = set(sampled_list)  # 将列表转换成集合
        return new_set

    # 扰动
    def perturbation(self, P):
        d = self.d
        m = self.m
        d_set = set(range(d))
        set_2 = d_set - self.data_list  # list相减是什么意思
        # print(set_2)
        random_number = np.random.rand()
        # print(random_number)
        c = self.find_position(P, random_number)
        # print(c)
        new_set_1 = self.random_subset(self.data_list, c)
        # print(new_set_1)
        new_set_2 = self.random_subset(set_2, m - c)
        # print(new_set_2)
        set_output = new_set_1 | new_set_2
        return set_output


class PrivSet_SERVER(object):
    def __init__(self, epsilon: float, domain: list, per_datalist: list, m: int):
        super(PrivSet_SERVER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 用户数量
        self.n = len(per_datalist)
        # 集合大小
        self.m = m
        # 数据域大小
        self.d = len(domain)
        # 频率估计结果
        self.es_data = []

        # 用户扰动数据集合
        self.per_datalist = per_datalist

    def estimate(self):
        perturbed_data = self.per_datalist
        n = self.n
        eps = self.epsilon
        d = self.d
        m = self.m
        threshold = d
        pt, pf = self.perturb_probability()
        estimated_counts = []
        estimated_distribution = []
        # 扰动数据的频数
        perturb_counts = self.count_elements_below_threshold(perturbed_data, threshold)
        # 真实数据频数的估计值
        for counts in perturb_counts:
            estimated_counts.append((counts - n * pf) / (pt - pf))
        # 真实数据的估计频率
        for counts in estimated_counts:
            estimated_distribution.append(counts / n)
        self.es_data = estimated_distribution

    def get_es_data(self):
        return self.es_data

    # 计算扰动概率
    def perturb_probability(self):
        d = self.d
        m = self.m
        eps = self.epsilon
        omega = math.comb(d - m, m) + (math.comb(d, m) - math.comb(d - m, m)) * pow(math.e, eps)
        pt = (math.comb(d - 1, m - 1) * pow(math.e, eps)) / omega
        pf = math.comb(d - m - 1, m - 1) / omega + (math.comb(d - 1, m - 1) - math.comb(d - m - 1, m - 1)) * pow(math.e,
                                                                                                                 eps) / omega
        P = []
        P.append(math.comb(d - m, m) / omega)
        for i in range(1, m + 1):
            prob = P[i - 1] + math.comb(m, i) * math.comb(d - m, m - i) * pow(math.e, eps) / omega
            P.append(prob)
        return pt, pf

    # 计数
    def count_elements_below_threshold(self, arr, threshold):
        # 生成一个新数组，其长度为threshold，每个位置初始化为0
        counts = [0] * threshold

        # 遍历数组中的每个元素
        for num in arr:
            # 如果元素小于阈值，则增加相应位置的计数
            if num < threshold:
                counts[num] += 1

        return counts

    # 分布正则化归一化
    def project_probability_simplex(self, p_estimate):
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


# 位置数据
class PL_USER(object):
    def __init__(self, epsilon: float, domain: list, data: list):
        super(PL_USER, self).__init__()
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 用户的原始数据
        self.data = data
        # 数据维度
        self.d = len(data)
        # 扰动数据
        self.per_data = []

    def run(self):
        perturbed_data = self.Laplace_pertubation()
        self.per_data = perturbed_data

    # 生成二维随机单位向量
    def get_random_unit_vector(self):
        # 从标准正态分布中抽取两个独立的样本
        vector = np.random.randn(2)
        # 规范化向量为单位长度
        unit_vector = vector / np.linalg.norm(vector)
        return unit_vector

    # gamma随机数
    def gamma_random_num(self):
        d = self.d
        eps = self.epsilon
        scale = 1 / eps
        l = gamma.rvs(a=d, scale=scale)

        return l

    # 生成二维拉普拉斯噪声
    def Laplace_noise(self):
        d = self.d
        eps = self.epsilon
        unit_vector = self.get_random_unit_vector()
        l = self.gamma_random_num()
        Laplace_noise = l * unit_vector

        return Laplace_noise

    # 实际数据加上噪声
    def Laplace_pertubation(self):
        d = self.d
        eps = self.epsilon
        v = self.data
        noise = self.Laplace_noise()
        new_vector = v + noise

        return new_vector


class PL_SERVER(object):
    def __init__(self, epsilon: float, domain: list, per_datalist: list):
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 所有用户的扰动数据
        self.per_datalist = per_datalist
        # 用户数量
        self.n = len(per_datalist)
        # 频率估计结果
        self.es_data = []

        # 固定参数
        self.d = 50
        self.r = 1 / 10

    def estimate(self):
        min_x, min_y, max_x, max_y = self.find_min_max_coordinates()
        rectangle_points = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
        square_size, squares_centers, squares_nums, num_cols, num_rows = self.divide_rectangle_into_squares(
            rectangle_points)
        rectangles_coordinates = self.plot_divided_grid_with_complete_coverage(squares_centers, square_size)
        new_proportion = self.count_points_in_rectangles(rectangles_coordinates)
        self.es_data = new_proportion

    def get_es_data(self):
        return self.es_data

        # 确认输入域边界

    def find_min_max_coordinates(self):
        coordinates = self.domain
        if not coordinates:
            return None  # 如果列表为空，返回 None

        # 初始化最小和最大坐标
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        # 遍历每个坐标
        for coord in coordinates:
            x, y = coord
            # 更新最小和最大值
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

        return min_x, min_y, max_x, max_y

    # 划分区域
    # 长方形分成正方形
    def divide_rectangle_into_squares(self, rectangle):
        num_squares = self.d
        # 计算矩形的宽度和高度
        length = abs(rectangle[2][0] - rectangle[0][0])
        width = abs(rectangle[1][1] - rectangle[0][1])

        num_rows = math.ceil(math.sqrt(num_squares * width / length))
        num_cols = math.ceil(math.sqrt(num_squares * length / width))

        # print(num_cols, num_rows)

        square_size = width / math.sqrt(num_squares * width / length)

        # 生成正方形中心点坐标
        squares_centers = []
        for row in range(num_rows):
            for col in range(num_cols):
                center_x = rectangle[0][0] + (col + 0.5) * square_size
                center_y = rectangle[0][1] + (row + 0.5) * square_size

                squares_centers.append((center_x, center_y))

        return square_size, squares_centers, len(squares_centers), num_cols, num_rows

    def find_factors_closest_to_sqrt(self, n):
        for i in range(int(n ** 0.5), 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n

    def plot_divided_grid_with_complete_coverage(self, grid_centers, grid_length):
        n = int(1 / self.r)
        # Calculate the bounding rectangle
        x_coords = [x for x, _ in grid_centers]
        y_coords = [y for _, y in grid_centers]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        rect_width = max_x - min_x + grid_length
        rect_height = max_y - min_y + grid_length

        # Find the division factors closest to the square root of n
        regions_across, regions_down = self.find_factors_closest_to_sqrt(n)
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

    # 计算大区域内点比例
    def count_points_in_rectangles(self, rectangles):
        points = self.per_datalist
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


class GBCUG_USER(object):
    def __init__(self, epsilon: float, domain: list, data: list):
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 用户的原始数据
        self.data = data
        # 数据维度
        self.d = len(data)
        # 扰动数据
        self.per_data = []

    def run(self):
        perturbed_data = self.generate_points_with_density()
        self.per_data = perturbed_data

    # 圆内生成随机点
    def generate_points_with_density(self):
        epsilon = self.epsilon
        coordinates = self.data
        points = []
        radius = np.random.gamma(2 + 1, 1 / epsilon, size=1)[0]
        angle = random.uniform(0, 2 * math.pi)
        r = radius * math.sqrt(random.uniform(0, 1))
        x = coordinates[0] + r * math.cos(angle)
        y = coordinates[1] + r * math.sin(angle)
        points.append((x, y))

        return points


class GBCUG_SERVER(object):
    def __init__(self, epsilon: float, domain: list, per_datalist: list):
        # 隐私预算
        self.epsilon = epsilon
        # 原始数据定义域
        self.domain = domain
        # 所有用户的扰动数据
        self.per_datalist = per_datalist
        # 用户数量
        self.n = len(per_datalist)
        # 频率估计结果
        self.es_data = []

        # 固定参数
        self.d = 50
        self.r = 1 / 10

    def estimate(self):
        min_x, min_y, max_x, max_y = self.find_min_max_coordinates()
        rectangle_points = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]
        square_size, squares_centers, squares_nums, num_cols, num_rows = self.divide_rectangle_into_squares(
            rectangle_points)
        rectangles_coordinates = self.plot_divided_grid_with_complete_coverage(squares_centers, square_size)
        new_proportion = self.count_points_in_rectangles(rectangles_coordinates)
        self.es_data = new_proportion

    def get_es_data(self):
        return self.es_data

        # 确认输入域边界

    def find_min_max_coordinates(self):
        coordinates = self.domain
        if not coordinates:
            return None  # 如果列表为空，返回 None

        # 初始化最小和最大坐标
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        # 遍历每个坐标
        for coord in coordinates:
            x, y = coord
            # 更新最小和最大值
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y

        return min_x, min_y, max_x, max_y

    # 划分区域
    # 长方形分成正方形
    def divide_rectangle_into_squares(self, rectangle):
        num_squares = self.d
        # 计算矩形的宽度和高度
        length = abs(rectangle[2][0] - rectangle[0][0])
        width = abs(rectangle[1][1] - rectangle[0][1])

        num_rows = math.ceil(math.sqrt(num_squares * width / length))
        num_cols = math.ceil(math.sqrt(num_squares * length / width))

        # print(num_cols, num_rows)

        square_size = width / math.sqrt(num_squares * width / length)

        # 生成正方形中心点坐标
        squares_centers = []
        for row in range(num_rows):
            for col in range(num_cols):
                center_x = rectangle[0][0] + (col + 0.5) * square_size
                center_y = rectangle[0][1] + (row + 0.5) * square_size

                squares_centers.append((center_x, center_y))

        return square_size, squares_centers, len(squares_centers), num_cols, num_rows

    def find_factors_closest_to_sqrt(self, n):
        for i in range(int(n ** 0.5), 0, -1):
            if n % i == 0:
                return i, n // i
        return 1, n

    def plot_divided_grid_with_complete_coverage(self, grid_centers, grid_length):
        n = int(1 / self.r)
        # Calculate the bounding rectangle
        x_coords = [x for x, _ in grid_centers]
        y_coords = [y for _, y in grid_centers]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        rect_width = max_x - min_x + grid_length
        rect_height = max_y - min_y + grid_length

        # Find the division factors closest to the square root of n
        regions_across, regions_down = self.find_factors_closest_to_sqrt(n)
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

    # 计算大区域内点比例
    def count_points_in_rectangles(self, rectangles):
        points = self.per_datalist
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


# 有序数据
class EM_USER(object):
    # U/domain：数据域
    # alpha：约等于隐私预算
    # v： 单个用户的真实值
    def __init__(self, epsilon, domain, x):
        self.U = domain
        self.alpha = self.convert(domain, epsilon)
        self.v = x
        # 扰动数据
        self.per_data = []

    def get_per_data(self):
        return self.per_data

    # 将epsilon转为alpha
    def convert(self, U, epsilon):
        alpha = 0
        ldp_matrix = self.perturb_ldp(epsilon, U)
        prior = [1 / len(U) for i in range(len(U))]  # 均匀分布
        mpc_ldp = self.MPC(U, ldp_matrix, prior)
        mpc_cldp = 0
        while mpc_cldp <= mpc_ldp:
            alpha += 0.01
            cldp_matrix = self.perturb_cldp(alpha, U)
            mpc_cldp = self.MPC(U, cldp_matrix, prior)
        return alpha - 0.01

    def MPC(self, U, matrix, prior):
        mpc = 0
        for j, y in enumerate(U):
            max_pro = 0
            sum = 0
            for i, x in enumerate(U):
                temp = matrix[i][j] * prior[i]
                sum += temp
                if temp > max_pro:
                    max_pro = temp
            if max_pro / sum > mpc:
                mpc = max_pro / sum
        return mpc

    def perturb_cldp(self, alpha, U):
        matrix = []
        for x in U:
            temp = []
            score_sum = 0
            for y in U:
                score = math.pow(math.e, -1 * alpha * self.d(x, y, U) * 0.5)
                score_sum = score_sum + score
                temp.append(score)
            temp = [i / score_sum for i in temp]
            matrix.append(temp)
        return matrix

    def d(self, x, y, U):
        X = list(U)
        # return math.fabs(X.index(x) - X.index(y))
        return math.fabs(x - y)

    def perturb_ldp(self, epsilon, U):
        matrix = []
        k = len(U)
        e_epsilon = math.pow(math.e, epsilon)
        for x in range(len(U)):
            temp = [1 / (e_epsilon + k - 1) for i in range(len(U))]
            temp[x] = e_epsilon / (e_epsilon + k - 1)
            matrix.append(temp)
        return matrix

    def run(self):
        perturbed_data = self.perturb()
        self.per_data = perturbed_data

    def dis(self, x, y):
        # return math.fabs(self.U.index(x) - self.U.index(y))
        return math.fabs(x - y)

    def perturb(self):
        v = self.v
        score_sum = 0
        probabilities = [score_sum]
        for z in self.U:
            score = math.pow(math.e, -1 * self.alpha * self.dis(v, z) * 0.5)
            score_sum = score_sum + score
            probabilities.append(score_sum)
        r = np.random.uniform(0, score_sum)
        l = len(probabilities)
        y = self.U[0]
        for i in range(l):
            if r >= probabilities[l - i - 1]:
                y = self.U[l - i - 1]
                break
        return y


class EM_SERVER(object):
    # per_datalist: 单值集合
    def __init__(self, epsilon, U, per_datalist):
        self.U = U
        self.alpha = self.convert(U, epsilon)
        self.est = []
        self.per_datalist = per_datalist
        # 频率估计结果
        self.es_data = []

    def get_es_data(self):
        return self.es_data

    # 将epsilon转为alpha
    def convert(self, U, epsilon):
        alpha = 0
        ldp_matrix = self.perturb_ldp(epsilon, U)
        prior = [1 / len(U) for i in range(len(U))]  # 均匀分布
        mpc_ldp = self.MPC(U, ldp_matrix, prior)
        mpc_cldp = 0
        while mpc_cldp <= mpc_ldp:
            alpha += 0.01
            cldp_matrix = self.perturb_cldp(alpha, U)
            mpc_cldp = self.MPC(U, cldp_matrix, prior)
        return alpha - 0.01

    def MPC(self, U, matrix, prior):
        mpc = 0
        for j, y in enumerate(U):
            max_pro = 0
            sum = 0
            for i, x in enumerate(U):
                temp = matrix[i][j] * prior[i]
                sum += temp
                if temp > max_pro:
                    max_pro = temp
            if max_pro / sum > mpc:
                mpc = max_pro / sum
        return mpc

    def perturb_cldp(self, alpha, U):
        matrix = []
        for x in U:
            temp = []
            score_sum = 0
            for y in U:
                score = math.pow(math.e, -1 * alpha * self.d(x, y, U) * 0.5)
                score_sum = score_sum + score
                temp.append(score)
            temp = [i / score_sum for i in temp]
            matrix.append(temp)
        return matrix

    def d(self, x, y, U):
        X = list(U)
        # return math.fabs(X.index(x) - X.index(y))
        return math.fabs(x - y)

    def perturb_ldp(self, epsilon, U):
        matrix = []
        k = len(U)
        e_epsilon = math.pow(math.e, epsilon)
        for x in range(len(U)):
            temp = [1 / (e_epsilon + k - 1) for i in range(len(U))]
            temp[x] = e_epsilon / (e_epsilon + k - 1)
            matrix.append(temp)
        return matrix

    def estimate(self):
        new_proportion = self.aggregate()

        self.es_data = new_proportion

    def dis(self, x, y):
        # return math.fabs(self.U.index(x) - self.U.index(y))
        return math.fabs(x - y)

    def stochastic_matrix(self):
        matrix = []
        for x in self.U:
            temp = []
            score_sum = 0
            for y in self.U:
                score = math.pow(math.e, -1 * self.alpha * self.dis(x, y) * 0.5)
                score_sum = score_sum + score
                temp.append(score)
            temp = [i / score_sum for i in temp]
            matrix.append(temp)
        return matrix

    def aggregate(self):  # 聚合时，以观测值代替真实频数
        obs = [0 for i in range(len(self.U))]
        est = [0 for i in range(len(self.U))]
        n = len(self.per_datalist)
        matrix = self.stochastic_matrix()
        for v in self.per_datalist:
            obs[self.U.index(v)] += 1
        for y in range(len(obs)):
            t = 0
            for x in range(len(obs)):
                if x == y:
                    continue
                t += obs[x] * matrix[x][y]
            est[y] = round((obs[y] - t) / matrix[y][y])
            if est[y] < 0:
                est[y] = 0
        sum_value = 0
        for i in est:
            sum_value += i
        est = [i / sum_value for i in est]
        self.est = est
        return est


class GEM_USER(object):
    # U 数据域
    # d 扰动输出的集合大小
    # alpha：约等于隐私预算
    # v： 单个用户的真实值
    def __init__(self, epsilon, U, v):
        self.alpha = self.convert(U, epsilon)
        self.U = U
        self.v = v
        self.k = len(U)
        self.d = self.findOptd(self.alpha, self.k)
        self.r = []
        self.P = []
        self.S = []  # 扰动输出结果, 大小为d的集合
        # 扰动数据
        self.per_data = []

    def get_per_data(self):
        return self.per_data

    # 找到最优的d
    def findOptd(self, alpha, k):
        max = self.fun(alpha, k, 1)
        opt_d = 1
        for d in range(2, k):
            cur = self.fun(alpha, k, d)
            if cur > max:
                max = cur
                opt_d = d
        return opt_d

    def fun(self, alpha, k, d):
        c = math.factorial(k) / (math.factorial(d) * math.factorial(k - d))
        res = math.log(c, math.e) - (
                alpha * (k - d) / 4 + 2 * math.log(c / 2) + 2 * math.log(1 + math.pow(math.e, -alpha * (k - d) / 4),
                                                                         math.e)) \
              / (1 + math.pow(math.e, alpha * (k - d) / 2))
        return res

    # 将epsilon转为alpha
    def convert(self, U, epsilon):
        alpha = 0
        ldp_matrix = self.perturb_ldp(epsilon, U)
        prior = [1 / len(U) for i in range(len(U))]  # 均匀分布
        mpc_ldp = self.MPC(U, ldp_matrix, prior)
        mpc_cldp = 0
        while mpc_cldp <= mpc_ldp:
            alpha += 0.01
            cldp_matrix = self.perturb_cldp(alpha, U)
            mpc_cldp = self.MPC(U, cldp_matrix, prior)
        return alpha - 0.01

    def MPC(self, U, matrix, prior):
        mpc = 0
        for j, y in enumerate(U):
            max_pro = 0
            sum = 0
            for i, x in enumerate(U):
                temp = matrix[i][j] * prior[i]
                sum += temp
                if temp > max_pro:
                    max_pro = temp
            if max_pro / sum > mpc:
                mpc = max_pro / sum
        return mpc

    def perturb_cldp(self, alpha, U):
        matrix = []
        for x in U:
            temp = []
            score_sum = 0
            for y in U:
                score = math.pow(math.e, -1 * alpha * self.d(x, y, U) * 0.5)
                score_sum = score_sum + score
                temp.append(score)
            temp = [i / score_sum for i in temp]
            matrix.append(temp)
        return matrix

    def d(self, x, y, U):
        X = list(U)
        # return math.fabs(X.index(x) - X.index(y))
        return math.fabs(x - y)

    def perturb_ldp(self, epsilon, U):
        matrix = []
        k = len(U)
        e_epsilon = math.pow(math.e, epsilon)
        for x in range(len(U)):
            temp = [1 / (e_epsilon + k - 1) for i in range(len(U))]
            temp[x] = e_epsilon / (e_epsilon + k - 1)
            matrix.append(temp)
        return matrix

    def run(self):
        perturbed_data = self.perturb()
        self.per_data = perturbed_data

    def dis(self, x, y):
        return math.fabs(x - y)

    def select(self, i, d, temp_U):
        if d == 0:
            return self.S
        p = self.r[i] * self.P[d - 1][i + 1] / self.P[d][i]
        r = random.random()
        if r < p:
            self.S.append(temp_U[i])
            self.select(i + 1, d - 1, temp_U)
        else:
            self.select(i + 1, d, temp_U)

    def solveP(self):
        for t in range(self.d + 1):
            if t == 0:
                temp = [1 for j in range(self.k + 1)]
                self.P.append(temp)
            else:
                temp = [0 for j in range(self.k - t + 1)]
                temp[self.k - t] = self.r[self.k - t] * self.P[t - 1][self.k - t + 1]
                for i in range(self.k - t):
                    temp[self.k - 1 - t - i] = temp[self.k - t - i] + self.r[self.k - 1 - t - i] * self.P[t - 1][
                        self.k - t - i]
                self.P.append(temp)

    def perturb(self):
        v = self.v
        self.S = []
        self.r = []
        self.P = []
        temp_U = copy.deepcopy(self.U)
        temp_U = sorted(temp_U, key=lambda x: self.dis(x, v))  # 定义域按离真实值距离重排序
        temp_U = list(temp_U)
        for index, item in enumerate(temp_U):
            self.r.append(math.pow(math.e, -self.alpha * self.dis(item, v) / (2 * self.d)))
        self.solveP()
        d = self.d
        self.select(0, d, temp_U)
        # print(len(self.S))
        # print(self.S)
        return self.S


class GEM_SERVER(object):
    def __init__(self, epsilon, U: list, per_datalist):
        self.per_datalist = per_datalist
        self.alpha = self.convert(U, epsilon)
        self.U = U
        self.k = len(U)
        self.d = self.findOptd(self.alpha, self.k)
        self.random_matrix = []
        self.obs = []
        self.total_perturb = []  # 所有子集的和
        # 频率估计结果
        self.es_data = []

    def get_es_data(self):
        return self.es_data

    # 找到最优的d
    def findOptd(self, alpha, k):
        max = self.fun(alpha, k, 1)
        opt_d = 1
        for d in range(2, k):
            cur = self.fun(alpha, k, d)
            if cur > max:
                max = cur
                opt_d = d
        return opt_d

    def fun(self, alpha, k, d):
        c = math.factorial(k) / (math.factorial(d) * math.factorial(k - d))
        res = math.log(c, math.e) - (
                alpha * (k - d) / 4 + 2 * math.log(c / 2) + 2 * math.log(1 + math.pow(math.e, -alpha * (k - d) / 4),
                                                                         math.e)) \
              / (1 + math.pow(math.e, alpha * (k - d) / 2))
        return res

    # 将epsilon转为alpha
    def convert(self, U, epsilon):
        alpha = 0
        ldp_matrix = self.perturb_ldp(epsilon, U)
        prior = [1 / len(U) for i in range(len(U))]  # 均匀分布
        mpc_ldp = self.MPC(U, ldp_matrix, prior)
        mpc_cldp = 0
        while mpc_cldp <= mpc_ldp:
            alpha += 0.01
            cldp_matrix = self.perturb_cldp(alpha, U)
            mpc_cldp = self.MPC(U, cldp_matrix, prior)
        return alpha - 0.01

    def MPC(self, U, matrix, prior):
        mpc = 0
        for j, y in enumerate(U):
            max_pro = 0
            sum = 0
            for i, x in enumerate(U):
                temp = matrix[i][j] * prior[i]
                sum += temp
                if temp > max_pro:
                    max_pro = temp
            if max_pro / sum > mpc:
                mpc = max_pro / sum
        return mpc

    def perturb_cldp(self, alpha, U):
        matrix = []
        for x in U:
            temp = []
            score_sum = 0
            for y in U:
                score = math.pow(math.e, -1 * alpha * self.d(x, y, U) * 0.5)
                score_sum = score_sum + score
                temp.append(score)
            temp = [i / score_sum for i in temp]
            matrix.append(temp)
        return matrix

    def d(self, x, y, U):
        X = list(U)
        # return math.fabs(X.index(x) - X.index(y))
        return math.fabs(x - y)

    def perturb_ldp(self, epsilon, U):
        matrix = []
        k = len(U)
        e_epsilon = math.pow(math.e, epsilon)
        for x in range(len(U)):
            temp = [1 / (e_epsilon + k - 1) for i in range(len(U))]
            temp[x] = e_epsilon / (e_epsilon + k - 1)
            matrix.append(temp)
        return matrix

    def estimate(self):
        new_proportion = self.EM_lognew()

        self.es_data = new_proportion

    def dis(self, x, y):
        return np.fabs(x - y)

    def P_1m(self, r):
        r = np.array(r)
        temp = [1.0 for j in range(self.k)]
        temp1 = np.array([temp for j in range(self.k)])
        for t in range(1, self.d + 1):
            temp0 = copy.deepcopy(temp1)
            if t == 1:
                temp1[:, (self.k - 1)] = np.array(r)[:, (self.k - 1)].T
                for a in range(self.k - 1):
                    temp1[:, self.k - 2 - a] = r[:, self.k - 2 - a] + temp1[:, self.k - 1 - a]
            else:
                temp1[:, self.k - t] = r[:, self.k - t] * temp0[:, self.k - t + 1]
                for a in range(self.k - t):
                    temp1[:, self.k - 1 - t - a] = temp1[:, self.k - t - a] + r[:, self.k - 1 - t - a] * \
                                                   temp0[:, self.k - t - a]
        return r[:, 0] * temp0[:, 1] / temp1[:, 0], temp1[:, 0]

    def solve_r(self, x, v):
        return np.power(math.e, -self.alpha * self.dis(x, v) / (2 * self.d))

    def solve_R(self, U, v):
        return self.solve_r(U, v)

    def switchR(self, r):
        temp_list = []
        for i in range(len(r)):
            tr = copy.deepcopy(r)
            temp = tr[0]
            tr[0] = tr[i]
            tr[i] = temp
            temp_list.append(tr)
        return temp_list

    def solveMatrix_np(self):
        for i in range(self.k):
            v = self.U[i]
            r = self.solve_r(np.array(self.U), v)  # 长为k的向量
            R = self.switchR(r)  # k*k的矩阵
            temp_matrix, temp_perturb = self.P_1m(R)
            self.random_matrix.append(temp_matrix)
            self.total_perturb.append(temp_perturb[0])
        # return random_matrix, total_perturb

    def solveMatrix(self):
        for i in range(self.k):  # 求真实值为i，扰动输出的子集包含j的概率
            v = self.U[i]
            temp_list = []
            for j in range(self.k):
                temp_U = copy.deepcopy(self.U)
                if j != 0:  # j提到第一个
                    temp = temp_U[0]
                    temp_U[0] = temp_U[j]
                    temp_U[j] = temp
                r = []
                for index, item in enumerate(temp_U):
                    r.append(math.pow(math.e, -self.alpha * self.dis(item, v) / (2 * self.d)))
                temp1 = [1 for j in range(self.k)]
                for t in range(1, self.d + 1):
                    temp0 = copy.deepcopy(temp1)
                    if t == 1:
                        temp1[self.k - 1] = r[self.k - 1]
                        for a in range(self.k - 1):
                            temp1[self.k - 2 - a] = r[self.k - 2 - a] + temp1[self.k - 1 - a]
                    else:
                        temp1[self.k - t] = r[self.k - t] * temp0[self.k - t + 1]
                        for a in range(self.k - t):
                            temp1[self.k - 1 - t - a] = temp1[self.k - t - a] + r[self.k - 1 - t - a] * \
                                                        temp0[self.k - t - a]
                temp_list.append(r[0] * temp0[1] / temp1[0])
            self.total_perturb.append(temp1[0])
            self.random_matrix.append(temp_list)

    def count(self):
        self.obs = [0 for i in range(self.k)]
        for item in self.per_datalist:
            for c in item:
                self.obs[self.U.index(c)] += 1
        # self.obs = [i/(len(self.per_datalist)*self.d) for i in self.obs]

    # 一对一
    def log_likelihood_new(self, est):
        matrix_trans = np.array(self.random_matrix).T
        ll = np.inner(self.obs, np.log(np.matmul(matrix_trans, est)))
        return ll

    # 似然函数观测值，输出中各项分布
    def EM_lognew(self):
        thre = 1e-4
        self.random_matrix = []
        self.obs = []
        self.total_perturb = []  # 所有子集的和
        self.count()
        self.solveMatrix_np()
        est = [i / (len(self.per_datalist) * self.d) for i in self.obs]
        ll = self.log_likelihood_new(est)
        delta = math.fabs(ll)
        count = 0
        while delta > thre:
            self.random_matrix = np.array(self.random_matrix)
            dom = np.matmul(self.random_matrix.T, est)
            TMP = self.random_matrix / dom
            p = np.copy(np.matmul(TMP, np.array(self.obs)))
            p = p * est
            est = np.copy(p / sum(p))
            ll_new = self.log_likelihood_new(est)
            delta = ll_new - ll
            ll = ll_new
            count += 1
        return est.tolist()


# 键值数据
class PCKVGRR_USER(object):
    # epsilon 隐私预算
    # S 单条用户集合
    # d key的大小
    def __init__(self, epsilon, S, d):
        self.epsilon = epsilon
        self.S = S
        self.l = 1
        self.d = d
        # 扰动数据
        self.per_data = []

    def run(self):
        perturbed_data = self.perturb()
        self.per_data = perturbed_data

    def pad_sample(self):
        n = len(self.S) / max([len(self.S), self.l])
        bernoulliDist = stats.bernoulli(n)
        B = bernoulliDist.rvs(1)
        if B == 1:
            a = random.choice(self.S)
            kv = copy.copy(a)
        else:
            num = list(range(self.d + 1, self.d + self.l + 1))
            kv = [random.choice(num), 0]

        p = (1 + kv[1]) / 2
        q = 1 - p
        R = random.random()
        if R < p:
            kv[1] = 1
        else:
            kv[1] = -1

        return kv

    def perturb(self):
        x = self.pad_sample()
        e_eps = math.pow(2.71828, self.epsilon)
        epsilon1 = math.log((self.l * (e_eps + 1) / 2) + 1)
        epsilon2 = math.log(self.l * (e_eps - 1) + 1)
        a = (self.l * (e_eps - 1) + 2) / (self.l * (e_eps - 1) + 2 * (self.d + self.l))
        b = (1 - a) / (self.d + self.l - 1)
        p = (self.l * (e_eps - 1) + 1) / (self.l * (e_eps - 1) + 2)

        R = random.random()
        if R < a:
            R = random.random()
            if R < p:
                Z = [x[0], x[1]]
            else:
                Z = [x[0], -x[1]]
        else:
            numbers = list(range(1, self.d + self.l + 1))
            # 移除k
            numbers.remove(x[0])
            # 随机选择一个数
            random_number = random.choice(numbers)
            R = random.random()
            if R <= 0.5:
                Z = [random_number, 1]
            else:
                Z = [random_number, -1]
        return Z

    def get_per_data(self):
        return self.per_data


class PCKVGRR_SERVER(object):
    def __init__(self, eps, Gt, d):
        self.Gt = Gt  # 扰动数据
        self.epsilon = eps
        self.d = d
        self.l = 1
        self.P = [0]  # 1-50有效
        self.F = [0]  # 1-50有效
        self.M = [0]
        # 频率估计结果
        self.es_data = []

    def estimate(self):
        e_eps = math.pow(2.71828, self.epsilon)
        epsilon1 = math.log((self.l * (e_eps + 1) / 2) + 1)
        epsilon2 = math.log(self.l * (e_eps - 1) + 1)
        a = (self.l * (e_eps - 1) + 2) / (self.l * (e_eps - 1) + 2 * (self.d + self.l))
        b = (1 - a) / (self.d + self.l - 1)
        p = (self.l * (e_eps - 1) + 1) / (self.l * (e_eps - 1) + 2)

        n = len(self.Gt)

        for k in range(1, self.d + 1):
            n1 = 0
            n2 = 0
            for Z in self.Gt:
                if Z[0] == k:
                    if Z[1] == 1:
                        n1 = n1 + 1
                    elif Z[1] == -1:
                        n2 = n2 + 1
            fk = (((n1 + n2) / n) - b) * self.l / (a - b)
            if fk < (1 / n):
                fk = 1 / n
            elif fk > 1:
                fk = 1
            #  方法二
            # 定义2×2的矩阵
            matrix_1 = np.array([[a * p - b / 2, a * (1 - p) - b / 2], [a * (1 - p) - b / 2, a * p - b / 2]])
            # 计算矩阵的逆
            inverse_matrix_1 = np.linalg.inv(matrix_1)
            matrix_2 = np.array([[n1 - n * b / 2], [n2 - n * b / 2]])
            result = np.dot(inverse_matrix_1, matrix_2)
            nn1 = result[0][0]
            nn2 = result[1][0]
            if nn1 < 0:
                nn1 = 0
            elif nn1 > (n * fk / self.l):
                nn1 = n * fk / self.l
            if nn2 < 0:
                nn2 = 0
            elif nn2 > (n * fk / self.l):
                nn2 = n * fk / self.l
            mk = (self.l * (nn1 - nn2)) / (n * fk)
            # mk = (self.l*(n1-n2)) / (n*fk*a*(2*p-1))
            self.P.append([fk, mk])
            self.F.append(fk)
            self.M.append(mk)
            self.es_data = self.P

    def get_T_F(self):
        return self.F

    def get_T_M(self):
        return self.M


class PCKVUE_USER(object):
    def __init__(self, epsilon, S, d):
        self.epsilon = epsilon
        self.S = S
        self.l = 1
        self.d = d
        # 扰动数据
        self.per_data = []

    def run(self):
        perturbed_data = self.perturb()
        self.per_data = perturbed_data

    def pad_sample(self):
        n = len(self.S) / max([len(self.S), self.l])
        bernoulliDist = stats.bernoulli(n)
        B = bernoulliDist.rvs(1)
        if B == 1:
            a = random.choice(self.S)
            kv = copy.copy(a)
        else:
            num = list(range(self.d + 1, self.d + self.l + 1))
            kv = [random.choice(num), 0]

        p = (1 + kv[1]) / 2
        q = 1 - p
        R = random.random()
        if R < p:
            kv[1] = 1
        else:
            kv[1] = -1

        return kv

    def perturb(self):
        x = self.pad_sample()
        e_eps = math.pow(2.71828, self.epsilon)
        epsilon1 = math.log((e_eps + 1) / 2)
        epsilon2 = self.epsilon
        a = 1 / 2
        b = 1 / (math.exp(epsilon1) + 1)
        p = math.exp(epsilon2) / (math.exp(epsilon2) + 1)

        X = [[0, 0]]  # 从第1号元素开始，即第0号元素不考虑
        for i in range(1, self.d + self.l + 1):
            if i == x[0]:
                X.append([1, x[1]])
            else:
                X.append([0, 0])

        for i in range(1, self.d + self.l + 1):
            if i == x[0]:
                R = random.random()
                if R < (1 - a):
                    X[i] = [0, 0]
                else:
                    R = random.random()
                    if R < (1 - p):
                        X[i][1] = -X[i][1]
            else:
                R = random.random()
                if R < (1 - b):
                    X[i] = [0, 0]
                else:
                    X[i] = [1, 0]
                    R = random.random()
                    if R < 0.5:
                        X[i][1] = 1
                    else:
                        X[i][1] = -1
        return X

    def get_per_data(self):
        return self.per_data


class PCKVUE_SERVER(object):
    def __init__(self, eps, Gt, d):
        self.Gt = Gt  # 扰动数据
        self.epsilon = eps
        self.d = d
        self.l = 1
        self.P = [0]  # 1-50有效
        self.F = [0]  # 1-50有效
        self.M = [0]
        # 频率估计结果
        self.es_data = []

    def estimate(self):
        e_eps = math.pow(2.71828, self.epsilon)
        epsilon1 = math.log((e_eps + 1) / 2)
        epsilon2 = self.epsilon
        a = 1 / 2
        b = 1 / (math.exp(epsilon1) + 1)
        p = math.exp(epsilon2) / (math.exp(epsilon2) + 1)
        n = len(self.Gt)

        for k in range(1, self.d + 1):
            n1 = 0
            n2 = 0
            for Z in self.Gt:
                if Z[k][1] == 1:
                    n1 = n1 + 1
                elif Z[k][1] == -1:
                    n2 = n2 + 1
            fk = (((n1 + n2) / n) - b) * self.l / (a - b)
            if fk < (1 / n):
                fk = 1 / n
            elif fk > 1:
                fk = 1
            #  方法二
            # 定义2×2的矩阵
            matrix_1 = np.array([[a * p - b / 2, a * (1 - p) - b / 2], [a * (1 - p) - b / 2, a * p - b / 2]])
            # 计算矩阵的逆
            inverse_matrix_1 = np.linalg.inv(matrix_1)
            matrix_2 = np.array([[n1 - n * b / 2], [n2 - n * b / 2]])
            result = np.dot(inverse_matrix_1, matrix_2)
            nn1 = result[0][0]
            nn2 = result[1][0]
            if nn1 < 0:
                nn1 = 0
            elif nn1 > (n * fk / self.l):
                nn1 = n * fk / self.l
            if nn2 < 0:
                nn2 = 0
            elif nn2 > (n * fk / self.l):
                nn2 = n * fk / self.l
            mk = (self.l * (nn1 - nn2)) / (n * fk)
            #  方法一：mk = (self.l*(n1-n2)) / (n*fk*a*(2*p-1))
            self.P.append([fk, mk])
            self.F.append(fk)
            self.M.append(mk)
            self.es_data = self.P

    def get_T_F(self):
        return self.F

    def get_T_M(self):
        return self.M
