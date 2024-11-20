# -*- coding: utf-8 -*-
# @Time : 2022/1/12 3:19 下午
# @Author : 贺星宇
# @File : main.py
# @Software: PyCharm

from LDPtool import *
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import csv

app = FastAPI()
'''中间件跨域'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 类别型单个机制
@app.post("/categorical_one")
async def process_categorical_one(dataset: UploadFile = File(...), epsilon_low: float = Form(...),
                                  epsilon_high: float = Form(...), mechanism: str = Form(...)):
    contents = await dataset.read()
    # 解码并按行分割
    lines = contents.decode().splitlines()
    # 去除每行的首尾空格，并过滤掉空行
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    # 将前端传来的文件储存到本地
    file_path = "./dataset/" + dataset.filename
    with open(file_path, 'w') as f:
        for line in cleaned_lines:
            f.write(f"{line}\n")
    # 根据不同的机制参数，执行不同函数

    domain, true_frequency, estimated_frequency, mse = categorical_mechanisms(file_path, epsilon_low, epsilon_high,
                                                                              mechanism)
    step = 0.2
    num = int(round((epsilon_high - epsilon_low) / step)) + 1
    epsilon_list = list(np.linspace(epsilon_low, epsilon_high, num=num))
    for i in range(len(epsilon_list)):
        epsilon_list[i] = round(epsilon_list[i], 1)

    return {
        "domain": domain,
        "epsilon_list": epsilon_list,
        "true": true_frequency,
        "estimated": estimated_frequency,
        "mse": mse,
    }


@app.post("/categorical_two")
async def process_categorical_two(dataset: UploadFile = File(...), epsilon_low: float = Form(...),
                                  epsilon_high: float = Form(...), mechanism1: str = Form(...),
                                  mechanism2: str = Form(...)):
    contents = await dataset.read()
    # 解码并按行分割
    lines = contents.decode().splitlines()
    # 去除每行的首尾空格，并过滤掉空行
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    # 将前端传来的文件储存到本地
    file_path = "./dataset/" + dataset.filename
    with open(file_path, 'w') as f:
        for line in cleaned_lines:
            f.write(f"{line}\n")

    domain, true_frequency, estimated_frequency1, mse1 = categorical_mechanisms(file_path, epsilon_low, epsilon_high,
                                                                                mechanism1)
    domain, true_frequency, estimated_frequency2, mse2 = categorical_mechanisms(file_path, epsilon_low, epsilon_high,
                                                                                mechanism2)
    step = 0.2
    num = int(round((epsilon_high - epsilon_low) / step)) + 1
    epsilon_list = list(np.linspace(epsilon_low, epsilon_high, num=num))
    for i in range(len(epsilon_list)):
        epsilon_list[i] = round(epsilon_list[i], 1)

    return {
        "domain": domain,
        "epsilon_list": epsilon_list,
        "true": true_frequency,
        "estimated1": estimated_frequency1,
        "estimated2": estimated_frequency2,
        "mse1": mse1,
        "mse2": mse2,
    }


# 类别型数据
def categorical_mechanisms(path, epsilon_low, epsilon_high, mechanism):
    data = Data(path, 'categorical')

    # 隐私预算的间距就默认为0.2
    step = 0.2
    num = int(round((epsilon_high - epsilon_low) / step)) + 1
    epsilon_list = list(np.linspace(epsilon_low, epsilon_high, num=num))
    for i in range(len(epsilon_list)):
        epsilon_list[i] = round(epsilon_list[i], 1)
    estimated_frequency_list = []
    mse_list = []
    for epsilon in epsilon_list:
        perturbed_data_list = []
        for x in data.data:
            user = None
            if mechanism == 'GRR':
                user = GRR_USER(epsilon, data.domain, x)
            elif mechanism == 'SUE':
                user = SUE_USER(epsilon, data.domain, x)
            elif mechanism == 'OUE':
                user = OUE_USER(epsilon, data.domain, x)
            elif mechanism == 'OLH':
                user = OLH_USER(epsilon, data.domain, x)
            elif mechanism == 'EFM':
                user = EFM_USER(epsilon, data.domain, x)
            elif mechanism == 'SS':
                user = SS_USER(epsilon, data.domain, x)
            else:
                print('错误，机制类别未匹配')
            user.run()
            perturbed_data = user.get_per_data()
            perturbed_data_list.append(perturbed_data)
        server = None
        if mechanism == 'GRR':
            server = GRR_SERVER(epsilon, data.domain, perturbed_data_list)
        elif mechanism == 'SUE':
            server = SUE_SERVER(epsilon, data.domain, perturbed_data_list)
        elif mechanism == 'OUE':
            server = OUE_SERVER(epsilon, data.domain, perturbed_data_list)
        elif mechanism == 'OLH':
            server = OLH_SERVER(epsilon, data.domain, perturbed_data_list)
        elif mechanism == 'EFM':
            server = EFM_SERVER(epsilon, data.domain, perturbed_data_list)
        elif mechanism == 'SS':
            server = SS_SERVER(epsilon, data.domain, perturbed_data_list)
        server.estimate()
        estimated_frequency = server.get_es_data()
        estimated_frequency_list.append(estimated_frequency)
        mse = get_mse(data.true_p, estimated_frequency)
        mse_list.append(mse)
    return data.domain, data.true_p, estimated_frequency_list[-1], mse_list

@app.post("/categorical_three")

def main_set_wheel():
    data = Data('./data/set_test.csv', 'set')
    epsilon = 2
    domain = data.domain
    perturbed_data_list = []
    # 每个用户手中持有数据的数量
    c = len(data.data[0])
    for x in data.data:
        # 用户根据隐私预算，原始数据定义域，自身真实数据对类进行初始化
        user = Wheel_USER(epsilon, data.domain, x)
        # 用户在本地进行扰动
        user.run()
        # 取出扰动后的数据
        perturbed_data = user.get_per_data()
        perturbed_data_list.append(perturbed_data)
    # 服务器根据隐私预算，原始数据定义域，收集到的用户扰动数据集合对类进行初始化
    server = Wheel_SERVER(epsilon, domain, perturbed_data_list, c)
    # 服务器进行聚合操作，估计各个数据的频率
    server.estimate()
    # 取出估计频率
    estimated_frequency = server.get_es_data()


def main_numeric_Duchi():
    data = Data('./data/numeric_test.csv', 'numeric')
    epsilon = 2.0
    perturbed_data_list = []
    for x in data.data:
        # 用户根据隐私预算，原始数据定义域，自身真实数据对类进行初始化
        user = Duchi_USER(epsilon, x)
        # 用户在本地进行扰动
        user.run()
        # 取出扰动后的数据
        perturbed_data = user.get_per_data()
        perturbed_data_list.append(perturbed_data)
    # 服务器根据隐私预算，原始数据定义域，收集到的用户扰动数据集合对类进行初始化
    server = Duchi_SERVER(epsilon, perturbed_data_list)
    # 服务器进行聚合操作，估计各个数据的频率
    server.estimate()
    # 取出估计频率
    estimated_mean = server.get_es_mean()
    true_mean = data.true_mean


def main_numeric_PM():
    data = Data('./data/numeric_test.csv', 'numeric')
    epsilon = 2.0
    perturbed_data_list = []
    for x in data.data:
        # 用户根据隐私预算，原始数据定义域，自身真实数据对类进行初始化
        user = PM_USER(epsilon, x)
        # 用户在本地进行扰动
        user.run()
        # 取出扰动后的数据
        perturbed_data = user.get_per_data()
        perturbed_data_list.append(perturbed_data)
    # 服务器根据隐私预算，原始数据定义域，收集到的用户扰动数据集合对类进行初始化
    server = PM_SERVER(epsilon, perturbed_data_list)
    # 服务器进行聚合操作，估计各个数据的频率
    server.estimate()
    # 取出估计频率
    estimated_mean = server.get_es_mean()
    true_mean = data.true_mean


def get_mse(lista, listb):
    d = len(lista)
    mse = 0
    for i in range(d):
        mse += (lista[i] - listb[i]) ** 2
    return mse


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=8006)
