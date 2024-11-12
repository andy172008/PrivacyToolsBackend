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
'''接受文件'''


# 类别型数据频率估计
@app.post("/GRR")
async def upload_csv(file: UploadFile = File(...), epsilon: float = Form(...)):
    # return {"error": "服务器回传文件错误"}
    if file.filename.endswith(".txt"):
        contents = await file.read()
        # 解码并按行分割
        lines = contents.decode().splitlines()
        # 去除每行的首尾空格，并过滤掉空行
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        # 将前端传来的文件储存到本地
        file_path = "./dataset/"+file.filename
        with open(file_path, 'w') as f:
            for line in cleaned_lines:
                f.write(f"{line}\n")  # 写入每个数字到新文件中
        true_frequency, estimated_frequency, mse, domain = main_categorical_GRR(file_path,epsilon)
        return {
            "true": true_frequency,
            "estimated": estimated_frequency,
            "mse": mse,
            "domain": domain
        }
    return {"error": "服务器回传文件错误"}


@app.post("/SUE")
async def upload_csv(file: UploadFile = File(...)):
    if file.filename.endswith(".txt"):
        contents = await file.read()
        lines = contents.decode().splitlines()
        reader = csv.reader(lines)
        keys = next(reader)
        data = []
        c = 0
        for row in reader:
            c += 1
            if c > 10:
                break
            row_data = dict(zip(keys, row))
            data.append(row_data)

        with open(file.filename + '2', 'wb') as f:
            f.write(contents)
        #main_categorical_SUE()
        return {"d": data, "h": keys}


@app.post("/OUE")
async def upload_csv(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        contents = await file.read()
        lines = contents.decode().splitlines()
        reader = csv.reader(lines)
        keys = next(reader)
        data = []
        c = 0
        for row in reader:
            c += 1
            if c > 10:
                break
            row_data = dict(zip(keys, row))
            data.append(row_data)

        with open(file.filename + '3', 'wb') as f:
            f.write(contents)
        #main_categorical_OUE()
        return {"d": data, "h": keys}


@app.post("/OLH")
async def upload_csv(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        contents = await file.read()
        lines = contents.decode().splitlines()
        reader = csv.reader(lines)
        keys = next(reader)
        data = []
        c = 0
        for row in reader:
            c += 1
            if c > 10:
                break
            row_data = dict(zip(keys, row))
            data.append(row_data)

        with open(file.filename + '4', 'wb') as f:
            f.write(contents)
        #main_categorical_OLH()
        return {"d": data, "h": keys}


@app.post("/SS")
async def upload_csv(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        contents = await file.read()
        lines = contents.decode().splitlines()
        reader = csv.reader(lines)
        keys = next(reader)
        data = []
        c = 0
        for row in reader:
            c += 1
            if c > 10:
                break
            row_data = dict(zip(keys, row))
            data.append(row_data)

        with open(file.filename + '5', 'wb') as f:
            f.write(contents)
        #main_categorical_SS()
        return {"d": data, "h": keys}


@app.post("/EFM")
async def upload_csv(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        contents = await file.read()
        lines = contents.decode().splitlines()
        reader = csv.reader(lines)
        keys = next(reader)
        data = []
        c = 0
        for row in reader:
            c += 1
            if c > 10:
                break
            row_data = dict(zip(keys, row))
            data.append(row_data)

        with open(file.filename + '6', 'wb') as f:
            f.write(contents)
        #main_categorical_EFM()
        return {"d": data, "h": keys}


@app.post("/set_wheel")
async def upload_csv(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        contents = await file.read()
        lines = contents.decode().splitlines()
        reader = csv.reader(lines)
        keys = next(reader)
        data = []
        c = 0
        for row in reader:
            c += 1
            if c > 10:
                break
            row_data = dict(zip(keys, row))
            data.append(row_data)

        with open(file.filename + '7', 'wb') as f:
            f.write(contents)
        #main_set_wheel()
        return {"d": data, "h": keys}


@app.post("/numeric_Duchi")
async def upload_csv(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        contents = await file.read()
        lines = contents.decode().splitlines()
        reader = csv.reader(lines)
        keys = next(reader)
        data = []
        c = 0
        for row in reader:
            c += 1
            if c > 10:
                break
            row_data = dict(zip(keys, row))
            data.append(row_data)

        with open(file.filename + '8', 'wb') as f:
            f.write(contents)
        #main_numeric_Duchi()
        return {"d": data, "h": keys}


@app.post("/numeric_PM")
async def upload_csv(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        contents = await file.read()
        lines = contents.decode().splitlines()
        reader = csv.reader(lines)
        keys = next(reader)
        data = []
        c = 0
        for row in reader:
            c += 1
            if c > 10:
                break
            row_data = dict(zip(keys, row))
            data.append(row_data)

        with open(file.filename + '9', 'wb') as f:
            f.write(contents)
        #main_numeric_PM()
        return {"d": data, "h": keys}


# 类别型数据频率估计的测试函数
def main_categorical_GRR(path,epsilon):
    data = Data(path, 'categorical')
    domain = data.domain
    print(domain)
    perturbed_data_list = []
    for x in data.data:
        # 用户根据隐私预算，原始数据定义域，自身真实数据对类进行初始化
        user = GRR_USER(epsilon, domain, x)
        # 用户在本地进行扰动
        user.run()
        # 取出扰动后的数据
        perturbed_data = user.get_per_data()
        perturbed_data_list.append(perturbed_data)
    # 服务器根据隐私预算，原始数据定义域，收集到的用户扰动数据集合对类进行初始化
    server = GRR_SERVER(epsilon, domain, perturbed_data_list)
    # 服务器进行聚合操作，估计各个数据的频率
    server.estimate()
    # 取出估计频率
    estimated_frequency = server.get_es_data()
    mse = get_mse(data.true_p, estimated_frequency)
    # 将真实频率和估计频率同时返回
    return data.true_p, estimated_frequency, mse, domain


def main_categorical_SUE():
    data = Data('./data/categorical_test.txt', 'categorical')
    epsilon = 2
    domain = data.domain
    perturbed_data_list = []
    for x in data.data:
        # 用户根据隐私预算，原始数据定义域，自身真实数据对类进行初始化
        user = SUE_USER(epsilon, domain, x)
        # 用户在本地进行扰动
        user.run()
        # 取出扰动后的数据
        perturbed_data = user.get_per_data()
        perturbed_data_list.append(perturbed_data)
    # 服务器根据隐私预算，原始数据定义域，收集到的用户扰动数据集合对类进行初始化
    server = SUE_SERVER(epsilon, domain, perturbed_data_list)
    # 服务器进行聚合操作，估计各个数据的频率
    server.estimate()
    # 取出估计频率
    estimated_frequency = server.get_es_data()



def main_categorical_OUE():
    data = Data('./data/categorical_test.txt', 'categorical')
    epsilon = 2
    domain = data.domain
    perturbed_data_list = []
    for x in data.data:
        # 用户根据隐私预算，原始数据定义域，自身真实数据对类进行初始化
        user = OUE_USER(epsilon, domain, x)
        # 用户在本地进行扰动
        user.run()
        # 取出扰动后的数据
        perturbed_data = user.get_per_data()
        perturbed_data_list.append(perturbed_data)
    # 服务器根据隐私预算，原始数据定义域，收集到的用户扰动数据集合对类进行初始化
    server = OUE_SERVER(epsilon, domain, perturbed_data_list)
    # 服务器进行聚合操作，估计各个数据的频率
    server.estimate()
    # 取出估计频率
    estimated_frequency = server.get_es_data()


def main_categorical_OLH():
    data = Data('./data/categorical_test.txt', 'categorical')
    epsilon = 2
    domain = data.domain
    perturbed_data_list = []
    for x in data.data:
        # 用户根据隐私预算，原始数据定义域，自身真实数据对类进行初始化
        user = OLH_USER(epsilon, domain, x)
        # 用户在本地进行扰动
        user.run()
        # 取出扰动后的数据
        perturbed_data = user.get_per_data()
        perturbed_data_list.append(perturbed_data)
    # 服务器根据隐私预算，原始数据定义域，收集到的用户扰动数据集合对类进行初始化
    server = OLH_SERVER(epsilon, domain, perturbed_data_list)
    # 服务器进行聚合操作，估计各个数据的频率
    server.estimate()
    # 取出估计频率
    estimated_frequency = server.get_es_data()


def main_categorical_SS():
    data = Data('./data/categorical_test.txt', 'categorical')
    epsilon = 2
    domain = data.domain
    perturbed_data_list = []
    # 方案中扰动数据里1的个数，可以在1到len(domain)中任意取值，这里暂定为4
    d = 4
    for x in data.data:
        # 用户根据隐私预算，原始数据定义域，自身真实数据和d对类进行初始化
        user = SS_USER(epsilon, domain, x, d)
        # 用户在本地进行扰动
        user.run()
        # 取出扰动后的数据
        perturbed_data = user.get_per_data()
        perturbed_data_list.append(perturbed_data)
    # 服务器根据隐私预算，原始数据定义域，收集到的用户扰动数据集合对类进行初始化
    server = SS_SERVER(epsilon, domain, perturbed_data_list, d)
    # 服务器进行聚合操作，估计各个数据的频率
    server.estimate()
    # 取出估计频率
    estimated_frequency = server.get_es_data()


def main_categorical_EFM():
    data = Data('./data/categorical_test.txt', 'categorical')
    epsilon = 2
    domain = data.domain
    perturbed_data_list = []
    # 请注意，应满足d<m
    # 方案中扰动数据里1的个数，可以在1到len(domain)中任意取值，这里暂定为4
    d = 4
    # 方案中扰动数据里1的个数，可以在1到len(domain)中任意取值，这里暂定为4
    m = 8
    for x in data.data:
        # 用户根据隐私预算，原始数据定义域，自身真实数据对类进行初始化
        user = EFM_USER(epsilon, domain, x, d, m)
        # 用户在本地进行扰动
        user.run()
        # 取出扰动后的数据
        perturbed_data = user.get_per_data()
        perturbed_data_list.append(perturbed_data)
    # 服务器根据隐私预算，原始数据定义域，收集到的用户扰动数据集合对类进行初始化
    server = EFM_SERVER(epsilon, domain, perturbed_data_list, d, m)
    # 服务器进行聚合操作，估计各个数据的频率
    server.estimate()
    # 取出估计频率
    estimated_frequency = server.get_es_data()


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

def get_mse(lista,listb):
    d = len(lista)
    mse = 0
    for i in range(d):
        mse += (lista[i] - listb[i])**2
    return mse

if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=8006)
    '''
    main_categorical_GRR()
    main_categorical_SUE()
    main_categorical_OUE()
    main_categorical_OLH()
    main_categorical_SS()
    main_categorical_EFM()

    main_set_wheel()

    main_numeric_Duchi()
    main_numeric_PM()
'''