# -*- coding: utf-8 -*-
# @Time : 2022/1/12 3:19 下午
# @Author : 贺星宇
# @File : main.py
# @Software: PyCharm

from LDPtool import *
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()
'''中间件跨域'''
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 可通过浏览器测试是否程序运行
@app.get("/")
async def read_root():
    return {"message": "当你看到这一行文字，说明后端部署完成"}


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
    epsilon_list = get_epsilon_list(epsilon_low, epsilon_high)

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
    print('开始响应categorical_two')
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
    epsilon_list = get_epsilon_list(epsilon_low, epsilon_high)

    return {
        "domain": domain,
        "epsilon_list": epsilon_list,
        "true": true_frequency,
        "estimated1": estimated_frequency1,
        "estimated2": estimated_frequency2,
        "mse1": mse1,
        "mse2": mse2,
    }


# 执行类别型数据单个机制
def categorical_mechanisms(path, epsilon_low, epsilon_high, mechanism):
    data = Data(path, 'categorical')

    epsilon_list = get_epsilon_list(epsilon_low, epsilon_high)

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


# 执行数值型数据单个机制
def numerical_mechanisms(path, epsilon_low, epsilon_high, mechanism):
    data = Data(path, 'numerical')

    epsilon_list = get_epsilon_list(epsilon_low, epsilon_high)

    estimated_mean_list = []
    mse_list = []
    for epsilon in epsilon_list:
        perturbed_data_list = []
        for x in data.data:
            user = None
            if mechanism == 'Duchi':
                user = Duchi_USER(epsilon, x)
            elif mechanism == 'PM':
                user = PM_USER(epsilon, x)
            else:
                print('错误，机制类型未匹配')
            user.run()
            perturbed_data = user.get_per_data()
            perturbed_data_list.append(perturbed_data)
        server = None
        if mechanism == 'Duchi':
            server = Duchi_SERVER(epsilon, perturbed_data_list)
        elif mechanism == 'PM':
            server = PM_SERVER(epsilon, perturbed_data_list)
        server.estimate()
        estimated_mean = server.get_es_mean()
        estimated_mean_list.append(estimated_mean)
        mse = get_mse([data.true_mean], [estimated_mean])
        mse_list.append(mse)
    return data.true_mean, estimated_mean_list, mse_list


# 执行集合型数据单个机制
def set_mechanisms(path, epsilon_low, epsilon_high, mechanism):
    data = Data(path, 'set')

    epsilon_list = get_epsilon_list(epsilon_low, epsilon_high)

    estimated_frequency_list = []
    mse_list = []
    for epsilon in epsilon_list:
        perturbed_data_list = []
        for x in data.data:
            user = None
            if mechanism == 'Wheel':
                user = Wheel_USER(epsilon, data.domain, x)
            elif mechanism == 'PrivSet':
                user = PrivSet_USER(epsilon, data.domain, x)
            else:
                print('错误，机制类型未匹配')
            user.run()
            perturbed_data = user.get_per_data()
            perturbed_data_list.append(perturbed_data)
        server = None
        if mechanism == 'Wheel':
            server = Wheel_SERVER(epsilon, data.domain, perturbed_data_list, data.set_size)
        elif mechanism == 'PrivSet':
            server = PrivSet_SERVER(epsilon, data.domain, perturbed_data_list, data.set_size)
        server.estimate()
        estimated_frequency = server.get_es_data()
        estimated_frequency_list.append(estimated_frequency)
        mse = get_mse(data.true_p, estimated_frequency)
        mse_list.append(mse)
    return data.domain, data.true_p, estimated_frequency_list[-1], mse_list


# 执行键值型数据单个机制
def key_value_mechanisms(path, epsilon_low, epsilon_high, mechanism):
    data = Data(path, 'key-value')

    epsilon_list = get_epsilon_list(epsilon_low, epsilon_high)

    estimated_frequency_list = []
    estimated_mean_list = []

    mse_frequency_list = []
    mse_mean_list = []

    for epsilon in epsilon_list:
        perturbed_data_list = []
        for x in data.data:
            user = None
            if mechanism == 'PCKVGRR':
                user = PCKVGRR_USER(epsilon, x, len(data.domain))
            elif mechanism == 'PCKVUE':
                user = PCKVUE_USER(epsilon, x, len(data.domain))
            user.run()
            perturbed_data = user.get_per_data()
            perturbed_data_list.append(perturbed_data)
        server = None
        if mechanism == 'PCKVGRR':
            server = PCKVGRR_SERVER(epsilon, perturbed_data_list, len(data.domain))
        elif mechanism == 'PCKVUE':
            server = PCKVUE_SERVER(epsilon, perturbed_data_list, len(data.domain))
        server.estimate()
        # 因为代码中0这个位置他没有放置元素，因此我们做一个切片
        temp_frequency = server.F[1:]
        estimated_frequency_list.append(temp_frequency)
        temp_mean = server.M[1:]
        estimated_mean_list.append(temp_mean)
        mse_frequency = get_mse(data.true_key_p, temp_frequency)
        mse_frequency_list.append(mse_frequency)
        mse_mean = get_mse(data.true_mean, temp_mean)
        mse_mean_list.append(mse_mean)
    return estimated_frequency_list, estimated_mean_list, mse_frequency_list, mse_mean_list


# TODO 执行位置数据单个机制
def location_mechanisms(path, epsilon_low, epsilon_high, mechanism):
    pass


# 执行有序数据单个机制
def order_mechanisms(path, epsilon_low, epsilon_high, mechanism):
    data = Data(path, 'order')

    epsilon_list = get_epsilon_list(epsilon_low, epsilon_high)

    estimated_frequency_list = []
    mse_list = []
    for epsilon in epsilon_list:
        perturbed_data_list = []
        count = 0
        for x in data.data:
            user = None
            if mechanism == 'EM':
                user = EM_USER(epsilon, data.domain, x)
            elif mechanism == 'GEM':
                user = GEM_USER(epsilon, data.domain, x)
            else:
                print('错误，机制类别未匹配')
            user.run()
            print(count)
            count += 1
            perturbed_data = user.get_per_data()
            perturbed_data_list.append(perturbed_data)
        server = None
        if mechanism == 'EM':
            server = EM_SERVER(epsilon, data.domain, perturbed_data_list)
        elif mechanism == 'GEM':
            server = GEM_SERVER(epsilon, data.domain, perturbed_data_list)
        server.estimate()
        estimated_frequency = server.get_es_data()
        estimated_frequency_list.append(estimated_frequency)
        mse = get_mse(data.true_p, estimated_frequency)
        mse_list.append(mse)
    return data.domain, data.true_p, estimated_frequency_list[-1], mse_list


# 根据隐私预算上下界，得到对应的隐私预算list
def get_epsilon_list(epsilon_low, epsilon_high):
    # 隐私预算的间距就默认为0.2
    step = 0.2
    num = int(round((epsilon_high - epsilon_low) / step)) + 1
    epsilon_list = list(np.linspace(epsilon_low, epsilon_high, num=num))
    for i in range(len(epsilon_list)):
        epsilon_list[i] = round(epsilon_list[i], 1)
    return epsilon_list


def get_mse(list1, list2):
    d = len(list1)
    mse = 0
    for i in range(d):
        mse += (list1[i] - list2[i]) ** 2
    return mse


def test():
    temp = order_mechanisms('./dataset/mini_categorical_test.txt', 1.0, 1.2, 'EM')
    pass


if __name__ == '__main__':
    test()
    # uvicorn.run(app=app, host="0.0.0.0", port=8006)
