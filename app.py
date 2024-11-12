from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import csv
from datetime import datetime

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
@app.post("/GRR")
async def upload_csv(file: UploadFile = File(...)):
    if file.filename.endswith(".csv"):
        contents = await file.read()
        lines = contents.decode().splitlines()
        reader = csv.reader(lines)
        keys = next(reader)
        data = []
        c = 0
        for row in reader:
            # c += 1
            # if c > 10:
            #     break
            row_data = dict(zip(keys, row))
            data.append(row_data)
            
        with open(file.filename+'1', 'wb') as f:
            f.write(contents)
    return {"d":data,"h":keys}


@app.post("/SUE")
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

        with open(file.filename + '2', 'wb') as f:
            f.write(contents)
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
    return {"d": data, "h": keys}


if __name__ == "__main__":
    uvicorn.run(app=app,host="0.0.0.0",port=8006)