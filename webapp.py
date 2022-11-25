from typing import List
from typing import Union
from pydantic import BaseModel
from joblib import load
from fastapi import FastAPI, Request

# http://127.0.0.1:8000/app?sepallength=100&sepalwidth=100&petallength=0.2&petalwidth=0.2
app = FastAPI()


class Outputtype(BaseModel):
    prediction: int
    # 'prediction' should match with the key of the return statement
    # return{'prediction': str(yhat[0])}
    human_readable: str

# load the model
modelPL = load('iris_bestpipeline.joblib')

# def main():
#    read_root()

@app.get("/app", response_model=Outputtype)
# the string argument defines the web page
# (relative to the root address)
# where this function will work
def read_root(req: Request):
    # collect input parameters
    petallength = float(req.query_params['petallength'])
    petalwidth = float(req.query_params['petalwidth'])
    sepallength = float(req.query_params['sepallength'])
    sepalwidth = float(req.query_params['sepalwidth'])
    sample = [petallength, petalwidth, sepallength, sepalwidth]

    # do the prediction
    yhat = modelPL.predict([sample])

    d = {
        0: 'iris setosa1',
        1: 'iris_setosa2',
        2: 'iris_setosa3'
    }
    return{'prediction': str(yhat[0]), 'human_readable': d[yhat[0]]}

# if __name__ == '__main__':
#    main()
