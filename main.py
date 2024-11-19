from fastapi import FastAPI
from prediction import current_version, main_prediction
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()


class Form(BaseModel):
    chain_list: list
    category_list: list
    period: str
    skip_months: int
    download_flag: int


@app.get('/status')  # get status of service
def status():
    return "My status is OK!!!"


@app.get('/version')  # get version of main prediction algo
def version():
    return current_version()


@app.post('/predict')  # start full algo for total data and status_name = Regular
def predict(mask: Form):
    cur_time = datetime.now()
    main_prediction(chain_list=mask.chain_list,
                    category_list=mask.category_list,
                    time_connection=cur_time,
                    period=mask.period,
                    skip_months=mask.skip_months,
                    download_flag=mask.download_flag
                    )
    main_prediction(chain_list=mask.chain_list,
                    category_list=mask.category_list,
                    time_connection=cur_time,
                    period=mask.period,
                    skip_months=mask.skip_months,
                    status_name=2,
                    download_flag=mask.download_flag
                    )
    return (f"Successful with buyers {mask.chain_list}, categories {mask.category_list} "
            f"and download is {mask.download_flag}")


@app.post('/predict_total')  # start full algo for total data
def predict(mask: Form):
    cur_time = datetime.now()
    main_prediction(chain_list=mask.chain_list,
                    category_list=mask.category_list,
                    time_connection=cur_time,
                    period=mask.period,
                    skip_months=mask.skip_months,
                    download_flag=mask.download_flag
                    )
    return (f"Successful with buyers {mask.chain_list}, categories {mask.category_list} "
            f"and download is {mask.download_flag}")


@app.post('/predict_regular')  # start full algo for data only with status_name = Regular
def predict(mask: Form):
    cur_time = datetime.now()
    main_prediction(chain_list=mask.chain_list,
                    category_list=mask.category_list,
                    time_connection=cur_time,
                    period=mask.period,
                    skip_months=mask.skip_months,
                    status_name=2,
                    download_flag=mask.download_flag
                    )
    return (f"Successful with buyers {mask.chain_list}, categories {mask.category_list} "
            f"and download is {mask.download_flag}")

