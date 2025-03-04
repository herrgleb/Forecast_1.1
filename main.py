from fastapi import FastAPI
from prediction import current_version, main_prediction
from prediction_week import main_prediction_v2
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()


class Form(BaseModel):
    chain_list: list
    category_list: list
    final_date: str
    skip_months: int
    period: int
    download_flag: int

class MODELTWO(BaseModel):
    date_type: str
    cpg_list: list
    ppg_list: list
    status_id: int
    time_connection: str
    final_fact_date: str
    rolling_dict: dict
    number_of_zeros: dict
    horizon_frcst: dict
    simplest_model_range: dict
    growing_range: dict


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
                    final_date=mask.final_date,
                    skip_months=mask.skip_months,
                    period = mask.period,
                    download_flag=mask.download_flag
                    )
    main_prediction(chain_list=mask.chain_list,
                    category_list=mask.category_list,
                    time_connection=cur_time,
                    final_date=mask.final_date,
                    skip_months=mask.skip_months,
                    period=mask.period,
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
                    final_date=mask.final_date,
                    skip_months=mask.skip_months,
                    period=mask.period,
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
                    final_date=mask.final_date,
                    skip_months=mask.skip_months,
                    period=mask.period,
                    status_name=2,
                    download_flag=mask.download_flag
                    )
    return (f"Successful with buyers {mask.chain_list}, categories {mask.category_list} "
            f"and download is {mask.download_flag}")


@app.post('/predict2_total')
def predict(mask: MODELTWO):
    cur_time = datetime.now()
    main_prediction_v2(date_type=mask.date_type,
                       cpg_list=mask.cpg_list,
                       ppg_list=mask.ppg_list,
                       status_id=mask.status_id,
                       time_connection=cur_time,
                       final_fact_date=mask.final_fact_date,
                       rolling_dict=mask.rolling_dict,
                       number_of_zeros=mask.number_of_zeros,
                       horizon_frcst=mask.horizon_frcst,
                       simplest_model_range=mask.simplest_model_range,
                       growing_range=mask.growing_range
                       )
    return (f"Successful with buyers {mask.cpg_list}, categories {mask.ppg_list}")


