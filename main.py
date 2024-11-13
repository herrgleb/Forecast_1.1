from fastapi import FastAPI
from prediction import current_version, main_prediction
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()


class Form(BaseModel):
    chain_list: list
    category_list: list
    channel: int


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
                    channel=mask.channel,
                    time_connection=cur_time
                    )
    main_prediction(chain_list=mask.chain_list,
                    category_list=mask.category_list,
                    channel=mask.channel,
                    time_connection=cur_time,
                    status_name=2
                    )
    return f"Successful with buyers {mask.chain_list}, categories {mask.category_list} and channels {mask.channel}"


@app.post('/predict_total')  # start full algo for total data
def predict(mask: Form):
    cur_time = datetime.now()
    main_prediction(chain_list=mask.chain_list,
                    category_list=mask.category_list,
                    channel=mask.channel,
                    time_connection=cur_time
                    )
    return f"Successful with buyers {mask.chain_list}, categories {mask.category_list} and channels {mask.channel}"


@app.post('/predict_regular')  # start full algo for data only with status_name = Regular
def predict(mask: Form):
    cur_time = datetime.now()
    main_prediction(chain_list=mask.chain_list,
                    category_list=mask.category_list,
                    channel=mask.channel,
                    time_connection=cur_time,
                    status_name=2
                    )
    return f"Successful with buyers {mask.chain_list}, categories {mask.category_list} and channels {mask.channel}"

# {
#     "buyer_list": [4019, 3356],
#     "category_list": [18, 19],
#     "channel": 18
# }

# {
#     "buyer_list": [2917],
#     "category_list": [1],
#     "channel": 18
# }
