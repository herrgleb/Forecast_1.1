from fastapi import FastAPI, Body
from prediction import current_version, main_prediction
from pydantic import BaseModel

app = FastAPI()


class Form(BaseModel):
    buyer_list: list
    category_list: list
    channel: int


@app.get('/status')
def status():
    return "My status is OK!"


@app.get('/version')
def version():
    return current_version()


@app.get('/mood')
def mood():
    return "Don't worry, be happy"


@app.post('/predict')
def predict(mask: Form):
    print(mask.buyer_list)
    print(mask.category_list)
    print(mask.channel)
    main_prediction(buyer_list=mask.buyer_list,
                    category_list=mask.category_list,
                    channel=mask.channel
                    )
    return f"Successful with buyers {mask.buyer_list}, categories {mask.category_list} and channels {mask.channel}"

# {
#     "buyer_list": [4019, 3356],
#     "category_list": [18, 19],
#     "channel": 18
# }
