import pandas as pd


Data=[
    {"name":"Abhishek","age":23,"city":"Bhopal"},
    {"name":"Nishal","age":24,"city":"Pune"},
    {"name":"Joheb","age":34,"city":"Hyderabad"}

]

df  = pd.DataFrame(Data)

df.to_csv("data/data.csv", index = False)