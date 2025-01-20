import pandas as pd
import os

Data=[
    {"name":"Abhishek","age":23,"city":"Bhopal"},
    {"name":"Nishal","age":24,"city":"Pune"},
    {"name":"Joheb","age":34,"city":"Hyderabad"},
    {"name":"Naveen","age":36,"city":"Hyderabad"}

]

df  = pd.DataFrame(Data)

if not os.path.exists("data"):
    os.makedirs("data")

# Save the DataFrame to a CSV file
df.to_csv("data/data.csv", index=False)