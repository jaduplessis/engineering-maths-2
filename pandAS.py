import pandas as pd

df = pd.read_csv('pokemon_data.csv')

# head = top of list, tail = bottom
# print(df.tail(3))

# read each column
# print(df.columns)
# print(df[['Name', 'Type 1', 'HP']][25:40])

# read each row
# print(df.iloc[1:4])

for index, row in df.iterrows():
    print(index, row)
