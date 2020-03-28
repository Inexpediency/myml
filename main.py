import pandas as pd
import matplotlib.pyplot as plt

#   Reading data frame from excel table
data_frame = pd.read_excel("data/usd_changing.xlsx")

print(data_frame.describe())

#   Creating plots
# plt.plot(data_frame.curs)
# plt.ylabel('Cost')
# plt.xlabel('Days')
# plt.show()
