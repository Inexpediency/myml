import pandas as pd
import matplotlib.pyplot as plt

#   Reading data frame from excel table
data_frame = pd.read_excel("data/usd_changing.xlsx")

#   Creating plots
# plt.plot(data_frame.curs)
# plt.ylabel('Cost')
# plt.xlabel('Days')
# plt.show()

# List of changing cost of usd
curs = data_frame.curs

past = 4 * 7  # Data for 2 weeks in last
future = 7  # Try predict curs for week in future

start = past  # Start point
end = len(curs) - future  # End point

# Creating new data frame
new_df = []
for i in range(start, end):
    cols = curs[(i-past):(i+future)]
    new_df.append(list(cols))
past_columns = ["Past "+str(p) for p in range(past)]
future_columns = ["Future "+str(f) for f in range(future)]
transformed_df = pd.DataFrame(new_df, columns=(past_columns+future_columns))

print(transformed_df.head())
