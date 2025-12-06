import pandas as pd

# Load Adult dataset directly from UCI
column_names = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","income"
]

url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
url_test = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

df_train = pd.read_csv(url_train, names=column_names, na_values=" ?", skipinitialspace=True)
df_test = pd.read_csv(url_test, names=column_names, na_values=" ?", skipinitialspace=True, skiprows=1)

df = pd.concat([df_train, df_test], ignore_index=True)
df.head()
