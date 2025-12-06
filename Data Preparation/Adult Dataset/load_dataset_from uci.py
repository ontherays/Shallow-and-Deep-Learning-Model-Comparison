import pandas as pd

adult_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
col_names = [
    'age','workclass','fnlwgt','education','education-num','marital-status','occupation',
    'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income'
]

df_adult = pd.read_csv(adult_url, names=col_names, skipinitialspace=True)

df_adult.head()
