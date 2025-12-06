df = df_adult.copy()

# Remove rows with '?'
df = df.replace('?', pd.NA).dropna()

# Encode categorical values
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split into X and y
X = df.drop('income', axis=1)
y = df['income']

X.shape, y.shape
