import pandas as pd

# Citirea fisierului CSV
file_path = '/Users/biancagorgovan/Desktop/PCLP3/proiect/train.csv'
data = pd.read_csv(file_path)

# Listarea coloanelor numerice si categoriale
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Completarea valorilor lipsa numerice
for col in numeric_cols:
    data[col] = data.groupby(['Pclass', 'Survived'], group_keys=False)[col].apply(lambda x: x.fillna(x.mean()))

# Completarea valorilor lipsa categoriale
for col in categorical_cols:
    data[col] = data.groupby(['Pclass', 'Survived'], group_keys=False)[col].apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))

# Verificarea completarii valorilor lipsa
print(data.isnull().sum())

# Salvarea datelor procesate intr-un fisier nou
data.to_csv('train_completed.csv', index=False)
