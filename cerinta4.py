import pandas as pd

# Citirea fisierului CSV
file_path = '/Users/biancagorgovan/Desktop/PCLP3/proiect/train.csv'
data = pd.read_csv(file_path)

# Identificarea coloanelor cu valori lipsa
columns_with_missing_values = data.columns[data.isnull().any()].tolist()

# Determinarea numarului si proportiei valorilor lipsa pentru fiecare coloana identificata
missing_values_info = {}
for column in columns_with_missing_values:
    num_missing_values = data[column].isnull().sum()
    proportion_missing_values = num_missing_values / len(data) * 100
    missing_values_info[column] = {'num_missing_values': num_missing_values, 'proportion_missing_values': proportion_missing_values}

# Determinarea procentului valorilor lipsa pentru fiecare dintre cele doua clase din coloana 'Survived'
survived_missing_percentage = data[data['Survived'] == 1][columns_with_missing_values].isnull().mean() * 100
not_survived_missing_percentage = data[data['Survived'] == 0][columns_with_missing_values].isnull().mean() * 100

# Afisarea rezultatelor
print("Coloanele cu valori lipsa:")
print(columns_with_missing_values)
print("\nNumarul si proportia valorilor lipsa pentru fiecare coloana:")
for column, info in missing_values_info.items():
    print(f"{column}: Numarul valorilor lipsa: {info['num_missing_values']}, Procentul: {info['proportion_missing_values']:.2f}%")
print("\nProcentul valorilor lipsa pentru fiecare clasa (Survived):")
print("Survived:")
print(survived_missing_percentage)
print("\nNot Survived:")
print(not_survived_missing_percentage)
