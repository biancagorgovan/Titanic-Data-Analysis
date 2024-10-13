import pandas as pd
import matplotlib.pyplot as plt

# Citirea fisierului CSV
file_path = '/Users/biancagorgovan/Desktop/PCLP3/proiect/train.csv'
data = pd.read_csv(file_path)

# Filtrarea datelor pentru a obtine doar informatiile despre barbati
male_data = data[data['Sex'] == 'male'].copy()

# Definirea intervalurilor de varsta
age_intervals = [(0, 20), (21, 40), (41, 60), (61, male_data['Age'].max())]

# Functie pentru determinarea categoriei de varsta
def get_age_category(age):
    for i, (start, end) in enumerate(age_intervals):
        if start <= age <= end:
            return i
    return len(age_intervals) - 1

# Adaugarea unei coloane suplimentare pentru indexul categoriei de varsta
male_data.loc[:, 'Age_Category_Index'] = male_data['Age'].apply(get_age_category)

# Calcularea numarului de supravietuitori si a procentului de supravietuire pentru fiecare categorie de varsta
survival_by_age_category = male_data.groupby('Age_Category_Index')['Survived'].agg(['sum', 'count'])
survival_by_age_category['SurvivalRate'] = (survival_by_age_category['sum'] / survival_by_age_category['count']) * 100

# Crearea graficului
plt.figure(figsize=(10, 6))
plt.bar(survival_by_age_category.index, survival_by_age_category['SurvivalRate'], color='skyblue', edgecolor='black')
plt.title('Procentul de supravietuire al barbatilor in functie de categorie de varsta')
plt.xlabel('Categorie de varsta')
plt.ylabel('Procentul de supravietuire (%)')
plt.xticks(range(len(age_intervals)), [f'{start}-{end}' for start, end in age_intervals], rotation=45)
plt.grid(True)
plt.show()

# Salvarea graficului
plt.savefig('survival_rate_by_age_category.png')

# Verificarea primelor cateva randuri ale DataFrame-ului pentru a vedea daca coloana suplimentara este prezenta
print(male_data.head())