import pandas as pd
import matplotlib.pyplot as plt

# Citirea fisierului CSV
file_path = '/Users/biancagorgovan/Desktop/PCLP3/proiect/train.csv'
data = pd.read_csv(file_path)

# Definirea intervalurilor de varsta
age_intervals = [(0, 20), (21, 40), (41, 60), (61, data['Age'].max())]

# Functie pentru determinarea categoriei de varsta
def get_age_category(age):
    for i, (start, end) in enumerate(age_intervals):
        if start <= age <= end:
            return i
    return len(age_intervals) - 1

# Adaugarea unei coloane suplimentare pentru indexul categoriei de varsta
data['Age_Category_Index'] = data['Age'].apply(get_age_category)

# Determinarea numarului de pasageri pentru fiecare categorie de varsta
passengers_per_category = data['Age_Category_Index'].value_counts().sort_index()

# Crearea graficului
plt.figure(figsize=(8, 6))
plt.bar(range(len(age_intervals)), passengers_per_category, color='skyblue', edgecolor='black')
plt.title('Numarul de pasageri in functie de categorie de varsta')
plt.xlabel('Categorie de varsta')
plt.ylabel('Numar de pasageri')
plt.xticks(range(len(age_intervals)), [f'{start}-{end}' for start, end in age_intervals], rotation=45)
plt.grid(True)
plt.show()
plt.savefig('categorii_varsta.png')
# Verificarea primelor cateva randuri ale DataFrame-ului pentru a vedea daca coloana suplimentara este prezenta
print(data.head())