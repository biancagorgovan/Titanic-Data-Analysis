import pandas as pd
import matplotlib.pyplot as plt

# Citirea fisierului CSV
file_path = '/Users/biancagorgovan/Desktop/PCLP3/proiect/train.csv'
data = pd.read_csv(file_path)

# Functie pentru extragerea titlurilor din nume
def extract_title(name):
    import re
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

# Adaugarea unei coloane noi pentru titlu
data['Title'] = data['Name'].apply(extract_title)

# Dictionar care mapeaza titlurile la sexul asteptat
title_to_sex = {
    'Mr': 'male', 'Master': 'male', 'Don': 'male', 'Sir': 'male', 'Rev': 'male',
    'Col': 'male', 'Capt': 'male', 'Major': 'male', 'Jonkheer': 'male',
    'Mrs': 'female', 'Miss': 'female', 'Mme': 'female', 'Ms': 'female', 'Mlle': 'female',
    'Lady': 'female', 'Dona': 'female', 'Countess': 'female'
}

# Functie pentru verificarea corectitudinii titlurilor folosind dictionarul
def check_title_sex(row):
    expected_sex = title_to_sex.get(row['Title'], None)
    return expected_sex is None or row['Sex'] == expected_sex

# Aplicarea functiei de verificare si adaugarea unei coloane pentru validare
data['Title_Valid'] = data.apply(check_title_sex, axis=1)

# Calcularea numarului de titluri valide si invalide
valid_title_counts = data[data['Title_Valid']]['Title'].value_counts()
invalid_title_counts = data[~data['Title_Valid']]['Title'].value_counts()

plt.figure(figsize=(14, 7))

# Grafic pentru titlurile valide
plt.subplot(1, 2, 1)
plt.bar(valid_title_counts.index, valid_title_counts.values, color='lightgreen', edgecolor='black')
plt.title('Distributia titlurilor valide')
plt.xlabel('Titlu')
plt.ylabel('Numar de persoane')
plt.xticks(rotation=45)

# Grafic pentru titlurile invalide
plt.subplot(1, 2, 2)
plt.bar(invalid_title_counts.index, invalid_title_counts.values, color='red', edgecolor='black')
plt.title('Distributia titlurilor invalide')
plt.xlabel('Titlu')
plt.ylabel('Numar de persoane')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Salvarea graficului
plt.savefig('titles_validation.png')

# Afisarea rezultatului verificarii
print(f"Numar de titluri nevalide: {(~data['Title_Valid']).sum()}")
print(data[['Name', 'Sex', 'Title', 'Title_Valid']])
