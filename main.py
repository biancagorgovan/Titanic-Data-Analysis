import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#CERINTA 1
print('Cerinta 1')
# Citirea fisierului CSV
file_path = '/Users/biancagorgovan/Desktop/PCLP3/proiect/train.csv'
data = pd.read_csv(file_path)

# Determinarea numarului de coloane
num_columns = data.shape[1]

# Tipurile de date ale fiecarei coloane
data_types = data.dtypes

# Numarul de valori lipsa pentru fiecare coloana
missing_values = data.isnull().sum()

# Numarul de linii
num_rows = data.shape[0]

# Verificarea existentei liniilor duplicate
duplicate_rows = data.duplicated().sum()

# Afisarea rezultatelor
print(f'Numarul de coloane: {num_columns}')
print(f'Tipurile datelor din fiecare coloana:\n{data_types}')
print(f'Numarul de valori lipsa pentru fiecare coloana:\n{missing_values}')
print(f'Numarul de linii: {num_rows}')
print(f'Numarul de linii duplicate: {duplicate_rows}')

#CERINTA 2
print('Cerinta 2')
# Calcularea procentului persoanelor care au supravietuit si care nu au supravietuit
survival_counts = data['Survived'].value_counts(normalize=True) * 100
survival_percentages = survival_counts.to_dict()

# Calcularea procentului pasagerilor pentru fiecare tip de clasa
class_counts = data['Pclass'].value_counts(normalize=True) * 100
class_percentages = class_counts.to_dict()

# Calcularea procentului barbatilor si al femeilor
gender_counts = data['Sex'].value_counts(normalize=True) * 100
gender_percentages = gender_counts.to_dict()

# Crearea graficelor
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Grafic pentru procentul de supravietuire
axs[0].bar(survival_percentages.keys(), survival_percentages.values(), color=['blue', 'orange'])
axs[0].set_title('Procentul persoanelor care au supravietuit vs. nu au supravietuit')
axs[0].set_xlabel('Supravietuire')
axs[0].set_ylabel('Procent (%)')
axs[0].set_xticks([0, 1])
axs[0].set_xticklabels(['Nu', 'Da'])


# Grafic pentru procentul pasagerilor pe clase
axs[1].bar(class_percentages.keys(), class_percentages.values(), color=['green', 'red', 'purple'])
axs[1].set_title('Procentul pasagerilor pentru fiecare tip de clasa')
axs[1].set_xlabel('Clasa')
axs[1].set_ylabel('Procent (%)')

# Grafic pentru procentul barbatilor si femeilor
axs[2].bar(gender_percentages.keys(), gender_percentages.values(), color=['blue', 'pink'])
axs[2].set_title('Procentul barbatilor vs. femeilor')
axs[2].set_xlabel('Gen')
axs[2].set_ylabel('Procent (%)')

# Afisarea graficelor
plt.savefig('CER2_grafice_procente.png')
plt.show()

# Afisarea rezultatelor in text
print(f'Procentul persoanelor care au supravietuit: {survival_percentages}')
print(f'Procentul pasagerilor pentru fiecare tip de clasa: {class_percentages}')
print(f'Procentul barbatilor si femeilor: {gender_percentages}')

#CERINTA 3
print('Cerinta 3')
# Selectarea doar a coloanelor cu valori numerice
numeric_columns = data.select_dtypes(include=['int', 'float']).columns

# Generarea histogramei pentru fiecare coloana numerica
for column in numeric_columns:
    plt.figure(figsize=(8, 6))
    plt.hist(data[column], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histograma pentru coloana "{column}"')
    plt.xlabel('Valoare')
    plt.ylabel('Frecventa')
    plt.grid(True)
    plt.show()
    plt.savefig(f'CER3_histograma_{column}.png')

#CERINTA 4
print('Cerinta 4')
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

#CERINTA 5
print('Cerinnta 5')
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
plt.savefig('CER5_categorii_varsta.png')
# Verificarea primelor cateva randuri ale DataFrame-ului pentru a vedea daca coloana suplimentara este prezenta
print(data.head())

#CERINTA 6
print('Cerinta 6')
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
plt.savefig('CER6_survival_rate_by_age_category.png')

# Verificarea primelor cateva randuri ale DataFrame-ului pentru a vedea daca coloana suplimentara este prezenta
print(male_data.head())

#CERINTA 7
print('Cerinta 7')

# Definirea conditiei pentru copii (varsta < 18 ani)
is_child = data['Age'] < 18

# Calcularea procentului copiilor aflati la bord
num_children = is_child.sum()
total_passengers = len(data)
percentage_children = (num_children / total_passengers) * 100
print(f'Procentul copiilor aflati la bord: {percentage_children:.2f}%')

# Calcularea ratei de supravietuire pentru copii si adulti
survival_rate_children = data[is_child]['Survived'].mean() * 100
survival_rate_adults = data[~is_child]['Survived'].mean() * 100

# Crearea unui DataFrame pentru vizualizarea ratelor de supravietuire
survival_rates = pd.DataFrame({
    'Category': ['Children', 'Adults'],
    'SurvivalRate': [survival_rate_children, survival_rate_adults]
})

# Crearea graficului
plt.figure(figsize=(8, 6))
bars = plt.bar(survival_rates['Category'], survival_rates['SurvivalRate'], color=['skyblue', 'lightgreen'], edgecolor='black')

# Adaugarea etichetelor pe bare
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', va='bottom')  # va='bottom' pentru a pune eticheta deasupra barei

plt.title('Rata de supravietuire pentru copii si adulti')
plt.xlabel('Categorie')
plt.ylabel('Rata de supravietuire (%)')
plt.grid(True)
plt.show()

# Salvarea graficului
plt.savefig('CER7_survival_rate_children_adults.png')

#CERINTA 8
print('Cerinta 8')
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

#CERINTA 9
print('Cerinta 9')
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
plt.savefig('CER9_titles_validation.png')

# Afisarea rezultatului verificarii
print(f"Numar de titluri nevalide: {(~data['Title_Valid']).sum()}")
print(data[['Name', 'Sex', 'Title', 'Title_Valid']])

#CERINTA 10
print('Cerinta 10')
# Determinarea daca pasagerul este singur
data['Is_Alone'] = (data['SibSp'] + data['Parch'] == 0)

# Crearea unei histograme pentru supravietuirea pasagerilor singuri versus cei cu rude
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='Survived', hue='Is_Alone', multiple='dodge', shrink=.8)
plt.title('Supravietuirea pe Titanic: Singuri vs. Cu Rude')
plt.xlabel('Supravietuit')
plt.ylabel('Numar de Pasageri')
plt.xticks([0, 1], ['Nu', 'Da'])
plt.legend(title='Singur', labels=['Nu', 'Da'])
plt.show()
plt.savefig('CER10_InfluenceAlone.png')

# Reducerea setului de date la primele 100 de inregistrari
data_100 = data.head(100)

plt.figure(figsize=(10, 8))
cat_plot = sns.catplot(data=data_100, x='Pclass', y='Fare', hue='Survived', kind='swarm', height=5, aspect=2, s=5)  # s este dimensiunea markerului
cat_plot.set_axis_labels('Clasa', 'Tarif')
cat_plot.fig.suptitle('Titanic: Tarif vs. Clasa vs. Supravietuire (Primele 100 inregistrari)', y=1.05)
plt.show()
plt.savefig('CER10_InfluenceClassEtc.png')
