import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Citirea fisierului CSV
file_path = '/Users/biancagorgovan/Desktop/PCLP3/proiect/train.csv'
data = pd.read_csv(file_path)

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
plt.savefig('InfluenceAlone.png')

# Reducerea setului de date la primele 100 de inregistrari
data_100 = data.head(100)

plt.figure(figsize=(10, 8))
cat_plot = sns.catplot(data=data_100, x='Pclass', y='Fare', hue='Survived', kind='swarm', height=5, aspect=2, s=5)  # s este dimensiunea markerului
cat_plot.set_axis_labels('Clasa', 'Tarif')
cat_plot.fig.suptitle('Titanic: Tarif vs. Clasa vs. Supravietuire (Primele 100 inregistrari)', y=1.05)
plt.show()
plt.savefig('InfluenceClassEtc.png')