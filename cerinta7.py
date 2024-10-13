import pandas as pd
import matplotlib.pyplot as plt

# Citirea fisierului CSV
file_path = '/Users/biancagorgovan/Desktop/PCLP3/proiect/train.csv'
data = pd.read_csv(file_path)

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
plt.savefig('survival_rate_children_adults.png')
