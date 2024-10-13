import pandas as pd
import matplotlib.pyplot as plt

# Citirea fisierului CSV
file_path = '/Users/biancagorgovan/Desktop/PCLP3/proiect/train.csv'
data = pd.read_csv(file_path)

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
    plt.savefig(f'histograma_{column}.png')