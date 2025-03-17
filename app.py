import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Carregar os dados
df = pd.read_csv("loan.csv")

# Remover colunas desnecessárias e valores nulos
df = df.dropna()
df = df.drop(columns=['Loan_ID'])

# Codificar colunas categóricas
label_encoders = {}
categorical_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents', 'Loan_Status']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Guardar os label encoders para uso futuro

# Separar variáveis independentes (X) e dependente (y)
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

# Salvar o modelo treinado
joblib.dump(model, "model\model.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")

print("Modelo treinado e salvo com sucesso!")
