import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Carregar dados
caminho_dados = os.path.join("..", "dados", "dados_sensor.json")
with open(caminho_dados, "r") as f:
    dados = json.load(f)

# Preparar os dados
X = np.array([[d["temperatura"], d["umidade"], d["co2"]] for d in dados])
y = np.array([d["alerta"] for d in dados])

# Normalização
scaler = StandardScaler()
X_normalizado = scaler.fit_transform(X)

# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_normalizado, y, test_size=0.2, random_state=42)

# Criar modelo
modelo = Sequential()
modelo.add(Dense(10, input_dim=3, activation='relu'))
modelo.add(Dense(8, activation='relu'))
modelo.add(Dense(1, activation='sigmoid'))

modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
modelo.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

# Salvar modelo e scaler
modelo.save("modelo_sensor.h5")
joblib.dump(scaler, "scaler.save")

print("✅ Treinamento finalizado e modelo salvo!")
