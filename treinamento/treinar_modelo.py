import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Carregar dados brutos
caminho_dados = os.path.join("..", "dados", "dados_sensor.json")
with open(caminho_dados, "r") as f:
    registros = json.load(f)

# Inicializa um dicionário para agrupar por ID ou por timestamp
conjuntos = {}

for item in registros:
    id_ = item["id"]
    topico = item["topico"]
    valor = float(item["valor"])

    if id_ not in conjuntos:
        conjuntos[id_] = {}

    conjuntos[id_][topico] = valor

# Agora construímos os vetores X e y
X = []
y = []

for id_, dados in conjuntos.items():
    # Assumindo que 'co' → CO₂, 'qual_ar' → umidade 
    co2 = dados.get("co", 0)
    umidade = dados.get("qual_ar", 0)
    temperatura = dados.get("temp", 25)  # valor padrão se não houver

    # Gerar a label
    # Exemplo: se co2 > 800 ou umidade < 30 → alerta = 1
    alerta = 1 if co2 > 500 or umidade < 30 else 0

    X.append([temperatura, umidade, co2])
    y.append(alerta)

X = np.array(X)
y = np.array(y)

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

print("✅ Modelo treinado e arquivos salvos!")
