import json
import numpy as np
import paho.mqtt.client as mqtt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import os

# === CONFIGURAÇÕES DO BROKER MQTT ===
BROKER = "0.0.0.0"  # Mude para o IP do seu broker se necessário
PORT = 1883
TOPICO = "CO / QUAL_AR"  # Substitua se o tópico for diferente

# === CARREGAR MODELO E SCALER ===
caminho_modelo = os.path.join("..", "treinamento", "modelo_sensor.h5")
caminho_scaler = os.path.join("..", "treinamento", "scaler.save")
modelo = load_model(caminho_modelo)
scaler: StandardScaler = joblib.load(caminho_scaler)

# === CALLBACKS DO MQTT ===
def on_connect(client, userdata, flags, rc):
    print("✅ Conectado ao broker! Código:", rc)
    client.subscribe(TOPICO)

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8")
        dados = json.loads(payload)
        print("📡 Dados recebidos:", dados)

        entrada = np.array([[dados["temperatura"], dados["umidade"], dados["co2"]]])
        entrada_normalizada = scaler.transform(entrada)

        resultado = modelo.predict(entrada_normalizada)
        status = "⚠️ ALERTA" if resultado[0][0] >= 0.5 else "✅ Normal"
        print("🧠 Previsão:", status)

    except Exception as e:
        print("❌ Erro ao processar mensagem:", e)

# === CONECTA AO BROKER E INICIA LOOP ===
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 6000)
client.loop_forever()
