import json
import numpy as np
import paho.mqtt.client as mqtt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import os

# === CONFIGURAÇÕES DO BROKER MQTT ===
BROKER = "localhost"  # Altere se for remoto
PORT = 1883
TOPICOS = [("co", 0), ("qual_ar", 0), ("temp", 0)]  # Substitua ou adicione outros se necessário

# === CAMINHOS PARA O MODELO E O SCALER ===
caminho_modelo = os.path.join("..", "treinamento", "modelo_sensor.h5")
caminho_scaler = os.path.join("..", "treinamento", "scaler.save")
modelo = load_model(caminho_modelo)
scaler: StandardScaler = joblib.load(caminho_scaler)

# === BUFFER PARA VALORES DOS SENSORES ===
valores_recebidos = {
    "co": None,
    "qual_ar": None,
    "temp": None
}

def tentar_prever():
    if None not in valores_recebidos.values():
        entrada = np.array([[valores_recebidos["temp"], valores_recebidos["qual_ar"], valores_recebidos["co"]]])
        entrada_normalizada = scaler.transform(entrada)
        resultado = modelo.predict(entrada_normalizada)
        status = "⚠️ ALERTA" if resultado[0][0] >= 0.5 else "✅ Normal"
        print("🧠 Previsão:", status)
    else:
        print("⏳ Aguardando dados completos...")

# === CALLBACKS MQTT ===
def on_connect(client, userdata, flags, rc):
    print("✅ Conectado ao broker! Código:", rc)
    client.subscribe(TOPICOS)

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8")
        dados = json.loads(payload)

        # Esperado: {"data": "...", "id": ..., "topico": "co", "valor": "435"}
        topico = dados.get("topico")
        valor = float(dados.get("valor", 0))

        if topico in valores_recebidos:
            valores_recebidos[topico] = valor
            print(f"📡 {topico.upper()}: {valor}")
            tentar_prever()

    except Exception as e:
        print("❌ Erro ao processar mensagem:", e)

# === CONECTA AO BROKER ===
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT, 60)
client.loop_forever()
