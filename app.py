from flask import Flask, render_template, request, jsonify, session
import uuid
import re
import random
from datetime import datetime

import rag_engine  # ← motor RAG

app = Flask(__name__)
app.secret_key = "chat_secret_key_2024"

# Capa social mínima. Todo lo que NO sea esto va al RAG.
SOCIAL_RESPONSES = {
    "saludo": {
        "patterns": [r"\b(hola|hey|buenos dias|buenas tardes|buenas noches|hi|hello|saludos)\b"],
        "responses": [
            "¡Hola! Soy Calidoso, tu asistente del CNA sobre Acreditación en alta calidad. ¿Qué quieres consultar?",
            "¡Bienvenido! Soy Calidoso. Pregúntame sobre el Modelo de Acreditación y te respondo con base en los documentos.",
        ],
    },
    "nombre": {
        "patterns": [r"\b(como te llamas|cual es tu nombre|quien eres|your name)\b"],
        "responses": [
            "Soy Calidoso, tu asistente del CNA sobre Acreditación en alta calidad. 🤖",
        ],
    },
    "estado": {
        "patterns": [r"\b(como estas|como te va|como te encuentras|how are you)\b"],
        "responses": [
            "¡Funcionando perfecto! Listo para consultar la base de conocimientos. ¿Qué necesitas?",
        ],
    },
    "despedida": {
        "patterns": [r"\b(adios|hasta luego|bye|chao|hasta pronto|nos vemos)\b"],
        "responses": [
            "¡Hasta luego! 👋",
            "¡Adiós! Vuelve cuando quieras. 😊",
        ],
    },
}


def get_bot_response(message):
    message_lower = message.lower().strip()

    # 1) Respuestas sociales
    for data in SOCIAL_RESPONSES.values():
        for pattern in data["patterns"]:
            if re.search(pattern, message_lower):
                return random.choice(data["responses"])

    # 2) Todo lo demás → RAG
    try:
        resultado = rag_engine.responder(message, k=5)
        respuesta = resultado["respuesta"]
        if resultado["fuentes"]:
            respuesta += "\n\n📚 Fuente(s): " + ", ".join(resultado["fuentes"])
        return respuesta
    except Exception as e:
        return f"⚠️ Ocurrió un error consultando la base de conocimientos: {e}"


conversation_history = {}


@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Mensaje vacío"}), 400

    session_id = session.get("session_id", str(uuid.uuid4()))
    conversation_history.setdefault(session_id, [])
    conversation_history[session_id].append({
        "role": "user", "content": user_message,
        "timestamp": datetime.now().strftime("%H:%M"),
    })

    bot_response = get_bot_response(user_message)

    conversation_history[session_id].append({
        "role": "bot", "content": bot_response,
        "timestamp": datetime.now().strftime("%H:%M"),
    })
    return jsonify({"response": bot_response, "timestamp": datetime.now().strftime("%H:%M")})


@app.route("/clear", methods=["POST"])
def clear_chat():
    session_id = session.get("session_id")
    if session_id and session_id in conversation_history:
        conversation_history[session_id] = []
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    rag_engine.init()   # carga embeddings + ChromaDB al arrancar
    # use_reloader=False evita que el modelo se cargue dos veces en modo debug
    app.run(debug=True, port=5000, use_reloader=False)