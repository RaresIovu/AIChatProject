from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI

# Încarcă variabilele de mediu
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# __name__ este o variabilă specială în Python care conține numele modulului curent.
# Dacă fișierul este rulat direct (de exemplu python app-fallback-openAI.py), atunci __name__ == "__main__".
# Flask folosește această informație pentru a ști unde să caute resursele (fișiere statice, șabloane etc.).

app = Flask(__name__)

# Foldere pentru imagini
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Client OpenAI (dacă există cheia)
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


# Ruta principală – pagina HTML
@app.route("/")
def index():
    return render_template("index.html")

# Endpoint pentru upload și procesare imagine
@app.route("/upload", methods=["POST"])
def upload():
    #  un obiect de tip FileStorage, adică un flux binar — nu o imagine încă!
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Nu a fost încărcată nicio imagine"}), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.dilate(edges, None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = [
        {"name":"Triunghiuri","count": 0},
        {"name":"Pătrate","count": 0},
        {"name":"Dreptunghiuri","count": 0},
        {"name":"Pentagoane","count": 0},
        {"name":"Cercuri","count": 0}
    ]
    def increment(name):
        for shape in shapes:
            if shape["name"] == name:
                shape["count"] += 1
                break
    for cnt in contours:
        label = ""
        approx = cv2.approxPolyDP(cnt,  0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            increment("Triunghiuri")
            label = "Triunghi"
        elif len(approx) == 4:
            _,_,w,h = cv2.boundingRect(approx)
            aspectRatio = w / float(h)
            if 0.9 < aspectRatio < 1.1:
                increment("Pătrate")
                label = "Patrat"
            else:
                increment("Dreptunghiuri")
                label = "Dreptunghi"
        elif len(approx) == 5:
            increment("Pentagoane")
            label = "Pentagon"
        else:
            increment("Cercuri")
            label = "Cerc"
        m = cv2.moments(cnt)
        cx = int(m["m10"]/m["m00"])
        cy = int(m["m01"]/m["m00"])
        brightness = gray[cy,cx]
        print("brightness is", brightness)
        if brightness > 50:
           text_color = (0,0,0)
        else:
           text_color = (255,255,255)
        (text_width, text_height),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        x = cx - text_width // 2
        y = cy - text_height // 2
        cv2.putText(img, label, (x, y+30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
        # Desenează contururile
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
    # Salvează imaginea procesată în static/uploads
    save_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
    cv2.imwrite(save_path, img)

    # Fallback GPT-4V (doar dacă există client OpenAI)
    gpt_result = None
    if client:
        try:
            with open(save_path, "rb") as f:
                image_bytes = f.read()
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # sau gpt-4v dacă ai acces
                messages=[
                    {"role": "system", "content": "Ești un profesor care recunoaște forme geometrice din imagini."},
                    {"role": "user", "content": "Numără câte triunghiuri, pătrate și cercuri sunt în această imagine.", "image": image_bytes}
                ]
            )
            gpt_result = response.choices[0].message.content
        except Exception as e:
            gpt_result = f"OpenAI fallback error: {str(e)}"

    return jsonify({
        "shapes": shapes,
        "processed_image": "/static/uploads/result.jpg",
        "gpt_result": gpt_result
    })

if __name__ == "__main__":
    app.run(debug=True, port=8080)