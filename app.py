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

    # file.read() -> Citește tot conținutul fișierului în memorie, sub formă de bytes (un șir binar, adică 0 și 1). file.read()
    # np.frombuffer(..., np.uint8) -> Transformă acei bytes într-un vector NumPy de tip uint8 (numere întregi între 0 și 255). Acum avem o reprezentare numerică a datelor imaginii, dar încă nu este o imagine 2D — doar o listă lungă de numere.
    # cv2.imdecode(..., cv2.IMREAD_COLOR) -> OpenCV ia vectorul de date binare (array-ul de mai sus) și îl decodează într-o imagine color (matrice 3D de pixeli).
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Fallback OpenCV Transformă poza color într-o poză alb-negru
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.threshold() compară fiecare pixel din imagine cu o valoare numită prag (threshold).
    #  gray – imaginea de intrare (alb-negru)
    # 127 – pragul de tăiere dacă valoarea < 127 →  pixel devine 0 (negru) altfel →  pixel devine 255 (alb)
    # 255 – valoarea de atribuit pixelilor mai mari decât pragul
    # cv2.THRESH_BINARY – tipul de prag (binar simplu)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # cv2.RETR_EXTERNAL — modul de extragere a contururilor.
    # RETR_EXTERNAL → ia doar contururile exterioare (nu și cele interioare, cum ar fi găurile din litera „O”).
    # cv2.CHAIN_APPROX_SIMPLE — metoda de simplificare a conturului.
    # Reduce numărul de puncte din contur, păstrând forma generală.
    # Fără asta (cv2.CHAIN_APPROX_NONE), conturul ar avea fiecare pixel inclus (mult mai mare și inutil).

    # contours  → o listă de contururi, unde fiecare contur este un array de puncte (coordonate x,y).
    # _         → o structură de ierarhie (nu ne interesează aici).
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Creează un dicționar Python pentru a număra fiecare tip de formă.
    shapes = {"triangle": 0, "square": 0, "circle": 0}
    
    # contours este lista tuturor formelor detectate în imagine (obținută cu cv2.findContours).
    # Fiecare cnt (contour) este o listă de coordonate (puncte x,y) care descriu marginea unei forme.
    # cv2.approxPolyDP() simplifică forma conturului pentru a afla câte „colțuri” are.
    # cv2.arcLength(cnt, True) – lungimea perimetrului conturului.
    # 0.04 * ... – precizia (toleranța). Cu cât valoarea este mai mică, cu atât forma este mai exactă. 0.04 înseamnă o toleranță de 4% din lungimea totală.
    # True – indică faptul că forma este închisă (nu o linie deschisă).
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            shapes["triangle"] += 1
        elif len(approx) == 4:
            shapes["square"] += 1
        else:
            shapes["circle"] += 1
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