from flask import Flask, request, render_template
from ultralytics import YOLO
from geopy.geocoders import Nominatim
import os, uuid, cv2
import time

app = Flask(__name__)
# Disable static file caching during development so browsers pick up CSS changes immediately
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# Cache-bust static assets in development by adding a version query param
app.config['STATIC_VERSION'] = int(time.time())
@app.context_processor
def inject_static_version():
    return dict(static_version=app.config['STATIC_VERSION'])

model = YOLO("pothole_guard.onnx")

# Use absolute paths to ensure it works on Windows and Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

geolocator = Nominatim(user_agent="air_pothole_v8_final")

@app.route("/")
def home(): return render_template("index.html")

@app.route("/history")
def history(): return render_template("history.html")

@app.route("/complaint")
def complaint(): return render_template("complaint.html")

@app.route("/last_result")
def last_result(): return render_template("last_result.html")

@app.route("/detect_multiple", methods=["POST"])
def detect_multiple():
    files = request.files.getlist("images")
    lat, lon = request.form.get("lat"), request.form.get("lon")
    all_results, total_potholes = [], 0

    for file in files:
        if not file: continue
        unique_id = uuid.uuid4().hex
        ext = file.filename.split('.')[-1].lower()
        path = os.path.normpath(os.path.join(UPLOAD_FOLDER, f"{unique_id}.{ext}"))
        file.save(path)

        if ext in ['mp4', 'mov', 'avi']:
            cap = cv2.VideoCapture(path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
                    tmp_p = os.path.join(UPLOAD_FOLDER, "frame.jpg")
                    cv2.imwrite(tmp_p, frame)
                    res = model(tmp_p, conf=0.25)
                    total_potholes += len(res[0].boxes)
                    out_name = f"v_{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(os.path.join(RESULT_FOLDER, out_name), res[0].plot())
                    all_results.append(out_name)
            cap.release()
        else:
            res = model(path, conf=0.25)
            total_potholes += len(res[0].boxes)
            out_name = f"r_{unique_id}.jpg"
            cv2.imwrite(os.path.join(RESULT_FOLDER, out_name), res[0].plot())
            all_results.append(out_name)

    lat = request.form.get("lat") or ""
    lon = request.form.get("lon") or ""
    lat = str(lat).strip() if lat else ""
    lon = str(lon).strip() if lon else ""

    # Default location so map and address always show (no permission needed)
    DEFAULT_LAT, DEFAULT_LON = "17.3850", "78.4867"
    if not lat or not lon:
        lat, lon = DEFAULT_LAT, DEFAULT_LON

    address = "Location Found"
    try:
        loc = geolocator.reverse("%s,%s" % (lat, lon), timeout=5)
        address = loc.address if loc else "%s, %s" % (lat, lon)
    except Exception:
        address = "%s, %s" % (lat, lon)

    return render_template("result.html", potholes=total_potholes, address=address, lat=lat, lon=lon, has_location=True, images=all_results)


@app.route('/prepare_ghmc', methods=['POST'])
def prepare_ghmc():
    """Save complaint payload to ghmc_payload.json (includes absolute image paths)."""
    data = request.get_json() or {}
    imgs = data.get('all_images', [])
    image_paths = [os.path.join(RESULT_FOLDER, img) for img in imgs]
    payload = {
        'potholes': data.get('potholes'),
        'address': data.get('address'),
        'lat': data.get('lat'),
        'lon': data.get('lon'),
        'message': data.get('message'),
        'date': data.get('date'),
        'images': image_paths
    }
    pfile = os.path.join(BASE_DIR, 'ghmc_payload.json')
    import json
    with open(pfile, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {'status': 'ok', 'path': pfile}





def _get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "YOUR_IP"

if __name__ == "__main__":
    import sys
    port = int(os.environ.get("PORT", 8000))
    use_https = os.environ.get("USE_HTTPS", "").lower() in ("1", "true", "yes") or "--https" in sys.argv
    cert_file = os.path.join(BASE_DIR, "cert.pem")
    key_file = os.path.join(BASE_DIR, "key.pem")
    ssl_ctx = None
    if use_https and os.path.isfile(cert_file) and os.path.isfile(key_file):
        import ssl
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(cert_file, key_file)
        port = 8443
    my_ip = _get_local_ip()
    print("\n" + "="*50)
    print("AIR Pothole Detector")
    print("="*50)
    if ssl_ctx:
        print("Local:  https://127.0.0.1:%s" % port)
        print("Mobile: https://%s:%s" % (my_ip, port))
    else:
        print("Local:  http://127.0.0.1:%s" % port)
        print("Mobile: http://%s:%s" % (my_ip, port))
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=port, ssl_context=ssl_ctx, use_reloader=False) 