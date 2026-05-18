from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import cv2

app = Flask(__name__)
CORS(app)

model = load_model('mnist_model.h5')

# ─────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────

def preprocess_pixels(image_data):
    """Raw pixel list (784 values) → (1,28,28,1) float32."""
    img = np.array(image_data, dtype='float32').reshape(28, 28) / 255.0
    img = np.clip(img, 0.0, 1.0)

    smoothed = np.copy(img)
    for y in range(28):
        for x in range(28):
            y0, y1 = max(0, y-1), min(28, y+2)
            x0, x1 = max(0, x-1), min(28, x+2)
            smoothed[y, x] = np.mean(img[y0:y1, x0:x1])
    img = np.clip(smoothed, 0.0, 1.0)

    mask = img > 0.05
    if np.any(mask):
        ys, xs = np.where(mask)
        digit = img[ys.min():ys.max()+1, xs.min():xs.max()+1]
        h, w  = digit.shape
        scale = 20.0 / max(h, w)
        nh, nw = max(1, int(round(h*scale))), max(1, int(round(w*scale)))
        digit  = np.array(
            Image.fromarray((digit*255).astype('uint8')).resize((nw, nh), Image.BILINEAR)
        ).astype('float32') / 255.0
        out = np.zeros((28,28), dtype='float32')
        yo, xo = (28-nh)//2, (28-nw)//2
        out[yo:yo+nh, xo:xo+nw] = digit
        img = out
    else:
        img = np.zeros((28,28), dtype='float32')

    return img.reshape(1,28,28,1)


def preprocess_uploaded_image(pil_image):
    """Uploaded single-digit image → (1,28,28,1) float32."""
    img = np.array(pil_image, dtype='float32') / 255.0
    if np.mean(img) > 0.5:
        img = 1.0 - img

    smoothed = np.copy(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            y0, y1 = max(0, y-1), min(img.shape[0], y+2)
            x0, x1 = max(0, x-1), min(img.shape[1], x+2)
            smoothed[y, x] = np.mean(img[y0:y1, x0:x1])
    img = np.clip(smoothed, 0.0, 1.0)

    mask = img > 0.1
    if np.any(mask):
        ys, xs = np.where(mask)
        digit = img[ys.min():ys.max()+1, xs.min():xs.max()+1]
        h, w  = digit.shape
        scale = 20.0 / max(h, w)
        nh, nw = max(1, int(round(h*scale))), max(1, int(round(w*scale)))
        digit  = np.array(
            Image.fromarray((digit*255).astype('uint8')).resize((nw, nh), Image.LANCZOS)
        ).astype('float32') / 255.0
        out = np.zeros((28,28), dtype='float32')
        yo, xo = (28-nh)//2, (28-nw)//2
        out[yo:yo+nh, xo:xo+nw] = digit
        img = out
    else:
        img = np.zeros((28,28), dtype='float32')

    return img.reshape(1,28,28,1)


# ─────────────────────────────────────────────
# Segmentation + recognition
# ─────────────────────────────────────────────

def _roi_to_tensor(roi_bin):
    """Crop binary ROI → padded 28×28 float32 tensor."""
    h, w = roi_bin.shape
    border = max(4, int(max(w, h) * 0.10))
    roi = cv2.copyMakeBorder(roi_bin, border, border, border, border,
                             cv2.BORDER_CONSTANT, value=0)
    rh, rw = roi.shape
    scale = 20.0 / max(rh, rw)
    nh, nw = max(1, int(round(rh*scale))), max(1, int(round(rw*scale)))
    roi = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.float32)
    yo, xo = (28-nh)//2, (28-nw)//2
    canvas[yo:yo+nh, xo:xo+nw] = roi.astype(np.float32) / 255.0
    return canvas.reshape(1, 28, 28, 1)


def segment_and_predict_digits(binary: np.ndarray) -> dict:
    """Segment image, predict digits, return frontend-ready response."""

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"error": "Empty canvas — nothing to recognise", "number": "",
                "digit_count": 0, "avg_confidence": 0.0, "digits": [], "has_low_conf": False}

    symbol_map = {
        0:'0',1:'1',2:'2',3:'3',4:'4',
        5:'5',6:'6',7:'7',8:'8',9:'9',
        10:'+',11:'-',12:'x',13:'/',14:'='
    }

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 2 or h < 2:
            continue
        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    if not boxes:
        return {"error": "No valid symbols detected", "number": "",
                "digit_count": 0, "avg_confidence": 0.0, "digits": [], "has_low_conf": False}

    digits_out = []

    for idx, (x, y, w, h) in enumerate(boxes):
        roi = binary[y:y+h, x:x+w]

        if roi.size == 0 or np.sum(roi) == 0:
            continue

        # Special-case '+' by shape heuristic
        h_ratio = h / w if w != 0 else 0
        if 0.8 < h_ratio < 1.2:
            center_row = roi[roi.shape[0]//2, :]
            center_col = roi[:, roi.shape[1]//2]
            if np.sum(center_row) > 1000 and np.sum(center_col) > 1000:
                digits_out.append({
                    "digit": "+",
                    "confidence": 0.9,
                    "low_conf": False,
                    "top3": [{"digit": "+", "confidence": 0.9}],
                    "bbox": [x, y, w, h]
                })
                continue

        inp  = _roi_to_tensor(roi)
        prob = model.predict(inp, verbose=0)[0]

        pred        = int(np.argmax(prob))
        conf        = float(prob[pred])
        second_pred = int(np.argsort(prob)[-2])
        second_conf = float(prob[second_pred])
        third_pred  = int(np.argsort(prob)[-3])
        third_conf  = float(prob[third_pred])

        symbol = symbol_map[pred]

        # Correction: model confuses 4 and +
        if symbol == '4' and conf < 0.85 and symbol_map[second_pred] == '+':
            symbol = '+'

        top3 = [
            {"digit": symbol_map[pred],        "confidence": round(conf, 4)},
            {"digit": symbol_map[second_pred],  "confidence": round(second_conf, 4)},
            {"digit": symbol_map[third_pred],   "confidence": round(third_conf, 4)},
        ]

        digits_out.append({
            "digit":      symbol,
            "confidence": round(conf, 4),
            "low_conf":   conf < 0.6,
            "top3":       top3,
            "bbox":       [x, y, w, h]
        })

    if not digits_out:
        return {"error": "Could not recognise any symbols", "number": "",
                "digit_count": 0, "avg_confidence": 0.0, "digits": [], "has_low_conf": False}

    number       = "".join(d["digit"] for d in digits_out)
    avg_conf     = float(np.mean([d["confidence"] for d in digits_out]))
    has_low_conf = any(d["low_conf"] for d in digits_out)

    return {
        "error":           None,
        "number":          number,
        "digit_count":     len(digits_out),
        "avg_confidence":  round(avg_conf, 4),
        "digits":          digits_out,
        "has_low_conf":    has_low_conf,
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return send_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    pixels = data.get('image')

    if pixels is None:
        return jsonify({"error": "Missing 'image' field"}), 400
    if not isinstance(pixels, list) or len(pixels) != 28 * 28:
        return jsonify({"error": "The 'image' field must contain exactly 784 pixel values"}), 400

    image      = preprocess_pixels(pixels)
    prediction = model.predict(image, verbose=0)[0]
    top3       = prediction.argsort()[-3:][::-1]
    return jsonify({
        "prediction": int(top3[0]),
        "confidence": float(prediction[top3[0]]),
        "top3": [{"digit": int(i), "confidence": float(prediction[i])} for i in top3],
    })


@app.route('/predict_image', methods=['POST'])
def predict_image():
    file = request.files.get('file')
    if file is None:
        return jsonify({"error": "No file uploaded"})
    pil_image  = Image.open(file).convert('L')
    image      = preprocess_uploaded_image(pil_image)
    if np.sum(image) == 0:
        return jsonify({"error": "Empty image after preprocessing"})
    prediction = model.predict(image, verbose=0)[0]
    top3       = prediction.argsort()[-3:][::-1]
    result     = {
        "prediction": int(top3[0]),
        "confidence": float(prediction[top3[0]]),
        "top3": [{"digit": int(i), "confidence": float(prediction[i])} for i in top3],
    }
    if result["confidence"] < 0.5:
        result["warning"] = "Low confidence prediction"
    return jsonify(result)


@app.route('/predict_multidigit', methods=['POST'])
def predict_multidigit():
    file = request.files.get('file')
    if file is None:
        return jsonify({"error": "No file uploaded", "number": "", "digit_count": 0,
                        "avg_confidence": 0.0, "digits": [], "has_low_conf": False})
    pil_image = Image.open(file).convert('L')
    img = np.array(pil_image, dtype=np.uint8)
    if np.mean(img) > 127:
        img = 255 - img
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.dilate(binary, np.ones((2,2), np.uint8), iterations=1)
    return jsonify(segment_and_predict_digits(binary))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)