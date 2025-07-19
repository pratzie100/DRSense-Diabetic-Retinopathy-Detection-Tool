import os
import logging
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file, Response
import tensorflow as tf
from groq import Groq
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from io import BytesIO
from flask_cors import CORS
import tempfile
import markdown
from bs4 import BeautifulSoup
import json
import time

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)


# Load model
MODEL_PATH = 'densenet_fundus_clahe_final.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

# Preprocessing functions (unchanged)
def clahe(image, clip_limit=5.0, tile_grid_size=(8, 8)):
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    channels = cv2.split(image)
    clahe_channels = [clahe_obj.apply(channel) for channel in channels]
    return cv2.merge(clahe_channels)

def adjust_brightness_contrast(image, brightness=30, contrast=50):
    image = np.int16(image)
    image = image * (contrast / 127 + 1) - contrast + brightness
    return np.clip(image, 0, 255).astype(np.uint8)

def preprocess_image(image_file, return_image=False):
    try:
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = clahe(image)
        image = adjust_brightness_contrast(image)
        if return_image:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.png', image_rgb)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            return image, base64_image
        image = image / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logging.error(f"Error in preprocess_image: {str(e)}")
        raise

# @app.route('/')
# def serve_index():
#     return send_from_directory('.', 'index.html')


@app.route('/')
def home():
    return jsonify({'message': 'DRSense API is up and running!!'})


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    try:
        _, base64_image = preprocess_image(file, return_image=True)
        return jsonify({'preprocessed_image': f'data:image/png;base64,{base64_image}'})
    except Exception as e:
        logging.error(f"Error in preprocess: {str(e)}")
        return jsonify({'error': 'Preprocessing failed'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    filename = file.filename.rsplit('.', 1)[0] + '.pdf'
    try:
        # Store raw uploaded image
        file.seek(0)
        original_buffer = file.read()
        original_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
        # Preprocessed image
        file.seek(0)
        _, preprocessed_base64 = preprocess_image(file, return_image=True)
        
        # Model prediction
        file.seek(0)
        processed_img = preprocess_image(file)
        pred_probs = model.predict(processed_img)
        pred_class = int(np.argmax(pred_probs, axis=1)[0])
        logging.info(f"Prediction: DR Level {pred_class}")
        
        return jsonify({
            'prediction': pred_class,
            'original_image': f'data:image/png;base64,{original_base64}',
            'preprocessed_image': f'data:image/png;base64,{preprocessed_base64}',
            'filename': filename,
            'disclaimer': 'This tool is for informational purposes only. Consult a healthcare professional for medical advice.'
        })
    except Exception as e:
        logging.error(f"Error in predict: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/stream_advice', methods=['POST'])
def stream_advice():
    data = request.json
    if not data or 'prediction' not in data:
        return Response("Missing prediction", status=400)
    
    pred_class = data['prediction']
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = f"""
    For a patient diagnosed with diabetic retinopathy (DR) level {pred_class}, provide immediate, concise, and actionable recommendations for healthcare professionals, as an expert ophthalmologist would in a hospital setting. Use the following guidelines for each level:
    - Level 0 (No DR): No immediate action; routine monitoring.
    - Level 1 (Mild DR): Refer to ophthalmologist within 6-12 months; control blood sugar.
    - Level 2 (Moderate DR): Urgent referral to retina specialist within 1-3 months; fundus photography.
    - Level 3 (Severe DR): Immediate referral to retina specialist (within weeks); consider laser or anti-VEGF.
    - Level 4 (Proliferative DR): Emergency referral to retina specialist (within days); urgent laser or surgery.
    Provide 3-5 bullet points, ensuring clarity and hospital-grade precision. Format as Markdown, including headers and bold text as needed. Use proper Markdown syntax: ** for bold, * for italics, ``` for code blocks, and - for bullet points. Ensure headings (e.g., ##) and text are separated by newlines. Do not escape Markdown characters (e.g., use **, not \\**).
    """

    def stream_groq():
        try:
            stream = groq_client.chat.completions.create(
                # model="llama3-8b-8192", #depricated
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Wise Assistant. Provide clear, concise, and well-structured responses "
                            "formatted in Markdown. Use proper Markdown syntax: ** for bold, * for italics, "
                            "``` for code blocks, and - for bullet points. "
                            "Ensure headings (e.g., ##) and text are separated by newlines for clarity. "
                            "Do not escape Markdown characters (e.g., use **, not \\**)."
                        )
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=8192,
                top_p=1,
                stream=True,
                stop=None,
            )

            buffer = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    # Clean up malformed Markdown
                    content = content.replace('\\**', '**').replace('\\`', '`').replace('\\\\', '\\')
                    content = content.replace('**\\', '**').replace('** ', '**')
                    buffer += content

                    # Yield complete lines or significant chunks
                    if '\n' in content or len(buffer) > 50:
                        if buffer.strip().startswith('##'):
                            buffer = buffer.strip() + '\n\n'
                        json_data = json.dumps({
                            "choices": [{"delta": {"content": buffer}}]
                        })
                        yield f"data: {json_data}\n\n"
                        buffer = ""
                    # Add a small server-side delay to slow streaming
                    time.sleep(0.05)  # 50ms delay per chunk
            if buffer:
                json_data = json.dumps({
                    "choices": [{"delta": {"content": buffer}}]
                })
                yield f"data: {json_data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_message = str(e)
            logging.error(f"Stream error: {error_message}")
            json_data = json.dumps({"error": error_message})
            yield f"data: {json_data}\n\n"

    return Response(stream_groq(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    })

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    try:
        required_keys = ['original_image', 'preprocessed_image', 'prediction', 'advice', 'filename']
        for key in required_keys:
            if key not in data:
                return jsonify({'error': f'Missing {key}'}), 400
        
        def decode_and_save_image(base64_str, prefix):
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            img_data = base64.b64decode(base64_str)
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({'error': f'Failed to decode {prefix} image'}), 400
            img = cv2.resize(img, (300, 225))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                logging.info(f"Saved {prefix} image to {tmp.name}")
                return tmp.name

        original_path = decode_and_save_image(data['original_image'], "original")
        preprocessed_path = decode_and_save_image(data['preprocessed_image'], "preprocessed")
        prediction = data['prediction']
        advice = data['advice']
        filename = data['filename']

        # Minimal preprocessing to remove only HR/setext markers
        advice_lines = advice.split('\n')
        cleaned_advice = []
        for i, line in enumerate(advice_lines):
            line = line.strip()
            if line and all(c in {'=', '-', '_', '*'} for c in line) and len(line) > 1:
                logging.info(f"Skipping HR/setext marker: {line}")
                continue
            cleaned_advice.append(line)
        advice = '\n'.join(cleaned_advice).strip()

        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.setTitle("DRSense Report")
        
        styles = {
            'h2': ParagraphStyle(name='h2', fontName='Helvetica-Bold', fontSize=14, leading=16, spaceAfter=12),
            'normal': ParagraphStyle(name='normal', fontName='Helvetica', fontSize=10, leading=12, spaceAfter=8),
            'bullet': ParagraphStyle(name='bullet', fontName='Helvetica', fontSize=10, leading=12, leftIndent=20, firstLineIndent=-10, spaceAfter=6),
            'bold': ParagraphStyle(name='bold', fontName='Helvetica-Bold', fontSize=10, leading=12, spaceAfter=4)
        }
        
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, 750, "DRSense - Diabetic Retinopathy Report")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, 730, "Generated by DRSense AI Tool")
        
        pdf.drawString(50, 700, "Original Fundus Image:")
        pdf.drawImage(original_path, 50, 600, width=150, height=112.5)
        
        pdf.drawString(50, 570, "Preprocessed Fundus Image:")
        pdf.drawImage(preprocessed_path, 50, 470, width=150, height=112.5)
        
        levels = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
        pdf.drawString(50, 450, f"Diagnosis: {levels[prediction]} (Level {prediction})")
        
        pdf.drawString(50, 430, "Recommended Actions:")
        y = 410
        
        # Convert Markdown to HTML
        html = markdown.markdown(advice, extensions=['extra'])
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted tags
        for unwanted in soup.find_all(['hr', 'pre', 'code']):
            logging.info(f"Removing unwanted tag: {unwanted}")
            unwanted.decompose()
        
        def parse_element(element, style='normal'):
            nonlocal y
            if element.name == 'h2':
                text = element.get_text().strip()
                if not text:
                    logging.warning(f"Empty h2 tag: {element}")
                    return
                logging.debug(f"Processing h2: {text}")
                p = Paragraph(text, styles['h2'])
            elif element.name == 'p':
                text = element.get_text().strip()
                if not text:
                    logging.warning(f"Empty p tag: {element}")
                    return
                html_text = str(element)
                html_text = html_text.replace('<strong>', '<font name="Helvetica-Bold">').replace('</strong>', '</font>')
                logging.debug(f"Processing p: {html_text}")
                p = Paragraph(html_text, styles[style])
            elif element.name == 'ul':
                li_elements = element.find_all('li', recursive=False)
                for i, li in enumerate(li_elements):
                    text = li.get_text().strip()
                    if not text:
                        logging.warning(f"Empty li tag: {li}")
                        continue
                    # Check if previous element was a bold subheading
                    prev_sibling = li.find_previous_sibling(['p', 'h3', 'h4'])
                    subheading = None
                    if prev_sibling and prev_sibling.get_text().strip() and i < len(li_elements):
                        subheading = prev_sibling.get_text().strip()
                        logging.debug(f"Processing subheading: {subheading}")
                        p = Paragraph(subheading, styles['bold'])
                        p.wrapOn(pdf, 500, 1000)
                        if y - p.height < 50:
                            pdf.showPage()
                            y = 750
                        p.drawOn(pdf, 50, y - p.height)
                        y -= p.height + 4
                    # Render bullet point
                    html_text = str(li).replace('<li>', '').replace('</li>', '')
                    html_text = html_text.replace('<strong>', '<font name="Helvetica-Bold">').replace('</strong>', '</font>')
                    logging.debug(f"Processing li: {html_text}")
                    p = Paragraph(f"• {html_text}", styles['bullet'])
                    p.wrapOn(pdf, 500, 1000)
                    if y - p.height < 50:
                        pdf.showPage()
                        y = 750
                    p.drawOn(pdf, 50, y - p.height)
                    y -= p.height + 6
                return
            else:
                logging.debug(f"Skipping element: {element.name}")
                return
            
            p.wrapOn(pdf, 500, 1000)
            if y - p.height < 50:
                pdf.showPage()
                y = 750
            p.drawOn(pdf, 50, y - p.height)
            y -= p.height + 6
        
        for element in soup.find_all(recursive=False):
            parse_element(element)
        
        pdf.drawString(50, y - 20, "Disclaimer: Consult a healthcare professional for medical advice.")
        
        pdf.showPage()
        pdf.save()
        buffer.seek(0)
        
        pdf_size = buffer.getbuffer().nbytes
        logging.info(f"Generated PDF size: {pdf_size} bytes")
        
        os.remove(original_path)
        os.remove(preprocessed_path)
        
        response = send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        return response
    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        return jsonify({'error': 'Report generation failed'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)