# import os
# import logging
# import base64
# import cv2
# import numpy as np
# from flask import Flask, request, jsonify, send_from_directory, send_file
# import tensorflow as tf
# from groq import Groq
# from dotenv import load_dotenv
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader
# from reportlab.platypus import Paragraph
# from reportlab.lib.styles import ParagraphStyle
# from reportlab.lib.enums import TA_LEFT
# import markdown
# from bs4 import BeautifulSoup
# from io import BytesIO
# from flask_cors import CORS

# # Suppress TensorFlow warnings
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# logging.getLogger('absl').setLevel(logging.ERROR)

# # Set up logging
# logging.basicConfig(level=logging.INFO)

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)
# CORS(app)

# # Load model
# MODEL_PATH = 'densenet_fundus_clahe_final.h5'
# try:
#     model = tf.keras.models.load_model(MODEL_PATH)
#     logging.info("Model loaded successfully")
# except Exception as e:
#     logging.error(f"Error loading model: {str(e)}")
#     raise

# # Preprocessing functions (unchanged)
# def clahe(image, clip_limit=5.0, tile_grid_size=(8, 8)):
#     clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     channels = cv2.split(image)
#     clahe_channels = [clahe_obj.apply(channel) for channel in channels]
#     return cv2.merge(clahe_channels)

# def adjust_brightness_contrast(image, brightness=30, contrast=50):
#     image = np.int16(image)
#     image = image * (contrast / 127 + 1) - contrast + brightness
#     return np.clip(image, 0, 255).astype(np.uint8)

# def preprocess_image(image_file, return_image=False):
#     try:
#         image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
#         image = cv2.resize(image, (224, 224))
#         image = clahe(image)
#         image = adjust_brightness_contrast(image)
#         if return_image:
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             _, buffer = cv2.imencode('.png', image_rgb)
#             base64_image = base64.b64encode(buffer).decode('utf-8')
#             return image, base64_image
#         image = image / 255.0
#         return np.expand_dims(image, axis=0)
#     except Exception as e:
#         logging.error(f"Error in preprocess_image: {str(e)}")
#         raise

# @app.route('/')
# def serve_index():
#     return send_from_directory('.', 'index.html')

# @app.route('/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('.', filename)

# @app.route('/preprocess', methods=['POST'])
# def preprocess():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'Empty filename'}), 400
#     try:
#         _, base64_image = preprocess_image(file, return_image=True)
#         return jsonify({'preprocessed_image': f'data:image/png;base64,{base64_image}'})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'Empty filename'}), 400
#     filename = file.filename.rsplit('.', 1)[0] + '.pdf'
#     try:
#         # Store raw uploaded image
#         file.seek(0)
#         original_buffer = file.read()
#         original_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
#         # Preprocessed image
#         file.seek(0)
#         _, preprocessed_base64 = preprocess_image(file, return_image=True)
        
#         # Model prediction
#         file.seek(0)
#         processed_img = preprocess_image(file)
#         pred_probs = model.predict(processed_img)
#         pred_class = int(np.argmax(pred_probs, axis=1)[0])
#         logging.info(f"Prediction: DR Level {pred_class}")
        
#         # Groq advice
#         groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#         prompt = f"""
#         For a patient diagnosed with diabetic retinopathy (DR) level {pred_class}, provide immediate, concise, and actionable recommendations for healthcare professionals, as an expert ophthalmologist would in a hospital setting. Use the following guidelines for each level:
#         - Level 0 (No DR): No immediate action; routine monitoring.
#         - Level 1 (Mild DR): Refer to ophthalmologist within 6-12 months; control blood sugar.
#         - Level 2 (Moderate DR): Urgent referral to retina specialist within 1-3 months; fundus photography.
#         - Level 3 (Severe DR): Immediate referral to retina specialist (within weeks); consider laser or anti-VEGF.
#         - Level 4 (Proliferative DR): Emergency referral to retina specialist (within days); urgent laser or surgery.
#         Provide 3-5 bullet points, ensuring clarity and hospital-grade precision. Format as Markdown, including headers and bold text as needed.
#         Instruction to remember: Do not use alternative markdown syntax (e.g., `====` or `----` underlines).
#         """
#         llm_response = groq_client.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model="llama3-8b-8192"
#         )
#         advice = llm_response.choices[0].message.content
        
#         return jsonify({
#             'prediction': pred_class,
#             'advice': advice,
#             'original_image': f'data:image/png;base64,{original_base64}',
#             'preprocessed_image': f'data:image/png;base64,{preprocessed_base64}',
#             'filename': filename,
#             'disclaimer': 'This tool is for informational purposes only. Consult a healthcare professional for medical advice.'
#         })
#     except Exception as e:
#         logging.error(f"Error in predict route: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/generate_report', methods=['POST'])
# def generate_report():
#     data = request.json
#     try:
#         # Validate required keys
#         required_keys = ['original_image', 'preprocessed_image', 'prediction', 'advice', 'filename']
#         for key in required_keys:
#             if key not in data:
#                 raise ValueError(f"Missing required key: {key}")
        
#         # Decode base64 images
#         def decode_base64_image(base64_str):
#             try:
#                 if ',' in base64_str:
#                     base64_str = base64_str.split(',')[1]
#                 img_data = base64.b64decode(base64_str)
#                 if len(img_data) < 100:
#                     raise ValueError("Base64 image data too small")
#                 return img_data
#             except Exception as e:
#                 logging.error(f"Error decoding base64 image: {str(e)}")
#                 raise ValueError(f"Invalid base64 image data: {str(e)}")

#         original_image_data = decode_base64_image(data['original_image'])
#         preprocessed_image_data = decode_base64_image(data['preprocessed_image'])
#         prediction = data['prediction']
#         advice = data['advice']
#         filename = data['filename']
        
#         # Log input data
#         logging.info(f"Original image size: {len(original_image_data)} bytes")
#         logging.info(f"Preprocessed image size: {len(preprocessed_image_data)} bytes")
#         logging.info(f"Prediction: {prediction}, Filename: {filename}")

#         # Re-encode images as PNG to ensure compatibility
#         def reencode_as_png(image_data):
#             try:
#                 img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
#                 if img is None:
#                     raise ValueError("Failed to decode image for re-encoding")
#                 _, buffer = cv2.imencode('.png', img)
#                 return buffer.tobytes()
#             except Exception as e:
#                 logging.error(f"Error re-encoding image: {str(e)}")
#                 raise ValueError(f"Failed to re-encode image: {str(e)}")

#         original_image_data = reencode_as_png(original_image_data)
#         preprocessed_image_data = reencode_as_png(preprocessed_image_data)

#         buffer = BytesIO()
#         pdf = canvas.Canvas(buffer, pagesize=letter)
#         pdf.setTitle("DRSense Report")
        
#         # Styles
#         styles = {
#             'h1': ParagraphStyle(name='h1', fontName='Helvetica-Bold', fontSize=15, leading=18, spaceAfter=12),
#             'h2': ParagraphStyle(name='h2', fontName='Helvetica-Bold', fontSize=14, leading=16, spaceAfter=10),
#             'h3': ParagraphStyle(name='h3', fontName='Helvetica-Bold', fontSize=12, leading=14, spaceAfter=8),
#             'normal': ParagraphStyle(name='normal', fontName='Helvetica', fontSize=10, leading=12, spaceAfter=6),
#             'bullet': ParagraphStyle(name='bullet', fontName='Helvetica', fontSize=10, leading=12, leftIndent=20, firstLineIndent=-10, spaceAfter=6)
#         }
        
#         # Header
#         pdf.setFont("Helvetica-Bold", 14)
#         pdf.drawString(50, 750, "DRSense - Diabetic Retinopathy Report")
#         pdf.setFont("Helvetica", 10)
#         pdf.drawString(50, 730, "Generated by DRSense AI Tool")
        
#         # Images
#         pdf.drawString(50, 700, "Original Fundus Image:")
#         try:
#             original_reader = ImageReader(BytesIO(original_image_data))
#             pdf.drawImage(original_reader, 50, 600, width=150, height=112.5)
#         except Exception as e:
#             logging.error(f"Error rendering original image: {str(e)}")
#             raise ValueError(f"Failed to render original image: {str(e)}")
        
#         pdf.drawString(50, 570, "Preprocessed Fundus Image:")
#         try:
#             preprocessed_reader = ImageReader(BytesIO(preprocessed_image_data))
#             pdf.drawImage(preprocessed_reader, 50, 470, width=150, height=112.5)
#         except Exception as e:
#             logging.error(f"Error rendering preprocessed image: {str(e)}")
#             raise ValueError(f"Failed to render preprocessed image: {str(e)}")
        
#         # Prediction
#         levels = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
#         pdf.drawString(50, 450, f"Diagnosis: {levels[prediction]} (Level {prediction})")
        
#         # Recommended Actions
#         pdf.drawString(50, 430, "Recommended Actions:")
#         y = 410
        
#         # Parse Markdown
#         html = markdown.markdown(advice, extensions=['extra'])
#         soup = BeautifulSoup(html, 'html.parser')
        
#         def parse_element(element, style='normal'):
#             nonlocal y
#             if element.name == 'h1':
#                 p = Paragraph(element.get_text(), styles['h1'])
#             elif element.name == 'h2':
#                 p = Paragraph(element.get_text(), styles['h2'])
#             elif element.name == 'h3':
#                 p = Paragraph(element.get_text(), styles['h3'])
#             elif element.name == 'p':
#                 text = str(element)
#                 text = text.replace('<strong>', '<font name="Helvetica-Bold">').replace('</strong>', '</font>')
#                 p = Paragraph(text, styles[style])
#             elif element.name == 'ul':
#                 for li in element.find_all('li', recursive=False):
#                     text = str(li)
#                     text = text.replace('<strong>', '<font name="Helvetica-Bold">').replace('</strong>', '</font>')
#                     text = text.replace('<li>', '').replace('</li>', '')
#                     p = Paragraph(f"• {text}", styles['bullet'])
#                     p.wrapOn(pdf, 500, 1000)
#                     if y - p.height < 50:
#                         pdf.showPage()
#                         y = 750
#                     p.drawOn(pdf, 50, y - p.height)
#                     y -= p.height + 6
#                 return
#             else:
#                 return
            
#             p.wrapOn(pdf, 500, 1000)
#             if y - p.height < 50:
#                 pdf.showPage()
#                 y = 750
#             p.drawOn(pdf, 50, y - p.height)
#             y -= p.height + 6
        
#         for element in soup.find_all(recursive=True):
#             parse_element(element)
        
#         pdf.drawString(50, y - 20, "Disclaimer: Consult a healthcare professional for medical advice.")
        
#         pdf.showPage()
#         pdf.save()
#         buffer.seek(0)
        
#         # Validate PDF
#         buffer_size = buffer.getbuffer().nbytes
#         logging.info(f"Generated PDF size: {buffer_size} bytes")
#         if buffer_size < 1000:
#             raise ValueError(f"Generated PDF is too small ({buffer_size} bytes), likely invalid")
        
#         response = send_file(
#             buffer,
#             as_attachment=True,
#             download_name=filename,
#             mimetype='application/pdf'
#         )
#         response.headers['Content-Length'] = buffer_size
#         return response
#     except Exception as e:
#         logging.error(f"Error generating report: {str(e)}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False)


# import os
# import logging
# import base64
# import cv2
# import numpy as np
# from flask import Flask, request, jsonify, send_from_directory, send_file
# import tensorflow as tf
# from groq import Groq
# from dotenv import load_dotenv
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.utils import ImageReader
# from reportlab.platypus import Paragraph
# from reportlab.lib.styles import ParagraphStyle
# from io import BytesIO
# from flask_cors import CORS
# import tempfile

# # Suppress TensorFlow warnings
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# logging.getLogger('absl').setLevel(logging.ERROR)

# # Set up minimal logging
# logging.basicConfig(level=logging.INFO)

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)
# CORS(app)

# # Load model
# MODEL_PATH = 'densenet_fundus_clahe_final.h5'
# try:
#     model = tf.keras.models.load_model(MODEL_PATH)
#     logging.info("Model loaded successfully")
# except Exception as e:
#     logging.error(f"Error loading model: {str(e)}")
#     raise

# # Preprocessing functions (unchanged)
# def clahe(image, clip_limit=5.0, tile_grid_size=(8, 8)):
#     clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     channels = cv2.split(image)
#     clahe_channels = [clahe_obj.apply(channel) for channel in channels]
#     return cv2.merge(clahe_channels)

# def adjust_brightness_contrast(image, brightness=30, contrast=50):
#     image = np.int16(image)
#     image = image * (contrast / 127 + 1) - contrast + brightness
#     return np.clip(image, 0, 255).astype(np.uint8)

# def preprocess_image(image_file, return_image=False):
#     try:
#         image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
#         image = cv2.resize(image, (224, 224))
#         image = clahe(image)
#         image = adjust_brightness_contrast(image)
#         if return_image:
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             _, buffer = cv2.imencode('.png', image_rgb)
#             base64_image = base64.b64encode(buffer).decode('utf-8')
#             return image, base64_image
#         image = image / 255.0
#         return np.expand_dims(image, axis=0)
#     except Exception as e:
#         logging.error(f"Error in preprocess_image: {str(e)}")
#         raise

# @app.route('/')
# def serve_index():
#     return send_from_directory('.', 'index.html')

# @app.route('/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('.', filename)

# @app.route('/preprocess', methods=['POST'])
# def preprocess():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'Empty filename'}), 400
#     try:
#         _, base64_image = preprocess_image(file, return_image=True)
#         return jsonify({'preprocessed_image': f'data:image/png;base64,{base64_image}'})
#     except Exception as e:
#         logging.error(f"Error in preprocess: {str(e)}")
#         return jsonify({'error': 'Preprocessing failed'}), 500

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'Empty filename'}), 400
#     filename = file.filename.rsplit('.', 1)[0] + '.pdf'
#     try:
#         # Store raw uploaded image
#         file.seek(0)
#         original_buffer = file.read()
#         original_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
#         # Preprocessed image
#         file.seek(0)
#         _, preprocessed_base64 = preprocess_image(file, return_image=True)
        
#         # Model prediction
#         file.seek(0)
#         processed_img = preprocess_image(file)
#         pred_probs = model.predict(processed_img)
#         pred_class = int(np.argmax(pred_probs, axis=1)[0])
#         logging.info(f"Prediction: DR Level {pred_class}")
        
#         # Groq advice
#         groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#         prompt = f"""
#         For a patient diagnosed with diabetic retinopathy (DR) level {pred_class}, provide immediate, concise, and actionable recommendations for healthcare professionals, as an expert ophthalmologist would in a hospital setting. Use the following guidelines for each level:
#         - Level 0 (No DR): No immediate action; routine monitoring.
#         - Level 1 (Mild DR): Refer to ophthalmologist within 6-12 months; control blood sugar.
#         - Level 2 (Moderate DR): Urgent referral to retina specialist within 1-3 months; fundus photography.
#         - Level 3 (Severe DR): Immediate referral to retina specialist (within weeks); consider laser or anti-VEGF.
#         - Level 4 (Proliferative DR): Emergency referral to retina specialist (within days); urgent laser or surgery.
#         Provide 3-5 bullet points, ensuring clarity and hospital-grade precision. Format as Markdown, including headers and bold text as needed.
#         Instruction to remember: Do not use alternative markdown syntax (e.g., `====` or `----` underlines).
#         """
#         llm_response = groq_client.chat.completions.create(
#             messages=[{"role": "user", "content": prompt}],
#             model="llama3-8b-8192"
#         )
#         advice = llm_response.choices[0].message.content
        
#         return jsonify({
#             'prediction': pred_class,
#             'advice': advice,
#             'original_image': f'data:image/png;base64,{original_base64}',
#             'preprocessed_image': f'data:image/png;base64,{preprocessed_base64}',
#             'filename': filename,
#             'disclaimer': 'This tool is for informational purposes only. Consult a healthcare professional for medical advice.'
#         })
#     except Exception as e:
#         logging.error(f"Error in predict: {str(e)}")
#         return jsonify({'error': 'Prediction failed'}), 500

# @app.route('/generate_report', methods=['POST'])
# def generate_report():
#     data = request.json
#     try:
#         # Validate required keys
#         required_keys = ['original_image', 'preprocessed_image', 'prediction', 'advice', 'filename']
#         for key in required_keys:
#             if key not in data:
#                 return jsonify({'error': f'Missing {key}'}), 400
        
#         # Decode base64 images and save to disk
#         def decode_and_save_image(base64_str, prefix):
#             if ',' in base64_str:
#                 base64_str = base64_str.split(',')[1]
#             img_data = base64.b64decode(base64_str)
#             img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
#             if img is None:
#                 return jsonify({'error': f'Failed to decode {prefix} image'}), 400
#             with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
#                 cv2.imwrite(tmp.name, img)
#                 logging.info(f"Saved {prefix} image to {tmp.name}")
#                 return tmp.name

#         original_path = decode_and_save_image(data['original_image'], "original")
#         preprocessed_path = decode_and_save_image(data['preprocessed_image'], "preprocessed")
#         prediction = data['prediction']
#         advice = data['advice']
#         filename = data['filename']

#         buffer = BytesIO()
#         pdf = canvas.Canvas(buffer, pagesize=letter)
#         pdf.setTitle("DRSense Report")
        
#         # Styles
#         styles = {
#             'normal': ParagraphStyle(name='normal', fontName='Helvetica', fontSize=10, leading=12, spaceAfter=6),
#             'bullet': ParagraphStyle(name='bullet', fontName='Helvetica', fontSize=10, leading=12, leftIndent=20, firstLineIndent=-10, spaceAfter=6)
#         }
        
#         # Header
#         pdf.setFont("Helvetica-Bold", 14)
#         pdf.drawString(50, 750, "DRSense - Diabetic Retinopathy Report")
#         pdf.setFont("Helvetica", 10)
#         pdf.drawString(50, 730, "Generated by DRSense AI Tool")
        
#         # Images
#         pdf.drawString(50, 700, "Original Fundus Image:")
#         pdf.drawImage(original_path, 50, 600, width=150, height=112.5)
        
#         pdf.drawString(50, 570, "Preprocessed Fundus Image:")
#         pdf.drawImage(preprocessed_path, 50, 470, width=150, height=112.5)
        
#         # Prediction
#         levels = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
#         pdf.drawString(50, 450, f"Diagnosis: {levels[prediction]} (Level {prediction})")
        
#         # Recommended Actions
#         pdf.drawString(50, 430, "Recommended Actions:")
#         y = 410
        
#         # Simplified Markdown parsing
#         for line in advice.split('\n'):
#             if line.strip().startswith('-'):
#                 p = Paragraph(f"• {line.strip()[2:].strip()}", styles['bullet'])
#                 p.wrapOn(pdf, 500, 1000)
#                 if y - p.height < 50:
#                     pdf.showPage()
#                     y = 750
#                 p.drawOn(pdf, 50, y - p.height)
#                 y -= p.height + 6
#             elif line.strip():
#                 p = Paragraph(line.strip(), styles['normal'])
#                 p.wrapOn(pdf, 500, 1000)
#                 if y - p.height < 50:
#                     pdf.showPage()
#                     y = 750
#                 p.drawOn(pdf, 50, y - p.height)
#                 y -= p.height + 6
        
#         pdf.drawString(50, y - 20, "Disclaimer: Consult a healthcare professional for medical advice.")
        
#         pdf.showPage()
#         pdf.save()
#         buffer.seek(0)
        
#         # Clean up temp files
#         os.remove(original_path)
#         os.remove(preprocessed_path)
        
#         response = send_file(
#             buffer,
#             as_attachment=True,
#             download_name=filename,
#             mimetype='application/pdf'
#         )
#         return response
#     except Exception as e:
#         logging.error(f"Error generating report: {str(e)}")
#         return jsonify({'error': 'Report generation failed'}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=False)

import os
import logging
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
import tensorflow as tf
from groq import Groq
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from io import BytesIO
from flask_cors import CORS
import tempfile
import markdown
from bs4 import BeautifulSoup

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)

# Set up minimal logging
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
        
        # Groq advice
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = f"""
        For a patient diagnosed with diabetic retinopathy (DR) level {pred_class}, provide immediate, concise, and actionable recommendations for healthcare professionals, as an expert ophthalmologist would in a hospital setting. Use the following guidelines for each level:
        - Level 0 (No DR): No immediate action; routine monitoring.
        - Level 1 (Mild DR): Refer to ophthalmologist within 6-12 months; control blood sugar.
        - Level 2 (Moderate DR): Urgent referral to retina specialist within 1-3 months; fundus photography.
        - Level 3 (Severe DR): Immediate referral to retina specialist (within weeks); consider laser or anti-VEGF.
        - Level 4 (Proliferative DR): Emergency referral to retina specialist (within days); urgent laser or surgery.
        Provide 3-5 bullet points, ensuring clarity and hospital-grade precision. Format as Markdown, including headers and bold text as needed.
        Instruction to remember: Do not use alternative markdown syntax (e.g., `====` or `----` underlines).
        """
        llm_response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"
        )
        advice = llm_response.choices[0].message.content
        
        return jsonify({
            'prediction': pred_class,
            'advice': advice,
            'original_image': f'data:image/png;base64,{original_base64}',
            'preprocessed_image': f'data:image/png;base64,{preprocessed_base64}',
            'filename': filename,
            'disclaimer': 'This tool is for informational purposes only. Consult a healthcare professional for medical advice.'
        })
    except Exception as e:
        logging.error(f"Error in predict: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    try:
        # Validate required keys
        required_keys = ['original_image', 'preprocessed_image', 'prediction', 'advice', 'filename']
        for key in required_keys:
            if key not in data:
                return jsonify({'error': f'Missing {key}'}), 400
        
        # Decode base64 images, resize, and save to disk
        def decode_and_save_image(base64_str, prefix):
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            img_data = base64.b64decode(base64_str)
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({'error': f'Failed to decode {prefix} image'}), 400
            # Resize to 300x225 (matches PDF display size, reduces file size)
            img = cv2.resize(img, (300, 225))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                # Save with compression
                cv2.imwrite(tmp.name, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                logging.info(f"Saved {prefix} image to {tmp.name}")
                return tmp.name

        original_path = decode_and_save_image(data['original_image'], "original")
        preprocessed_path = decode_and_save_image(data['preprocessed_image'], "preprocessed")
        prediction = data['prediction']
        advice = data['advice']
        filename = data['filename']

        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.setTitle("DRSense Report")
        
        # Styles
        styles = {
            'h2': ParagraphStyle(name='h2', fontName='Helvetica-Bold', fontSize=14, leading=16, spaceAfter=10),
            'normal': ParagraphStyle(name='normal', fontName='Helvetica', fontSize=10, leading=12, spaceAfter=6),
            'bullet': ParagraphStyle(name='bullet', fontName='Helvetica', fontSize=10, leading=12, leftIndent=20, firstLineIndent=-10, spaceAfter=6)
        }
        
        # Header
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, 750, "DRSense - Diabetic Retinopathy Report")
        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, 730, "Generated by DRSense AI Tool")
        
        # Images
        pdf.drawString(50, 700, "Original Fundus Image:")
        pdf.drawImage(original_path, 50, 600, width=150, height=112.5)
        
        pdf.drawString(50, 570, "Preprocessed Fundus Image:")
        pdf.drawImage(preprocessed_path, 50, 470, width=150, height=112.5)
        
        # Prediction
        levels = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
        pdf.drawString(50, 450, f"Diagnosis: {levels[prediction]} (Level {prediction})")
        
        # Recommended Actions
        pdf.drawString(50, 430, "Recommended Actions:")
        y = 410
        
        # Parse Markdown
        html = markdown.markdown(advice, extensions=['extra'])
        soup = BeautifulSoup(html, 'html.parser')
        
        def parse_element(element, style='normal'):
            nonlocal y
            if element.name == 'h2':
                p = Paragraph(element.get_text(), styles['h2'])
            elif element.name == 'p':
                text = str(element)
                text = text.replace('<strong>', '<font name="Helvetica-Bold">').replace('</strong>', '</font>')
                p = Paragraph(text, styles[style])
            elif element.name == 'ul':
                for li in element.find_all('li', recursive=False):
                    text = str(li)
                    text = text.replace('<strong>', '<font name="Helvetica-Bold">').replace('</strong>', '</font>')
                    text = text.replace('<li>', '').replace('</li>', '')
                    p = Paragraph(f"• {text}", styles['bullet'])
                    p.wrapOn(pdf, 500, 1000)
                    if y - p.height < 50:
                        pdf.showPage()
                        y = 750
                    p.drawOn(pdf, 50, y - p.height)
                    y -= p.height + 6
                return
            else:
                return
            
            p.wrapOn(pdf, 500, 1000)
            if y - p.height < 50:
                pdf.showPage()
                y = 750
            p.drawOn(pdf, 50, y - p.height)
            y -= p.height + 6
        
        for element in soup.find_all(recursive=True):
            parse_element(element)
        
        pdf.drawString(50, y - 20, "Disclaimer: Consult a healthcare professional for medical advice.")
        
        pdf.showPage()
        pdf.save()
        buffer.seek(0)
        
        # Log PDF size
        pdf_size = buffer.getbuffer().nbytes
        logging.info(f"Generated PDF size: {pdf_size} bytes")
        
        # Clean up temp files
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
    app.run(host='0.0.0.0', port=5000, debug=False)