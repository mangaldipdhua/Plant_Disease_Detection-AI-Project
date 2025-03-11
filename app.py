from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import openai
import logging
import requests
from PIL import Image, ImageEnhance, ImageFilter


app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg'}

# MongoDB Connection Setup using PyMongo
MONGO_URI = "mongodb+srv://sappy:2004@cluster0.g6nuw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)

# Get the database
db = client.get_database("feedback_data")  # Access the default database
feedback_collection = db.feedback  # Assuming you have a "feedback" collection

# Ensure MongoDB connection is successful
try:
    client.admin.command('ping')
    print("MongoDB connection successful")
except Exception as e:
    print("MongoDB connection failed:", e)

# Create upload folder if it does not exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained plant disease model
model = tf.keras.models.load_model('cnn_plant_disease_model.keras')

# Allowed file check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Image preprocessing function
def preprocess_image(filepath, target_size=(128, 128)):
    img = image.load_img(filepath, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# List of disease names corresponding to the model's outputs
disease_names = [
    "Apple Apple scab", "Apple Black rot", "Apple Cedar apple rust", "Apple healthy",
    "Blueberry healthy", "Cherry Powdery mildew", "Cherry healthy",
    "Corn Cercospora leaf spot Gray leaf spot", "Corn Common rust", "Corn Northern Leaf Blight", "Corn healthy",
    "Grape Black rot", "Grape Esca (Black Measles)", "Grape Leaf blight (Isariopsis Leaf Spot)", "Grape healthy",
    "Orange Haunglongbing (Citrus greening)", "Peach Bacterial spot", "Peach healthy",
    "Pepper bell Bacterial spot", "Pepper bell healthy", "Potato Early blight", "Potato Late blight", "Potato healthy",
    "Raspberry healthy", "Soybean healthy", "Squash Powdery mildew", "Strawberry Leaf scorch", "Strawberry healthy",
    "Tomato Bacterial spot", "Tomato Early blight", "Tomato Late blight", "Tomato Leaf Mold",
    "Tomato Septoria leaf spot", "Tomato Spider mites Two-spotted spider mite", "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus", "Tomato mosaic virus", "Tomato healthy"
]

# Function to predict the disease from the uploaded image
def predict_disease(filepath):
    img_array = preprocess_image(filepath)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)[0]
    disease = disease_names[class_index]
    return disease

disease_info_dict = {
    "Apple Apple scab": {
        "summary": "Apple scab is a fungal disease caused by Venturia inaequalis, which affects leaves and fruits.",
        "treatments": [
            "Use fungicide sprays like Captan or Mancozeb.",
            "Ensure proper pruning for better air circulation.",
            "Remove and destroy infected leaves and fruit.",
            "Plant resistant apple varieties."
        ]
    },
    "Apple Black rot": {
        "summary": "Black rot is a fungal disease caused by Botryosphaeria obtusa, affecting apple fruits, leaves, and bark.",
        "treatments": [
            "Remove infected branches and debris.",
            "Apply fungicides containing thiophanate-methyl or captan.",
            "Avoid injuries to the tree bark.",
            "Provide adequate spacing for airflow."
        ]
    },
    "Apple Cedar apple rust": {
        "summary": "Cedar apple rust is caused by the fungus Gymnosporangium juniperi-virginianae and requires both cedar and apple trees to complete its lifecycle.",
        "treatments": [
            "Apply fungicides like myclobutanil or propiconazole.",
            "Remove nearby cedar trees or galls if feasible.",
            "Plant resistant apple varieties."
        ]
    },
    "Apple healthy": {
        "summary": "The apple plant is healthy and free from any observable diseases.",
        "treatments": [
            "Maintain proper watering and nutrient levels.",
            "Monitor regularly for early signs of disease.",
            "Use organic or balanced fertilizers."
        ]
    },
    "Blueberry healthy": {
        "summary": "The blueberry plant is healthy and thriving without any diseases.",
        "treatments": [
            "Continue regular watering and fertilization.",
            "Prune to ensure good air circulation.",
            "Monitor for pests or diseases periodically."
        ]
    },
    "Cherry Powdery mildew": {
        "summary": "Powdery mildew is a fungal disease caused by Podosphaera clandestina, forming a white powdery growth on leaves.",
        "treatments": [
            "Apply sulfur-based fungicides.",
            "Ensure proper spacing for air circulation.",
            "Remove and destroy affected leaves."
        ]
    },
    "Cherry healthy": {
        "summary": "The cherry plant is healthy with no signs of disease.",
        "treatments": [
            "Maintain regular care, including watering and fertilization.",
            "Inspect for pests or early signs of disease.",
            "Ensure proper pruning for airflow."
        ]
    },
    "Corn Cercospora leaf spot Gray leaf spot": {
        "summary": "Gray leaf spot is a fungal disease caused by Cercospora zeae-maydis, affecting corn leaves.",
        "treatments": [
            "Apply fungicides like strobilurins or triazoles.",
            "Use crop rotation to prevent buildup of pathogens.",
            "Plant resistant hybrids if available."
        ]
    },
    "Corn Common rust": {
        "summary": "Common rust is a fungal disease caused by Puccinia sorghi, forming rust-colored pustules on leaves.",
        "treatments": [
            "Apply fungicides such as mancozeb or chlorothalonil.",
            "Plant rust-resistant corn varieties.",
            "Promote good air circulation by spacing plants properly."
        ]
    },
    "Corn Northern Leaf Blight": {
        "summary": "Northern Leaf Blight is caused by Exserohilum turcicum, resulting in cigar-shaped lesions on corn leaves.",
        "treatments": [
            "Use resistant corn hybrids.",
            "Apply fungicides like propiconazole or mancozeb.",
            "Practice crop rotation and residue management."
        ]
    },
    "Corn healthy": {
        "summary": "The corn plant is healthy with no observable diseases.",
        "treatments": [
            "Ensure regular irrigation and balanced fertilization.",
            "Monitor for pests and diseases.",
            "Use sustainable farming practices."
        ]
    },
    "Grape Black rot": {
        "summary": "Black rot is a fungal disease caused by Guignardia bidwellii, affecting grape leaves and fruits.",
        "treatments": [
            "Remove and destroy infected leaves and fruits.",
            "Apply fungicides like myclobutanil or mancozeb.",
            "Ensure proper pruning for good air circulation."
        ]
    },
    "Grape Esca (Black Measles)": {
        "summary": "Esca, also known as Black Measles, is a fungal disease caused by Phaeoacremonium species, leading to wood decay and leaf discoloration.",
        "treatments": [
            "Remove infected vines and woody debris.",
            "Avoid over-irrigation to reduce stress on plants.",
            "Use bio-control agents where available."
        ]
    },
    "Grape Leaf blight (Isariopsis Leaf Spot)": {
        "summary": "Leaf blight is caused by Pseudopezicula tracheiphila, forming spots that merge into blighted areas.",
        "treatments": [
            "Apply fungicides like thiophanate-methyl or captan.",
            "Ensure proper vineyard management to reduce humidity.",
            "Remove and destroy infected leaves."
        ]
    },
    "Grape healthy": {
        "summary": "The grapevine is healthy and thriving with no signs of disease.",
        "treatments": [
            "Ensure proper pruning for good airflow.",
            "Monitor regularly for pests or diseases.",
            "Provide balanced fertilization and irrigation."
        ]
    },
    "Orange Haunglongbing (Citrus greening)": {
        "summary": "Citrus greening, or Huanglongbing, is a bacterial disease spread by psyllid insects, causing fruit deformities.",
        "treatments": [
            "Control psyllid populations with insecticides.",
            "Remove and destroy infected trees.",
            "Plant resistant citrus varieties where available."
        ]
    },
    "Peach Bacterial spot": {
        "summary": "Bacterial spot is caused by Xanthomonas campestris, leading to leaf and fruit spotting.",
        "treatments": [
            "Apply copper-based bactericides.",
            "Remove infected leaves and fruits.",
            "Plant resistant peach varieties."
        ]
    },
    "Peach healthy": {
        "summary": "The peach plant is healthy and free from any diseases.",
        "treatments": [
            "Maintain proper irrigation and fertilization.",
            "Inspect regularly for pests and diseases.",
            "Ensure good orchard management practices."
        ]
    },
    "Pepper bell Bacterial spot": {
        "summary": "Bacterial spot is caused by Xanthomonas campestris, affecting pepper leaves and fruits.",
        "treatments": [
            "Apply copper-based bactericides.",
            "Remove and destroy infected plant parts.",
            "Use resistant pepper varieties."
        ]
    },
    "Pepper bell healthy": {
        "summary": "The bell pepper plant is healthy and disease-free.",
        "treatments": [
            "Provide balanced fertilization and irrigation.",
            "Monitor for pests and diseases regularly.",
            "Maintain proper spacing for airflow."
        ]
    },
    "Potato Early blight": {
        "summary": "Early blight, caused by Alternaria solani, forms dark spots on potato leaves.",
        "treatments": [
            "Apply fungicides with chlorothalonil or mancozeb.",
            "Remove and destroy infected plant debris.",
            "Practice crop rotation and use resistant varieties."
        ]
    },
    "Potato Late blight": {
        "summary": "Late blight is caused by Phytophthora infestans, leading to rapid leaf and tuber damage.",
        "treatments": [
            "Apply systemic fungicides like metalaxyl.",
            "Remove infected plants promptly.",
            "Use certified disease-free potato seeds."
        ]
    },
    "Potato healthy": {
        "summary": "The potato plant is healthy and thriving.",
        "treatments": [
            "Maintain proper watering and nutrient management.",
            "Inspect for pests and early signs of disease.",
            "Use crop rotation to prevent disease buildup."
        ]
    },
    "Raspberry healthy": {
        "summary": "The raspberry plant is healthy and flourishing.",
        "treatments": [
            "Ensure balanced fertilization and irrigation.",
            "Inspect for pests and diseases regularly.",
            "Prune old canes to promote new growth."
        ]
    },
    "Soybean healthy": {
        "summary": "The soybean plant is healthy with no signs of diseases.",
        "treatments": [
            "Maintain optimal soil conditions.",
            "Monitor for pests and diseases.",
            "Rotate crops to reduce disease risk."
        ]
    },
    "Squash Powdery mildew": {
        "summary": "Powdery mildew is caused by fungi like Erysiphe cichoracearum, forming a white powdery coating on leaves.",
        "treatments": [
            "Apply sulfur-based or potassium bicarbonate fungicides.",
            "Ensure proper spacing for air circulation.",
            "Avoid overhead watering to reduce humidity."
        ]
    },
    "Strawberry Leaf scorch": {
        "summary": "Leaf scorch is caused by Diplocarpon earlianum, leading to brown spots and drying of leaves.",
        "treatments": [
            "Remove and destroy infected leaves.",
            "Apply fungicides like captan or myclobutanil.",
            "Ensure proper irrigation and fertilization."
        ]
    },
    "Strawberry healthy": {
        "summary": "The strawberry plant is healthy and disease-free.",
        "treatments": [
            "Maintain balanced watering and fertilization.",
            "Inspect for early signs of pests or diseases.",
            "Provide mulching to retain soil moisture."
        ]
    },
    "Tomato Bacterial spot": {
        "summary": "Bacterial spot is caused by Xanthomonas campestris, forming small, dark spots on leaves and fruits.",
        "treatments": [
            "Apply copper-based bactericides.",
            "Remove infected plant parts.",
            "Plant resistant tomato varieties."
        ]
    },
    "Tomato Early blight": {
        "summary": "Early blight is caused by Alternaria solani, forming target-like spots on leaves.",
        "treatments": [
            "Apply fungicides like chlorothalonil.",
            "Remove and destroy infected plant debris.",
            "Use crop rotation and resistant varieties."
        ]
    },
    "Tomato Late blight": {
        "summary": "Late blight is caused by Phytophthora infestans, leading to rapid leaf and fruit damage.",
        "treatments": [
            "Apply fungicides like mancozeb or metalaxyl.",
            "Remove infected plants promptly.",
            "Plant disease-resistant tomato varieties."
        ]
    },
    "Tomato Leaf Mold": {
        "summary": "Leaf mold is caused by Passalora fulva, leading to yellow spots and moldy growth on the undersides of leaves.",
        "treatments": [
            "Apply fungicides like chlorothalonil or copper sprays.",
            "Ensure proper air circulation and reduce humidity.",
            "Remove infected leaves."
        ]
    },
    "Tomato Septoria leaf spot": {
        "summary": "Septoria leaf spot is a fungal disease caused by Septoria lycopersici, forming circular spots with dark margins on leaves.",
        "treatments": [
            "Apply fungicides like chlorothalonil or mancozeb.",
            "Remove infected leaves promptly.",
            "Practice crop rotation to prevent reinfection."
        ]
    },
    "Tomato Spider mites Two-spotted spider mite": {
        "summary": "Two-spotted spider mites are tiny pests that suck sap from tomato leaves, causing yellowing and webbing.",
        "treatments": [
            "Spray with neem oil or insecticidal soap.",
            "Encourage natural predators like ladybugs.",
            "Avoid over-fertilizing, which can attract mites."
        ]
    },
    "Tomato Target Spot": {
        "summary": "Target Spot is caused by Corynespora cassiicola, leading to spots on leaves and fruits.",
        "treatments": [
            "Apply fungicides like azoxystrobin or mancozeb.",
            "Remove and destroy infected plant debris.",
            "Provide proper spacing for airflow."
        ]
    },
    "Tomato Yellow Leaf Curl Virus": {
        "summary": "Yellow Leaf Curl Virus is a viral disease spread by whiteflies, causing leaf curling and stunted growth.",
        "treatments": [
            "Control whiteflies with insecticides or traps.",
            "Remove and destroy infected plants.",
            "Plant resistant tomato varieties."
        ]
    },
    "Tomato mosaic virus": {
        "summary": "Mosaic virus causes mottled patterns on tomato leaves, stunting plant growth.",
        "treatments": [
            "Remove infected plants and sanitize tools.",
            "Use disease-resistant seeds.",
            "Avoid handling plants excessively to prevent spread."
        ]
    },
    "Tomato healthy": {
        "summary": "The tomato plant is healthy with no visible signs of disease.",
        "treatments": [
            "Maintain proper irrigation and nutrient levels.",
            "Inspect regularly for pests and diseases.",
            "Practice crop rotation to prevent soil-borne diseases."
        ]
    }
}

def fetch_disease_info(disease_name):
    disease_info = disease_info_dict.get(disease_name)
    if not disease_info:
        return "Information about this disease is not available."
    
    summary = disease_info.get("summary", "No summary available.")
    treatments = disease_info.get("treatments", [])
    
    # Use HTML tags for formatting
    formatted_info = (
        f"<strong>Disease Summary:</strong><br><br>{summary}<br><br>"
        f"<strong>Recommended Treatments:</strong><br><br>"
        + "<ul>" + "".join([f"<li>{treatment}</li>" for treatment in treatments]) + "</ul>"
    )
    return formatted_info

# def fetch_disease_info(disease_name):
#     try:
#         # Google Generative Language API setup
#         api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyCrHX-k1oOCzvnzBA31V6QbTeTQSY2tMpk"
#         headers = {"Content-Type": "application/json"}
#         data = {
#             "contents": [
#                 {
#                     "parts": [{"text": f"Provide a concise summary about {disease_name} and 4-5 recommended treatments."}]
#                 }
#             ]
#         }

#         # Make a POST request to the Google API
#         response = requests.post(api_url, headers=headers, json=data)

#         if response.status_code == 200:
#             # Parse the API response
#             response_json = response.json()
#             print("API response JSON:", response_json)  # Debug: Print the full API response
            
#             # Extract the text content properly from the nested structure
#             disease_info_raw = (
#                 response_json.get("candidates", [{}])[0]
#                 .get("content", {})
#                 .get("parts", [{}])[0]
#                 .get("text", "No response")
#             )

#             if disease_info_raw == "No response":
#                 return "The API did not provide a response. Please try again."

#             # Remove unwanted symbols and clean up formatting
#             disease_info_raw = disease_info_raw.replace("##", "").replace("*", "").strip()

#             # Split content into summary and treatments sections
#             lines = disease_info_raw.split("\n")
#             summary, cures = [], []
#             in_cures = False

#             for line in lines:
#                 line = line.strip()
#                 if line.lower().startswith("treatments:") or line.lower().startswith("recommended treatments"):
#                     in_cures = True  # Mark the start of cures section
#                     cures.append(line)
#                 elif line:
#                     if in_cures:
#                         cures.append(line)
#                     else:
#                         summary.append(line)

#             # Format the final response with extra line breaks for readability
#             formatted_info = (
#                 f"Disease Summary:\n\n{''.join(summary)}\n\n"
#                 f"Recommended Treatments:\n\n" + "\n\n".join([f"- {cure}" for cure in cures])
#             )

#             # Ensure the text is clean and remove any leftover formatting symbols
#             formatted_info = formatted_info.strip()
            
#             return formatted_info
        
#         else:
#             print(f"Error fetching disease info: {response.status_code}, {response.text}")
#             return "Brief information not available."
    
#     except Exception as e:
#         print(f"Error fetching disease info: {e}")
#         return "Brief information not available."



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict disease based on the uploaded image
        try:
            disease = predict_disease(filepath)
            disease_info = fetch_disease_info(disease)
            
            # Return a JSON response instead of rendering the template
            return {"filename": filename, "disease": disease, "disease_info": disease_info}
        except Exception as e:
            print(f"Prediction error: {e}")
            flash("Failed to process the image. Please try again.")
            return {"error": "Prediction error"}, 500
    else:
        flash('Allowed image types are .jpg and .jpeg')
        return {"error": "Invalid file type"}, 400


@app.route('/result/<filename>')
def result(filename):
    disease = request.args.get('disease', 'Unknown')
    disease_info = request.args.get('disease_info', 'No additional information available')
    return render_template('result.html', filename=filename, disease=disease, disease_info=disease_info)

# Feedback route to display feedback form
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        # Get the form data
        accuracy = request.form.get('accuracy')
        info_helpfulness = request.form.get('info_helpfulness')
        additional_info = request.form.get('additional_info')
        treatment_helpfulness = request.form.get('treatment_helpfulness')
        speed_satisfaction = request.form.get('speed_satisfaction')
        issues = request.form.get('issues')
        additional_feedback = request.form.get('additional_feedback')

        # Save feedback data to MongoDB
        feedback_data = {
            "accuracy": accuracy,
            "info_helpfulness": info_helpfulness,
            "additional_info": additional_info,
            "treatment_helpfulness": treatment_helpfulness,
            "speed_satisfaction": speed_satisfaction,
            "issues": issues,
            "additional_feedback": additional_feedback
        }

        # Insert feedback into the MongoDB collection
        feedback_collection.insert_one(feedback_data)

        # Flash success message
        flash("Feedback submitted successfully!", "success")

        # Redirect to a thank you page or back to feedback form
        return redirect(url_for('feedback'))

    return render_template('feedback.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)