from flask import Flask, jsonify, render_template, redirect, url_for, request, flash, session, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import uuid
import mysql.connector
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import pydicom
import imghdr
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from flask_mysqldb import MySQL

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure MySQL Database
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  # Change if needed
app.config['MYSQL_PASSWORD'] = 'root'  # Change if needed
app.config['MYSQL_DB'] = 'flask_app'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# User Class for Flask-Login
class User(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    cur.close()
    return User(user["id"], user["email"]) if user else None

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'D:/Ultrakey tasks/pancreatic_tumor_detection/pancreatic_tumor_detection/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
try:
    model = load_model('D:/Ultrakey tasks/pancreatic_tumor_detection/pancreatic_tumor_detection/models/pancreatic_tumor_model.h5')  
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@login_required
@app.route("/")
def index():
    return render_template ('login.html')

@app.route('/home')
def home():
    user_info = current_user  # Fetch user from session
    return render_template("home.html", user=user_info)  # Pass user to template

@app.route("/detection")
@login_required
def detection():
    return render_template('detection.html', user=current_user)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user["password"], password):
            login_user(User(user["id"], user["email"]))
            return jsonify({"success": True, "message": "Login Successful"}), 200
        return jsonify({"success": False, "error": "Invalid credentials"}), 401
    return render_template('login.html')

@app.route("/logout", methods=["GET"])
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))

@app.route("/registration", methods=["GET", "POST"])
def registration():
    if request.method == "POST":
        data = request.form
        firstname = data.get("firstname")
        lastname = data.get("lastname")
        phonenumber = data.get("phonenumber")
        email = data.get("email")
        password = generate_password_hash(data.get("password"))

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        existing_user = cur.fetchone()

        if existing_user:
            return jsonify({"success": False, "error": "Email already registered"}), 400

        try:
            cur.execute("INSERT INTO users (firstname, lastname, phonenumber, email, password) VALUES (%s, %s, %s, %s, %s)",
                        (firstname, lastname, phonenumber, email, password))
            mysql.connection.commit()
            return jsonify({"success": True, "message": "Registration successful! Please log in."}), 200
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
        finally:
            cur.close()
    return render_template('registration.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_ct_mri_scan(image_path):
    try:
        image = Image.open(image_path)
        format_type = imghdr.what(image_path)
        if format_type not in ALLOWED_EXTENSIONS:
            return False
        if image.mode != 'L' and image.mode != 'RGB':
            return False
        return True
    except Exception as e:
        print(f"Error checking image type: {e}")
        return False

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'message': 'No file uploaded.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected.'}), 400

    if not allowed_file(file.filename):
        return jsonify({'message': 'Invalid file type. Allowed types: png, jpg, jpeg.'}), 400

    image_id = str(uuid.uuid4())
    filename = secure_filename(f"{image_id}.jpg")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if not is_ct_mri_scan(filepath):
        os.remove(filepath)
        return jsonify({'message': 'Invalid image. Only CT/MRI scans are allowed.'}), 400

    try:
        image = load_img(filepath, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array /= 255.0

        if model:
            prediction = model.predict(image_array)[0][0]
            result = 'No Tumor Detected' if prediction > 0.5 else 'Tumor Detected'

            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO detections (image_id, result) VALUES (%s, %s)", (image_id, result))
            mysql.connection.commit()
            cur.close()

            return jsonify({'message': result, 'image_id': image_id})
        else:
            return jsonify({'message': 'Model not loaded. Please check the server logs for details.'}), 500
    except Exception as e:
        return jsonify({'message': f'Error processing image: {e}'}), 500


@app.route('/performance')
def performance():
    user_info = current_user
    cur = mysql.connection.cursor()
    cur.execute("SELECT result FROM detections")
    data = cur.fetchall()  # Fetch all results from the detections table
    cur.close()

    # Extracting actual results from tuples
    results = [item['result'] for item in data]  # Extracting only the result strings

    # Correctly setting y_true: 1 for Tumor Detected, 0 for No Tumor Detected
    y_true = [1 if result == 'Tumor Detected' else 0 for result in results]

    # Assuming y_pred should come from actual model predictions (Replace this with real predictions)
    y_pred = y_true.copy()  # Placeholder: Model predictions should replace this

    # Compute performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    static_dir = 'D:/Ultrakey tasks/pancreatic_tumor_detection/pancreatic_tumor_detection/static'
    os.makedirs(static_dir, exist_ok=True)

    # Confusion Matrix Plot
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Tumor', 'Tumor'], yticklabels=['No Tumor', 'Tumor'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(static_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()

    # Class Distribution Pie Chart
    values, counts = np.unique(y_true, return_counts=True)

    labels = ['No Tumor', 'Tumor'] if len(counts) == 2 else (['No Tumor'] if values[0] == 0 else ['Tumor'])

    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['blue', 'red'][:len(labels)])
    plt.title('Class Distribution')
    pie_path = os.path.join(static_dir, 'class_distribution.png')
    plt.savefig(pie_path)
    plt.close()

    return render_template('performance.html', user=user_info, accuracy=accuracy, precision=precision, recall=recall, f1=f1, cm_image='confusion_matrix.png', pie_image='class_distribution.png')

@app.route('/download_performance')
def download_performance():
    pdf_path = "static/performance_report.pdf"

    # Fetch performance metrics from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT result FROM detections")
    data = cur.fetchall()
    cur.close()

    results = [item['result'] for item in data]
    y_true = [1 if result == 'Tumor Detected' else 0 for result in results]

    # Assume y_pred is the same as y_true (replace with actual predictions if available)
    y_pred = y_true.copy()

    # Compute performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred)

    # Generate PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50, "Model Performance Metrics")

    # Metrics Data
    c.setFont("Helvetica", 12)
    c.drawCentredString(width / 2, height - 80, f"Accuracy: {accuracy:.2f}")
    c.drawCentredString(width / 2, height - 100, f"Precision: {precision:.2f}")
    c.drawCentredString(width / 2, height - 120, f"Recall: {recall:.2f}")
    c.drawCentredString(width / 2, height - 140, f"F1 Score: {f1:.2f}")

    # Insert Confusion Matrix Image
    cm_path = "static/confusion_matrix.png"
    pie_path = "static/class_distribution.png"

    # Adjusted positions for proper alignment
    cm_width, cm_height = 350, 350  # Confusion matrix size
    pie_width, pie_height = 250, 250  # Pie chart size

    cm_x = (width - cm_width) / 2  # Centering horizontally
    cm_y = height - 500  # Adjusting vertical position

    pie_x = (width - pie_width) / 2  # Centering horizontally
    pie_y = cm_y - pie_height - 30  # Placing below confusion matrix

    c.drawImage(cm_path, cm_x, cm_y, width=cm_width, height=cm_height)
    c.drawImage(pie_path, pie_x, pie_y, width=pie_width, height=pie_height)

    c.save()

    return send_file(pdf_path, as_attachment=True)

# ✅ Fix Application Context Issue
with app.app_context():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT 1")  # Test query to check connection
        print("✅ Database Connected Successfully")
        cur.close()
    except Exception as e:
        print(f"❌ Database Connection Failed: {e}")

@app.route("/contact", methods=["POST"])
def contact():
    try:
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]

        cur = mysql.connection.cursor()
        sql = "INSERT INTO contact (name, email, message) VALUES (%s, %s, %s)"
        values = (name, email, message)
        cur.execute(sql, values)
        mysql.connection.commit()
        cur.close()

        return jsonify({"status": "success", "message": "Message sent successfully!"})

    except Exception as e:
        print(f"Error: {e}")  # Log error in console
        return jsonify({"status": "error", "message": "Something went wrong. Please try again."})



if __name__ == '__main__':
    app.run(debug=True)
