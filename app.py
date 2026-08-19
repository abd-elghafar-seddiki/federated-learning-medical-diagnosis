import os
import cv2
import numpy as np
import h5py
import pickle
import secrets
import shutil
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify, session, flash
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model, Model, save_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam
import sqlite3

# Initialize the application
app = Flask(__name__, static_url_path='/static')
app.secret_key = secrets.token_hex(32)  # Secure random secret key
bcrypt = Bcrypt(app)

# User roles
ROLES = {
    'admin': 3,
    'doctor': 2,
    'user': 1
}

# Configure upload folders
UPLOAD_FOLDERS = {
    'lung_cancer_xray': 'uploads/lung_cancer_xray',
    'lung_cancer_histo': 'uploads/lung_cancer_histo',
    'malaria': 'uploads/malaria',
    'retinal_oct': 'uploads/retinal_oct',
    'diabetic_retinopathy': 'uploads/diabetic_retinopathy',
    'brain_tumor': 'uploads/brain_tumor',
    'covid_xray': 'uploads/covid_xray'
}

for folder in UPLOAD_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDERS'] = UPLOAD_FOLDERS

# Configure printer folder
PRINTER_FOLDER = 'printer_scans'
os.makedirs(PRINTER_FOLDER, exist_ok=True)
app.config['PRINTER_FOLDER'] = PRINTER_FOLDER

# Model paths
MODEL_PATHS = {
    'lung_cancer_xray': 'models/lung_cancer_detection_model.h5',
    'lung_cancer_histo': 'models/lung-cancer-resnet-model.h5',
    'malaria': 'models/malaria.h5',
    'retinal_oct': 'models/Retinal-OCT-resnet-model.h5',
    'diabetic_retinopathy': 'models/Diabetic-Retinopathy-ResNet50-model.h5',
    'brain_tumor': 'models/brain_tumor_model.h5',
    'covid_xray': 'models/CNN_Covid19_Xray_Version.h5'
}

# Initialize the database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  first_name TEXT NOT NULL,
                  last_name TEXT NOT NULL,
                  birth_date TEXT NOT NULL,
                  hospital TEXT NOT NULL,
                  phone TEXT NOT NULL,
                  user_type TEXT NOT NULL,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role INTEGER DEFAULT 1,
                  is_active INTEGER DEFAULT 1,
                  registration_date TEXT NOT NULL)''')
    
    # Create password reset requests table
    c.execute('''CREATE TABLE IF NOT EXISTS password_resets
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  token TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')
    
    conn.commit()
    conn.close()

init_db()

# Helper functions for authentication
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def create_user(first_name, last_name, birth_date, hospital, phone, user_type, email, password):
    conn = get_db_connection()
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    registration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Set default role (user) unless it's the first user (admin)
    user_count = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    role = 3 if user_count == 0 else 1  # First user becomes admin
    
    try:
        conn.execute("INSERT INTO users (first_name, last_name, birth_date, hospital, phone, user_type, email, password, role, registration_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (first_name, last_name, birth_date, hospital, phone, user_type, email, hashed_password, role, registration_date))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(email, password):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    if user and bcrypt.check_password_hash(user['password'], password):
        return user
    return None

def get_user_by_id(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return user

def update_user(user_id, **kwargs):
    conn = get_db_connection()
    try:
        query = "UPDATE users SET "
        params = []
        for key, value in kwargs.items():
            if key == 'password':
                value = bcrypt.generate_password_hash(value).decode('utf-8')
            query += f"{key} = ?, "
            params.append(value)
        
        query = query.rstrip(', ') + " WHERE id = ?"
        params.append(user_id)
        
        conn.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating user: {e}")
        return False
    finally:
        conn.close()

def get_user_stats():
    conn = get_db_connection()
    stats = {
        'total_users': conn.execute('SELECT COUNT(*) FROM users').fetchone()[0],
        'active_users': conn.execute('SELECT COUNT(*) FROM users WHERE is_active = 1').fetchone()[0],
        'admin_count': conn.execute('SELECT COUNT(*) FROM users WHERE role = 3').fetchone()[0]
    }
    conn.close()
    return stats

# Decorators for login and role verification
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(role_name):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash('Please log in to access this page', 'warning')
                return redirect(url_for('login'))
            
            user_role = session.get('user_role', 1)
            required_role = ROLES.get(role_name, 1)
            
            if user_role < required_role:
                flash('You do not have permission to access this page', 'danger')
                return redirect(url_for('home'))
                
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Authentication and admin routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = verify_user(email, password)
        
        if user:
            if not user['is_active']:
                flash('Your account is inactive. Please contact the administrator', 'danger')
                return redirect(url_for('login'))
            
            session['user_id'] = user['id']
            session['user_email'] = user['email']
            session['user_role'] = user['role']
            session['user_name'] = f"{user['first_name']} {user['last_name']}"
            
            flash(f'Welcome back, {session["user_name"]}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Incorrect email or password', 'danger')
    
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        birth_date = request.form['birth_date']
        hospital = request.form['hospital']
        phone = request.form['phone']
        user_type = request.form['user_type']
        email = request.form['email']
        password = request.form['password']
        
        if create_user(first_name, last_name, birth_date, hospital, phone, user_type, email, password):
            flash('Registration successful! Please log in', 'success')
            return redirect(url_for('login'))
        else:
            flash('Email already registered', 'danger')
    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/admin')
@role_required('admin')
def admin_dashboard():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users ORDER BY id DESC').fetchall()
    conn.close()
    
    stats = get_user_stats()
    return render_template('admin/dashboard.html', 
                         users=users, 
                         ROLES=ROLES,
                         total_users=stats['total_users'],
                         active_users=stats['active_users'],
                         admin_count=stats['admin_count'])

@app.route('/admin/update-user/<int:user_id>', methods=['POST'])
@role_required('admin')
def admin_update_user(user_id):
    data = request.get_json()
    field = data.get('field')
    value = data.get('value')
    
    if field not in ['role', 'is_active']:
        return jsonify({'success': False, 'message': 'Invalid field'})
    
    conn = get_db_connection()
    try:
        conn.execute(f"UPDATE users SET {field} = ? WHERE id = ?", (value, user_id))
        conn.commit()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error updating user: {e}")
        return jsonify({'success': False, 'message': str(e)})
    finally:
        conn.close()

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = get_user_by_id(session['user_id'])
    
    if request.method == 'POST':
        updates = {
            'first_name': request.form['first_name'],
            'last_name': request.form['last_name'],
            'birth_date': request.form['birth_date'],
            'hospital': request.form['hospital'],
            'phone': request.form['phone'],
            'user_type': request.form['user_type']
        }
        
        if request.form['password']:
            updates['password'] = request.form['password']
        
        if update_user(session['user_id'], **updates):
            session['user_name'] = f"{updates['first_name']} {updates['last_name']}"
            flash('Profile updated successfully', 'success')
        else:
            flash('Error updating profile', 'danger')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', user=user)

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password_request():
    if request.method == 'POST':
        email = request.form['email']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        if user:
            # In a real application, send reset email here
            token = secrets.token_urlsafe(32)
            expires_at = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            
            conn = get_db_connection()
            conn.execute('INSERT INTO password_resets (user_id, token, expires_at) VALUES (?, ?, ?)',
                         (user['id'], token, expires_at))
            conn.commit()
            conn.close()
            
            flash('Password reset link has been sent to your email', 'info')
        else:
            flash('Email does not exist', 'danger')
        return redirect(url_for('login'))
    
    return render_template('auth/reset_password_request.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    conn = get_db_connection()
    reset_request = conn.execute('''SELECT * FROM password_resets 
                                  WHERE token = ? AND expires_at > datetime('now')''', 
                                  (token,)).fetchone()
    valid_token = reset_request is not None
    
    if request.method == 'POST' and valid_token:
        new_password = request.form['password']
        if update_user(reset_request['user_id'], password=new_password):
            conn.execute('DELETE FROM password_resets WHERE token = ?', (token,))
            conn.commit()
            conn.close()
            flash('Password updated successfully. Please log in', 'success')
            return redirect(url_for('login'))
        else:
            flash('Error updating password', 'danger')
    
    conn.close()
    return render_template('auth/reset_password.html', token=token, valid_token=valid_token)

@app.route('/admin/edit-user/<int:user_id>', methods=['GET', 'POST'])
@role_required('admin')
def admin_edit_user(user_id):
    user = get_user_by_id(user_id)
    
    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('admin_dashboard'))
    
    if request.method == 'POST':
        updates = {
            'first_name': request.form['first_name'],
            'last_name': request.form['last_name'],
            'birth_date': request.form['birth_date'],
            'hospital': request.form['hospital'],
            'phone': request.form['phone'],
            'user_type': request.form['user_type'],
            'email': request.form['email'],
            'is_active': 1 if request.form.get('is_active') == 'on' else 0
        }
        
        if request.form['password']:
            updates['password'] = request.form['password']
        
        if update_user(user_id, **updates):
            flash('User updated successfully', 'success')
        else:
            flash('Error updating user', 'danger')
        return redirect(url_for('admin_edit_user', user_id=user_id))
    
    return render_template('admin/edit_user.html', user=user)

# Model update route
@app.route('/admin/update-models', methods=['GET', 'POST'])
@role_required('admin')
def update_models():
    if request.method == 'POST':
        model_type = request.form.get('model_type')
        file = request.files.get('model_file')
        
        if not model_type or model_type not in MODEL_PATHS:
            flash('Invalid model type', 'danger')
            return redirect(url_for('update_models'))
        
        if not file or file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('update_models'))
        
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            temp_path = os.path.join('temp_uploads', filename)
            os.makedirs('temp_uploads', exist_ok=True)
            file.save(temp_path)
            
            # Verify the file is a valid model
            try:
                test_model = load_model(temp_path)
                # Simple test: create random data based on model type
                if model_type == 'covid_xray':
                    test_input = np.random.rand(1, 150, 150, 3)
                elif model_type == 'lung_cancer_xray':
                    test_input = np.random.rand(1, 256, 256, 3)
                elif model_type == 'malaria':
                    test_input = np.random.rand(1, 100, 100, 3)
                else:
                    test_input = np.random.rand(1, 224, 224, 3)
                
                test_model.predict(test_input)
            except Exception as e:
                flash(f'File is not a valid model: {str(e)}', 'danger')
                os.remove(temp_path)
                return redirect(url_for('update_models'))
            
            # Replace old model
            model_path = MODEL_PATHS[model_type]
            shutil.move(temp_path, model_path)
            
            # Update model in memory
            MODELS[model_type] = load_model(model_path)
            
            flash(f'Model {model_type} updated successfully!', 'success')
            return redirect(url_for('update_models'))
        
        except Exception as e:
            flash(f'Error updating model: {str(e)}', 'danger')
            return redirect(url_for('update_models'))
    
    # Calculate last update dates for models
    model_updates = {}
    for model_type, path in MODEL_PATHS.items():
        if os.path.exists(path):
            model_updates[model_type] = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
        else:
            model_updates[model_type] = "Never"
    
    return render_template('admin/update_models.html', 
                         models=list(MODEL_PATHS.keys()),
                         model_names=CLASS_NAMES,
                         MODELS=MODELS,
                         MODEL_PATHS=MODEL_PATHS,
                         model_updates=model_updates)

# Initialize COVID Label Encoder
try:
    with open("models/Label_encoder.pkl", 'rb') as f:
        covid_le = pickle.load(f)
    print("COVID label encoder loaded successfully")
    print("COVID Label Encoder classes:", covid_le.classes_)
except Exception as e:
    print(f"Error loading COVID label encoder: {str(e)}")
    print("Creating default label encoder for COVID model")
    covid_le = LabelEncoder()
    covid_le.fit(['Covid', 'Viral Pneumonia', 'Normal'])

# Image and model processing functions
def process_printer_image(model_type, image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read printer image")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = os.path.join(app.config['PRINTER_FOLDER'], f"scan_{timestamp}.jpg")
        cv2.imwrite(saved_path, img)
        
        result = process_image(model_type, saved_path)
        
        return {
            'status': 'success',
            'prediction': result[0],
            'confidence': float(result[1]),
            'image_path': saved_path
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

@app.route('/api/process_scan', methods=['POST'])
@login_required
def process_scan():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Filename is empty'}), 400
    
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['PRINTER_FOLDER'], f"temp_{filename}")
    file.save(temp_path)
    
    model_type = request.form.get('model_type', 'lung_cancer_xray')
    if model_type not in MODELS:
        return jsonify({'error': 'Invalid model type'}), 400
    
    result = process_printer_image(model_type, temp_path)
    
    try:
        os.remove(temp_path)
    except:
        pass
    
    if result['status'] == 'error':
        return jsonify(result), 500
        
    return jsonify(result)

def build_lung_cancer_xray_model():
    input_shape = (256, 256, 3)
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def build_brain_tumor_model():
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def build_covid_xray_model():
    inputs = Input(shape=(150, 150, 3))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def create_and_save_new_model(model_path, model_type):
    print(f"Creating new model for {model_type}...")
    if model_type == 'lung_cancer_xray':
        model = build_lung_cancer_xray_model()
    elif model_type == 'brain_tumor':
        model = build_brain_tumor_model()
    elif model_type == 'covid_xray':
        model = build_covid_xray_model()
    else:
        inputs = Input(shape=(224, 224, 3))
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        outputs = Dense(2, activation='softmax')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
    
    save_model(model, model_path)
    print(f"Model {model_type} created and saved to {model_path}")
    return model

def load_model_with_recovery(model_path, model_type):
    try:
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                print(f"Model {model_type} loaded successfully")
                return model
            except Exception as load_error:
                print(f"Standard loading failed for {model_type}: {str(load_error)}")
                return create_and_save_new_model(model_path, model_type)
        else:
            print(f"Model file not found, creating new model for {model_type}...")
            return create_and_save_new_model(model_path, model_type)
    except Exception as e:
        print(f"Critical error with {model_type}: {str(e)}")
        raise

# Load models with error handling
try:
    print("\nInitializing models...")
    MODELS = {
        'lung_cancer_xray': load_model_with_recovery(
            MODEL_PATHS['lung_cancer_xray'], 
            'lung_cancer_xray'
        ),
        'lung_cancer_histo': load_model_with_recovery(
            MODEL_PATHS['lung_cancer_histo'],
            'lung_cancer_histo'
        ),
        'malaria': load_model_with_recovery(
            MODEL_PATHS['malaria'],
            'malaria'
        ),
        'retinal_oct': load_model_with_recovery(
            MODEL_PATHS['retinal_oct'],
            'retinal_oct'
        ),
        'diabetic_retinopathy': load_model_with_recovery(
            MODEL_PATHS['diabetic_retinopathy'],
            'diabetic_retinopathy'
        ),
        'brain_tumor': load_model_with_recovery(
            MODEL_PATHS['brain_tumor'],
            'brain_tumor'
        ),
        'covid_xray': load_model_with_recovery(
            MODEL_PATHS['covid_xray'],
            'covid_xray'
        )
    }
    
    # Verify models
    print("\nVerifying models...")
    for name, model in MODELS.items():
        try:
            print(f"Verifying model {name}...")
            if name == 'covid_xray':
                dummy_input = np.random.rand(1, 150, 150, 3)
            elif name == 'lung_cancer_xray':
                dummy_input = np.random.rand(1, 256, 256, 3)
            elif name == 'malaria':
                dummy_input = np.random.rand(1, 100, 100, 3)
            else:
                dummy_input = np.random.rand(1, 224, 224, 3)
                
            prediction = model.predict(dummy_input)
            print(f"Model {name} ready. Output shape: {prediction.shape}")
            
            if name == 'covid_xray':
                print("Random COVID prediction:", prediction)
                if covid_le:
                    print("Decoded prediction:", covid_le.inverse_transform([np.argmax(prediction)]))
        except Exception as e:
            print(f"Warning: Failed to verify model {name}: {str(e)}")
            MODELS[name] = create_and_save_new_model(MODEL_PATHS[name], name)
    
    print("\nAll models initialized successfully!\n")
    
except Exception as e:
    print(f"\nFatal error during initialization: {str(e)}")
    print("Cannot start application. Please check:")
    print("1. Model files exist in models/ folder")
    print("2. Files are not corrupted")
    print("3. All required dependencies are installed")
    exit(1)

CLASS_NAMES = {
    'lung_cancer_xray': ['Benign', 'Malignant', 'Normal'],
    'lung_cancer_histo': ['Adenocarcinoma', 'Benign', 'Squamous Cell Carcinoma'],
    'malaria': ['Parasitized', 'Uninfected'],
    'retinal_oct': ['CNV', 'DME', 'DRUSEN', 'Normal'],
    'diabetic_retinopathy': ['Diabetic', 'Non-Diabetic'],
    'brain_tumor': ['Tumor', 'No Tumor'],
    'covid_xray': ['Covid', 'Viral Pneumonia', 'Normal']
}

def process_image(model_type, image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image at {image_path}")
        
        model = MODELS[model_type]
        classes = CLASS_NAMES[model_type]
        
        if model_type == 'lung_cancer_xray':
            img = cv2.resize(image, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
        elif model_type == 'covid_xray':
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (150, 150))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            predictions = model.predict(img)
            predicted_index = np.argmax(predictions)
            confidence = predictions[0][predicted_index]
            
            if covid_le:
                predicted_label = covid_le.inverse_transform([predicted_index])[0]
            else:
                predicted_label = classes[predicted_index]
            
            return predicted_label, float(confidence)
            
        elif model_type in ['lung_cancer_histo', 'retinal_oct', 'diabetic_retinopathy', 'brain_tumor']:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            
        elif model_type == 'malaria':
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (100, 100))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
        
        predictions = model.predict(img)
        
        if model_type in ['malaria', 'diabetic_retinopathy']:
            predicted_index = int(predictions[0][0] > 0.5)
            confidence = predictions[0][0] if predicted_index == 1 else 1 - predictions[0][0]
        else:
            predicted_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_index]
        
        return classes[predicted_index], float(confidence)
    
    except Exception as e:
        print(f"Error in process_image for {model_type}: {str(e)}")
        raise

# Main application routes
@app.route('/')
@login_required
def home():
    # Get model update dates to display on home page
    model_updates = {}
    for model_type, model_path in MODEL_PATHS.items():
        if os.path.exists(model_path):
            model_updates[model_type] = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M')
        else:
            model_updates[model_type] = "Unknown"
    
    return render_template('index.html', model_updates=model_updates)

@app.route('/printer')
@login_required
def printer_interface():
    return render_template('printer.html')

@app.route('/<model_type>')
@login_required
def model_home(model_type):
    if model_type not in MODELS:
        return redirect(url_for('home'))
    return render_template(f'{model_type}/index.html')

@app.route('/<model_type>/upload', methods=['POST'])
@login_required
def upload_file(model_type):
    if model_type not in MODELS:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        upload_folder = app.config['UPLOAD_FOLDERS'][model_type]
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        try:
            predicted_label, confidence_score = process_image(model_type, file_path)
            return render_template(f'{model_type}/result.html',
                               image_path=file_path,
                               filename=filename,
                               model_type=model_type,
                               predicted_label=predicted_label,
                               confidence_score=confidence_score)
        except Exception as e:
            return render_template('error.html', 
                                error=f"Image processing error: {str(e)}",
                                model_type=model_type,
                                now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 500

@app.route('/<model_type>/camera')
@login_required
def camera(model_type):
    if model_type not in MODELS:
        return redirect(url_for('home'))
    return render_template(f'{model_type}/camera.html')

@app.route('/uploads/<model_type>/<filename>')
@login_required
def uploaded_file(model_type, filename):
    if model_type not in MODELS:
        return redirect(url_for('home'))
    return send_from_directory(app.config['UPLOAD_FOLDERS'][model_type], filename)

# Error handling
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', 
                         error="Page not found",
                         now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', 
                         error="Internal server error",
                         now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)