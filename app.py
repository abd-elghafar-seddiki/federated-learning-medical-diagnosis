import os
import cv2
import numpy as np
import pickle
import secrets
import shutil
import sqlite3

from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask,
    request,
    render_template,
    redirect,
    url_for,
    send_from_directory,
    jsonify,
    session,
    flash
)

from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import load_model, Model, save_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import Adam


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

app = Flask(
    __name__,
    static_url_path="/static",
    static_folder="static",
    template_folder="templates"
)

# IMPORTANT:
# On Render, create SECRET_KEY as an Environment Variable.
app.secret_key = os.environ.get(
    "SECRET_KEY",
    secrets.token_hex(32)
)

bcrypt = Bcrypt(app)


# ============================================================
# KAGGLE CONFIGURATION
# ============================================================

# Your public Kaggle dataset
KAGGLE_DATASET = os.environ.get(
    "KAGGLE_DATASET",
    "abdelghafarseddiki/federated-medical-diagnosis-models"
)

# Local folder where downloaded models will be stored
MODELS_DIR = os.environ.get(
    "MODELS_DIR",
    "models"
)

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# USER ROLES
# ============================================================

ROLES = {
    "admin": 3,
    "doctor": 2,
    "user": 1
}


# ============================================================
# UPLOAD FOLDERS
# ============================================================

UPLOAD_FOLDERS = {
    "lung_cancer_xray": "uploads/lung_cancer_xray",
    "lung_cancer_histo": "uploads/lung_cancer_histo",
    "malaria": "uploads/malaria",
    "retinal_oct": "uploads/retinal_oct",
    "diabetic_retinopathy": "uploads/diabetic_retinopathy",
    "brain_tumor": "uploads/brain_tumor",
    "covid_xray": "uploads/covid_xray"
}

for folder in UPLOAD_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDERS"] = UPLOAD_FOLDERS


# ============================================================
# PRINTER FOLDER
# ============================================================

PRINTER_FOLDER = "printer_scans"

os.makedirs(PRINTER_FOLDER, exist_ok=True)

app.config["PRINTER_FOLDER"] = PRINTER_FOLDER


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_PATHS = {
    "lung_cancer_xray":
        os.path.join(MODELS_DIR, "lung_cancer_detection_model.h5"),

    "lung_cancer_histo":
        os.path.join(MODELS_DIR, "lung-cancer-resnet-model.h5"),

    "malaria":
        os.path.join(MODELS_DIR, "malaria.h5"),

    "retinal_oct":
        os.path.join(MODELS_DIR, "Retinal-OCT-resnet-model.h5"),

    "diabetic_retinopathy":
        os.path.join(
            MODELS_DIR,
            "Diabetic-Retinopathy-ResNet50-model.h5"
        ),

    "brain_tumor":
        os.path.join(MODELS_DIR, "brain_tumor_model.h5"),

    "covid_xray":
        os.path.join(
            MODELS_DIR,
            "CNN_Covid19_Xray_Version.h5"
        )
}


# ============================================================
# MODEL DISPLAY NAMES
# ============================================================

MODEL_DISPLAY_NAMES = {
    "lung_cancer_xray": "Lung Cancer X-Ray",
    "lung_cancer_histo": "Lung Cancer Histopathology",
    "malaria": "Malaria Detection",
    "retinal_oct": "Retinal OCT",
    "diabetic_retinopathy": "Diabetic Retinopathy",
    "brain_tumor": "Brain Tumor MRI",
    "covid_xray": "COVID-19 X-Ray"
}


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    "lung_cancer_xray": [
        "Benign",
        "Malignant",
        "Normal"
    ],

    "lung_cancer_histo": [
        "Adenocarcinoma",
        "Benign",
        "Squamous Cell Carcinoma"
    ],

    "malaria": [
        "Parasitized",
        "Uninfected"
    ],

    "retinal_oct": [
        "CNV",
        "DME",
        "DRUSEN",
        "Normal"
    ],

    "diabetic_retinopathy": [
        "Diabetic",
        "Non-Diabetic"
    ],

    "brain_tumor": [
        "Tumor",
        "No Tumor"
    ],

    "covid_xray": [
        "Covid",
        "Viral Pneumonia",
        "Normal"
    ]
}


# ============================================================
# MODEL CACHE
# ============================================================

# IMPORTANT:
# We DO NOT load all models when the application starts.
#
# This is necessary because Render Free has limited RAM.
#
# Only the model requested by the user is loaded.
MODELS = {}


# ============================================================
# KAGGLE MODEL DOWNLOADER
# ============================================================

def download_model_from_kaggle(model_type):
    """
    Download one specific model from the public Kaggle dataset.

    This avoids downloading the complete 1 GB dataset.
    """

    if model_type not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    local_path = MODEL_PATHS[model_type]

    # Already downloaded
    if os.path.exists(local_path):
        print(
            f"[KAGGLE] Model already exists: {local_path}"
        )
        return local_path

    filename = os.path.basename(local_path)

    print(
        f"[KAGGLE] Downloading: {filename}"
    )

    try:

        import kagglehub

        downloaded_path = kagglehub.dataset_download(
            KAGGLE_DATASET,
            path=filename,
            output_dir=MODELS_DIR
        )

        # kagglehub normally returns the exact file path
        if os.path.isfile(downloaded_path):

            if downloaded_path != local_path:

                os.makedirs(
                    os.path.dirname(local_path),
                    exist_ok=True
                )

                shutil.copy2(
                    downloaded_path,
                    local_path
                )

            print(
                f"[KAGGLE] Download completed: {local_path}"
            )

            return local_path

        # Fallback: search the models directory
        possible_path = os.path.join(
            MODELS_DIR,
            filename
        )

        if os.path.exists(possible_path):

            print(
                f"[KAGGLE] Model found: {possible_path}"
            )

            return possible_path

        raise FileNotFoundError(
            f"Kaggle downloaded the file but it "
            f"could not be located: {filename}"
        )

    except Exception as e:

        print(
            f"[KAGGLE ERROR] Could not download "
            f"{filename}: {e}"
        )

        raise RuntimeError(
            f"Unable to download model '{filename}' "
            f"from Kaggle. Error: {e}"
        )


# ============================================================
# COVID LABEL ENCODER
# ============================================================

COVID_LABEL_ENCODER_FILENAME = "Label_encoder.pkl"

COVID_LABEL_ENCODER_PATH = os.path.join(
    MODELS_DIR,
    COVID_LABEL_ENCODER_FILENAME
)

covid_le = None


def load_covid_label_encoder():
    """
    Load the COVID label encoder from Kaggle.

    If unavailable, use the known class ordering.
    """

    global covid_le

    if covid_le is not None:
        return covid_le

    try:

        if not os.path.exists(
            COVID_LABEL_ENCODER_PATH
        ):

            import kagglehub

            print(
                "[KAGGLE] Downloading COVID "
                "label encoder..."
            )

            downloaded_path = kagglehub.dataset_download(
                KAGGLE_DATASET,
                path=COVID_LABEL_ENCODER_FILENAME,
                output_dir=MODELS_DIR
            )

            if os.path.isfile(downloaded_path):

                if downloaded_path != COVID_LABEL_ENCODER_PATH:

                    shutil.copy2(
                        downloaded_path,
                        COVID_LABEL_ENCODER_PATH
                    )

        if os.path.exists(
            COVID_LABEL_ENCODER_PATH
        ):

            with open(
                COVID_LABEL_ENCODER_PATH,
                "rb"
            ) as f:

                covid_le = pickle.load(f)

            print(
                "[KAGGLE] COVID Label Encoder loaded."
            )

            return covid_le

    except Exception as e:

        print(
            f"[KAGGLE] Label encoder unavailable: {e}"
        )

    # Safe fallback
    covid_le = LabelEncoder()

    covid_le.fit([
        "Covid",
        "Viral Pneumonia",
        "Normal"
    ])

    return covid_le


# ============================================================
# LOAD MODEL ONLY WHEN NEEDED
# ============================================================

def get_model(model_type):
    """
    Lazy-load a model.

    The model is downloaded from Kaggle only if necessary,
    then loaded into memory only when requested.
    """

    if model_type not in MODEL_PATHS:

        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    # Model already in RAM
    if model_type in MODELS:

        return MODELS[model_type]

    print(
        f"\n[MODEL] Preparing {model_type}..."
    )

    model_path = MODEL_PATHS[model_type]

    # Download model if it does not exist
    if not os.path.exists(model_path):

        download_model_from_kaggle(
            model_type
        )

    print(
        f"[MODEL] Loading {model_path}"
    )

    try:

        model = load_model(
            model_path,
            compile=False
        )

        MODELS[model_type] = model

        print(
            f"[MODEL] {model_type} loaded successfully."
        )

        return model

    except Exception as e:

        print(
            f"[MODEL ERROR] {model_type}: {e}"
        )

        # Remove corrupted downloaded file
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
        except:
            pass

        raise RuntimeError(
            f"Could not load model "
            f"{model_type}: {e}"
        )


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():

    conn = sqlite3.connect(
        "users.db"
    )

    conn.row_factory = sqlite3.Row

    return conn


def init_db():

    conn = get_db_connection()

    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            registration_date TEXT NOT NULL
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS password_resets
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id)
            REFERENCES users(id)
        )
        """
    )

    conn.commit()

    conn.close()


init_db()


# ============================================================
# USER FUNCTIONS
# ============================================================

def create_user(
    first_name,
    last_name,
    birth_date,
    hospital,
    phone,
    user_type,
    email,
    password
):

    conn = get_db_connection()

    hashed_password = (
        bcrypt
        .generate_password_hash(password)
        .decode("utf-8")
    )

    registration_date = (
        datetime.now()
        .strftime("%Y-%m-%d %H:%M:%S")
    )

    user_count = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    # First registered user = admin
    role = 3 if user_count == 0 else 1

    try:

        conn.execute(
            """
            INSERT INTO users
            (
                first_name,
                last_name,
                birth_date,
                hospital,
                phone,
                user_type,
                email,
                password,
                role,
                registration_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                first_name,
                last_name,
                birth_date,
                hospital,
                phone,
                user_type,
                email,
                hashed_password,
                role,
                registration_date
            )
        )

        conn.commit()

        return True

    except sqlite3.IntegrityError:

        return False

    finally:

        conn.close()


def verify_user(email, password):

    conn = get_db_connection()

    user = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        (email,)
    ).fetchone()

    conn.close()

    if user:

        if bcrypt.check_password_hash(
            user["password"],
            password
        ):

            return user

    return None


def get_user_by_id(user_id):

    conn = get_db_connection()

    user = conn.execute(
        "SELECT * FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()

    conn.close()

    return user


def update_user(user_id, **kwargs):

    conn = get_db_connection()

    try:

        query = "UPDATE users SET "

        params = []

        for key, value in kwargs.items():

            if key == "password":

                value = (
                    bcrypt
                    .generate_password_hash(value)
                    .decode("utf-8")
                )

            query += f"{key} = ?, "

            params.append(value)

        query = query.rstrip(", ")

        query += " WHERE id = ?"

        params.append(user_id)

        conn.execute(
            query,
            params
        )

        conn.commit()

        return True

    except Exception as e:

        print(
            f"[DATABASE ERROR] {e}"
        )

        return False

    finally:

        conn.close()


def get_user_stats():

    conn = get_db_connection()

    stats = {

        "total_users":
            conn.execute(
                "SELECT COUNT(*) FROM users"
            ).fetchone()[0],

        "active_users":
            conn.execute(
                """
                SELECT COUNT(*)
                FROM users
                WHERE is_active = 1
                """
            ).fetchone()[0],

        "admin_count":
            conn.execute(
                """
                SELECT COUNT(*)
                FROM users
                WHERE role = 3
                """
            ).fetchone()[0]
    }

    conn.close()

    return stats


# ============================================================
# AUTH DECORATORS
# ============================================================

def login_required(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please log in to access this page",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        return f(*args, **kwargs)

    return decorated_function


def role_required(role_name):

    def decorator(f):

        @wraps(f)
        def decorated_function(
            *args,
            **kwargs
        ):

            if "user_id" not in session:

                flash(
                    "Please log in to access this page",
                    "warning"
                )

                return redirect(
                    url_for("login")
                )

            user_role = session.get(
                "user_role",
                1
            )

            required_role = ROLES.get(
                role_name,
                1
            )

            if user_role < required_role:

                flash(
                    "You do not have permission "
                    "to access this page",
                    "danger"
                )

                return redirect(
                    url_for("home")
                )

            return f(*args, **kwargs)

        return decorated_function

    return decorator


# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form["email"]

        password = request.form["password"]

        user = verify_user(
            email,
            password
        )

        if user:

            if not user["is_active"]:

                flash(
                    "Your account is inactive. "
                    "Please contact the administrator",
                    "danger"
                )

                return redirect(
                    url_for("login")
                )

            session["user_id"] = user["id"]

            session["user_email"] = user["email"]

            session["user_role"] = user["role"]

            session["user_name"] = (
                f"{user['first_name']} "
                f"{user['last_name']}"
            )

            flash(
                f"Welcome back, "
                f"{session['user_name']}!",
                "success"
            )

            return redirect(
                url_for("home")
            )

        flash(
            "Incorrect email or password",
            "danger"
        )

    return render_template(
        "auth/login.html"
    )


@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        first_name = request.form["first_name"]

        last_name = request.form["last_name"]

        birth_date = request.form["birth_date"]

        hospital = request.form["hospital"]

        phone = request.form["phone"]

        user_type = request.form["user_type"]

        email = request.form["email"]

        password = request.form["password"]

        if create_user(
            first_name,
            last_name,
            birth_date,
            hospital,
            phone,
            user_type,
            email,
            password
        ):

            flash(
                "Registration successful! "
                "Please log in",
                "success"
            )

            return redirect(
                url_for("login")
            )

        flash(
            "Email already registered",
            "danger"
        )

    return render_template(
        "auth/register.html"
    )


@app.route("/logout")
@login_required
def logout():

    session.clear()

    flash(
        "Logged out successfully",
        "info"
    )

    return redirect(
        url_for("login")
    )


# ============================================================
# ADMIN
# ============================================================

@app.route("/admin")
@role_required("admin")
def admin_dashboard():

    conn = get_db_connection()

    users = conn.execute(
        """
        SELECT *
        FROM users
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    stats = get_user_stats()

    return render_template(
        "admin/dashboard.html",
        users=users,
        ROLES=ROLES,
        total_users=stats["total_users"],
        active_users=stats["active_users"],
        admin_count=stats["admin_count"]
    )


@app.route(
    "/admin/update-user/<int:user_id>",
    methods=["POST"]
)
@role_required("admin")
def admin_update_user(user_id):

    data = request.get_json()

    field = data.get("field")

    value = data.get("value")

    if field not in [
        "role",
        "is_active"
    ]:

        return jsonify({
            "success": False,
            "message": "Invalid field"
        })

    conn = get_db_connection()

    try:

        conn.execute(
            f"""
            UPDATE users
            SET {field} = ?
            WHERE id = ?
            """,
            (
                value,
                user_id
            )
        )

        conn.commit()

        return jsonify({
            "success": True
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        })

    finally:

        conn.close()


@app.route(
    "/admin/edit-user/<int:user_id>",
    methods=["GET", "POST"]
)
@role_required("admin")
def admin_edit_user(user_id):

    user = get_user_by_id(user_id)

    if not user:

        flash(
            "User not found",
            "danger"
        )

        return redirect(
            url_for("admin_dashboard")
        )

    if request.method == "POST":

        updates = {

            "first_name":
                request.form["first_name"],

            "last_name":
                request.form["last_name"],

            "birth_date":
                request.form["birth_date"],

            "hospital":
                request.form["hospital"],

            "phone":
                request.form["phone"],

            "user_type":
                request.form["user_type"],

            "email":
                request.form["email"],

            "is_active":
                1
                if request.form.get("is_active") == "on"
                else 0
        }

        if request.form["password"]:

            updates["password"] = (
                request.form["password"]
            )

        if update_user(
            user_id,
            **updates
        ):

            flash(
                "User updated successfully",
                "success"
            )

        else:

            flash(
                "Error updating user",
                "danger"
            )

        return redirect(
            url_for(
                "admin_edit_user",
                user_id=user_id
            )
        )

    return render_template(
        "admin/edit_user.html",
        user=user
    )


# ============================================================
# PROFILE
# ============================================================

@app.route(
    "/profile",
    methods=["GET", "POST"]
)
@login_required
def profile():

    user = get_user_by_id(
        session["user_id"]
    )

    if request.method == "POST":

        updates = {

            "first_name":
                request.form["first_name"],

            "last_name":
                request.form["last_name"],

            "birth_date":
                request.form["birth_date"],

            "hospital":
                request.form["hospital"],

            "phone":
                request.form["phone"],

            "user_type":
                request.form["user_type"]
        }

        if request.form["password"]:

            updates["password"] = (
                request.form["password"]
            )

        if update_user(
            session["user_id"],
            **updates
        ):

            session["user_name"] = (
                f"{updates['first_name']} "
                f"{updates['last_name']}"
            )

            flash(
                "Profile updated successfully",
                "success"
            )

        else:

            flash(
                "Error updating profile",
                "danger"
            )

        return redirect(
            url_for("profile")
        )

    return render_template(
        "profile.html",
        user=user
    )


# ============================================================
# PASSWORD RESET
# ============================================================

@app.route(
    "/reset-password",
    methods=["GET", "POST"]
)
def reset_password_request():

    if request.method == "POST":

        email = request.form["email"]

        conn = get_db_connection()

        user = conn.execute(
            """
            SELECT *
            FROM users
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        conn.close()

        if user:

            token = secrets.token_urlsafe(32)

            expires_at = (
                datetime.now()
                + timedelta(hours=1)
            ).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            conn = get_db_connection()

            conn.execute(
                """
                INSERT INTO password_resets
                (
                    user_id,
                    token,
                    expires_at
                )
                VALUES (?, ?, ?)
                """,
                (
                    user["id"],
                    token,
                    expires_at
                )
            )

            conn.commit()

            conn.close()

            flash(
                "Password reset link has been sent "
                "to your email",
                "info"
            )

        else:

            flash(
                "Email does not exist",
                "danger"
            )

        return redirect(
            url_for("login")
        )

    return render_template(
        "auth/reset_password_request.html"
    )


@app.route(
    "/reset-password/<token>",
    methods=["GET", "POST"]
)
def reset_password(token):

    conn = get_db_connection()

    reset_request = conn.execute(
        """
        SELECT *
        FROM password_resets
        WHERE token = ?
        AND expires_at > datetime('now')
        """,
        (token,)
    ).fetchone()

    valid_token = (
        reset_request is not None
    )

    if request.method == "POST" and valid_token:

        new_password = (
            request.form["password"]
        )

        if update_user(
            reset_request["user_id"],
            password=new_password
        ):

            conn.execute(
                """
                DELETE FROM password_resets
                WHERE token = ?
                """,
                (token,)
            )

            conn.commit()

            conn.close()

            flash(
                "Password updated successfully. "
                "Please log in",
                "success"
            )

            return redirect(
                url_for("login")
            )

        flash(
            "Error updating password",
            "danger"
        )

    conn.close()

    return render_template(
        "auth/reset_password.html",
        token=token,
        valid_token=valid_token
    )


# ============================================================
# MODEL UPDATE
# ============================================================

@app.route(
    "/admin/update-models",
    methods=["GET", "POST"]
)
@role_required("admin")
def update_models():

    if request.method == "POST":

        model_type = request.form.get(
            "model_type"
        )

        file = request.files.get(
            "model_file"
        )

        if (
            not model_type
            or model_type not in MODEL_PATHS
        ):

            flash(
                "Invalid model type",
                "danger"
            )

            return redirect(
                url_for("update_models")
            )

        if (
            not file
            or file.filename == ""
        ):

            flash(
                "No file selected",
                "danger"
            )

            return redirect(
                url_for("update_models")
            )

        try:

            filename = secure_filename(
                file.filename
            )

            os.makedirs(
                "temp_uploads",
                exist_ok=True
            )

            temp_path = os.path.join(
                "temp_uploads",
                filename
            )

            file.save(temp_path)

            # Validate model
            test_model = load_model(
                temp_path,
                compile=False
            )

            # Replace the Kaggle model locally
            model_path = MODEL_PATHS[
                model_type
            ]

            os.makedirs(
                MODELS_DIR,
                exist_ok=True
            )

            shutil.move(
                temp_path,
                model_path
            )

            # Replace cached model
            MODELS[model_type] = test_model

            flash(
                f"Model {model_type} updated successfully!",
                "success"
            )

            return redirect(
                url_for("update_models")
            )

        except Exception as e:

            try:

                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except:
                pass

            flash(
                f"Error updating model: {e}",
                "danger"
            )

            return redirect(
                url_for("update_models")
            )

    model_updates = {}

    for model_type, path in MODEL_PATHS.items():

        if os.path.exists(path):

            model_updates[model_type] = (
                datetime
                .fromtimestamp(
                    os.path.getmtime(path)
                )
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

        else:

            model_updates[model_type] = (
                "Not downloaded"
            )

    return render_template(
        "admin/update_models.html",
        models=list(
            MODEL_PATHS.keys()
        ),
        model_names=CLASS_NAMES,
        MODELS=MODELS,
        MODEL_PATHS=MODEL_PATHS,
        model_updates=model_updates
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(
    model_type,
    image_path
):

    try:

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise ValueError(
                f"Failed to read image at "
                f"{image_path}"
            )

        # IMPORTANT:
        # Get ONLY the requested model.
        model = get_model(
            model_type
        )

        classes = CLASS_NAMES[
            model_type
        ]

        # ----------------------------------------------------
        # LUNG CANCER X-RAY
        # ----------------------------------------------------

        if model_type == "lung_cancer_xray":

            img = cv2.resize(
                image,
                (256, 256)
            )

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            img = img / 255.0

            img = np.expand_dims(
                img,
                axis=0
            )

        # ----------------------------------------------------
        # COVID X-RAY
        # ----------------------------------------------------

        elif model_type == "covid_xray":

            img = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            img = cv2.resize(
                img,
                (150, 150)
            )

            img = img / 255.0

            img = np.expand_dims(
                img,
                axis=0
            )

            predictions = model.predict(
                img,
                verbose=0
            )

            predicted_index = int(
                np.argmax(
                    predictions
                )
            )

            confidence = float(
                predictions[0][
                    predicted_index
                ]
            )

            encoder = (
                load_covid_label_encoder()
            )

            try:

                predicted_label = (
                    encoder
                    .inverse_transform(
                        [predicted_index]
                    )[0]
                )

            except Exception:

                predicted_label = (
                    classes[
                        predicted_index
                    ]
                )

            return (
                predicted_label,
                confidence
            )

        # ----------------------------------------------------
        # RESNET MODELS
        # ----------------------------------------------------

        elif model_type in [
            "lung_cancer_histo",
            "retinal_oct",
            "diabetic_retinopathy",
            "brain_tumor"
        ]:

            img = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            img = cv2.resize(
                img,
                (224, 224)
            )

            img = preprocess_input(
                img
            )

            img = np.expand_dims(
                img,
                axis=0
            )

        # ----------------------------------------------------
        # MALARIA
        # ----------------------------------------------------

        elif model_type == "malaria":

            img = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            img = cv2.resize(
                img,
                (100, 100)
            )

            img = img / 255.0

            img = np.expand_dims(
                img,
                axis=0
            )

        else:

            raise ValueError(
                f"Unsupported model type: "
                f"{model_type}"
            )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            img,
            verbose=0
        )

        # Binary models
        if model_type in [
            "malaria",
            "diabetic_retinopathy"
        ]:

            score = float(
                predictions[0][0]
            )

            predicted_index = (
                1
                if score > 0.5
                else 0
            )

            confidence = (
                score
                if predicted_index == 1
                else 1 - score
            )

        else:

            predicted_index = int(
                np.argmax(
                    predictions[0]
                )
            )

            confidence = float(
                predictions[0][
                    predicted_index
                ]
            )

        if predicted_index >= len(classes):

            predicted_index = (
                len(classes) - 1
            )

        predicted_label = classes[
            predicted_index
        ]

        return (
            predicted_label,
            float(confidence)
        )

    except Exception as e:

        print(
            f"[PREDICTION ERROR] "
            f"{model_type}: {e}"
        )

        raise


# ============================================================
# PRINTER PROCESSING
# ============================================================

def process_printer_image(
    model_type,
    image_path
):

    try:

        img = cv2.imread(
            image_path
        )

        if img is None:

            raise ValueError(
                "Failed to read printer image"
            )

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

        saved_path = os.path.join(
            app.config["PRINTER_FOLDER"],
            f"scan_{timestamp}.jpg"
        )

        cv2.imwrite(
            saved_path,
            img
        )

        result = process_image(
            model_type,
            saved_path
        )

        return {
            "status": "success",
            "prediction": result[0],
            "confidence": float(result[1]),
            "image_path": saved_path
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }


@app.route(
    "/api/process_scan",
    methods=["POST"]
)
@login_required
def process_scan():

    if "file" not in request.files:

        return jsonify({
            "error": "No file provided"
        }), 400

    file = request.files["file"]

    if file.filename == "":

        return jsonify({
            "error": "Filename is empty"
        }), 400

    filename = secure_filename(
        file.filename
    )

    temp_path = os.path.join(
        app.config["PRINTER_FOLDER"],
        f"temp_{filename}"
    )

    file.save(temp_path)

    model_type = request.form.get(
        "model_type",
        "lung_cancer_xray"
    )

    if model_type not in MODEL_PATHS:

        try:
            os.remove(temp_path)
        except:
            pass

        return jsonify({
            "error": "Invalid model type"
        }), 400

    result = process_printer_image(
        model_type,
        temp_path
    )

    try:

        os.remove(temp_path)

    except:

        pass

    if result["status"] == "error":

        return jsonify(
            result
        ), 500

    return jsonify(
        result
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_status():

    result = {}

    for model_type, path in MODEL_PATHS.items():

        result[model_type] = {

            "name":
                MODEL_DISPLAY_NAMES[
                    model_type
                ],

            "downloaded":
                os.path.exists(path),

            "loaded":
                model_type in MODELS
        }

    return result


# ============================================================
# MAIN PAGE
# ============================================================

@app.route("/")
@login_required
def home():

    model_updates = {}

    for model_type, model_path in MODEL_PATHS.items():

        if os.path.exists(
            model_path
        ):

            model_updates[model_type] = (
                datetime
                .fromtimestamp(
                    os.path.getmtime(
                        model_path
                    )
                )
                .strftime(
                    "%Y-%m-%d %H:%M"
                )
            )

        else:

            model_updates[model_type] = (
                "Available on Kaggle"
            )

    return render_template(
        "index.html",
        model_updates=model_updates,
        model_status=get_model_status()
    )


# ============================================================
# PRINTER
# ============================================================

@app.route("/printer")
@login_required
def printer_interface():

    return render_template(
        "printer.html"
    )


# ============================================================
# MODEL HOME
# ============================================================

@app.route("/<model_type>")
@login_required
def model_home(model_type):

    if model_type not in MODEL_PATHS:

        return redirect(
            url_for("home")
        )

    return render_template(
        f"{model_type}/index.html"
    )


# ============================================================
# MODEL UPLOAD
# ============================================================

@app.route(
    "/<model_type>/upload",
    methods=["POST"]
)
@login_required
def upload_file(model_type):

    if model_type not in MODEL_PATHS:

        return redirect(
            request.url
        )

    if "file" not in request.files:

        return redirect(
            request.url
        )

    file = request.files["file"]

    if file.filename == "":

        return redirect(
            request.url
        )

    filename = secure_filename(
        file.filename
    )

    upload_folder = (
        app.config["UPLOAD_FOLDERS"][
            model_type
        ]
    )

    os.makedirs(
        upload_folder,
        exist_ok=True
    )

    file_path = os.path.join(
        upload_folder,
        filename
    )

    file.save(file_path)

    try:

        predicted_label, confidence_score = (
            process_image(
                model_type,
                file_path
            )
        )

        return render_template(
            f"{model_type}/result.html",
            image_path=file_path,
            filename=filename,
            model_type=model_type,
            predicted_label=predicted_label,
            confidence_score=confidence_score
        )

    except Exception as e:

        return render_template(
            "error.html",
            error=(
                f"Image processing error: {str(e)}"
            ),
            model_type=model_type,
            now=datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ), 500


# ============================================================
# CAMERA
# ============================================================

@app.route(
    "/<model_type>/camera"
)
@login_required
def camera(model_type):

    if model_type not in MODEL_PATHS:

        return redirect(
            url_for("home")
        )

    return render_template(
        f"{model_type}/camera.html"
    )


# ============================================================
# UPLOADED FILE
# ============================================================

@app.route(
    "/uploads/<model_type>/<filename>"
)
@login_required
def uploaded_file(
    model_type,
    filename
):

    if model_type not in MODEL_PATHS:

        return redirect(
            url_for("home")
        )

    return send_from_directory(
        app.config[
            "UPLOAD_FOLDERS"
        ][model_type],
        filename
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "application": "Federated Medical Diagnosis",
        "kaggle_dataset": KAGGLE_DATASET,
        "models_available": len(
            MODEL_PATHS
        ),
        "models_loaded_in_memory": len(
            MODELS
        )
    })


# ============================================================
# MODEL STATUS API
# ============================================================

@app.route(
    "/api/models/status"
)
@login_required
def model_status_api():

    return jsonify(
        get_model_status()
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(e):

    return render_template(
        "error.html",
        error="Page not found",
        now=datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    ), 404


@app.errorhandler(500)
def server_error(e):

    return render_template(
        "error.html",
        error="Internal server error",
        now=datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    ), 500


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )
