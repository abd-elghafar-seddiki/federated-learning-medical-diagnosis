import os
import cv2
import gc
import numpy as np
import pickle
import secrets
import shutil
import threading
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

import kagglehub


# ============================================================
# APPLICATION INITIALIZATION
# ============================================================

app = Flask(__name__, static_url_path='/static')

app.secret_key = os.environ.get(
    "SECRET_KEY",
    secrets.token_hex(32)
)

bcrypt = Bcrypt(app)


# ============================================================
# KAGGLE CONFIGURATION
# ============================================================

KAGGLE_DATASET = (
    "abdelghafarseddiki/"
    "federated-medical-diagnosis-models"
)

# Local directory where downloaded models will be stored.
# This directory is ignored by Git.
MODELS_DIR = os.path.join(
    os.getcwd(),
    "models"
)

os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# MODEL FILE NAMES
# ============================================================

MODEL_FILES = {
    "lung_cancer_xray":
        "lung_cancer_detection_model.h5",

    "lung_cancer_histo":
        "lung-cancer-resnet-model.h5",

    "malaria":
        "malaria.h5",

    "retinal_oct":
        "Retinal-OCT-resnet-model.h5",

    "diabetic_retinopathy":
        "Diabetic-Retinopathy-ResNet50-model.h5",

    "brain_tumor":
        "brain_tumor_model.h5",

    "covid_xray":
        "CNN_Covid19_Xray_Version.h5"
}


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATHS = {
    model_type: os.path.join(
        MODELS_DIR,
        filename
    )
    for model_type, filename in MODEL_FILES.items()
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

# Models are loaded only when needed.
MODELS = {}

# Prevent two simultaneous requests from downloading/loading
# the same model at the same time.
MODEL_LOCK = threading.Lock()


# ============================================================
# KAGGLE MODEL DOWNLOADER
# ============================================================

def download_model_from_kaggle(model_type):
    """
    Download one specific model from the public Kaggle dataset.

    Only the requested model is downloaded.
    We do NOT download the complete 1 GB dataset.
    """

    if model_type not in MODEL_FILES:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    filename = MODEL_FILES[model_type]
    local_path = MODEL_PATHS[model_type]

    # Already downloaded
    if os.path.exists(local_path):
        print(
            f"[KAGGLE] Model already exists: {filename}"
        )
        return local_path

    print("=" * 70)
    print(
        f"[KAGGLE] Downloading model: {filename}"
    )
    print(
        f"[KAGGLE] Dataset: {KAGGLE_DATASET}"
    )
    print("=" * 70)

    try:

        downloaded_path = kagglehub.dataset_download(
            KAGGLE_DATASET,
            path=filename,
            output_dir=MODELS_DIR
        )

        print(
            f"[KAGGLE] Downloaded path: "
            f"{downloaded_path}"
        )

        # KaggleHub normally returns the exact file path.
        # However, we handle possible nested paths too.
        if os.path.isfile(downloaded_path):

            if os.path.abspath(
                downloaded_path
            ) != os.path.abspath(local_path):

                shutil.copy2(
                    downloaded_path,
                    local_path
                )

        elif not os.path.exists(local_path):

            # Search for the requested filename
            found_path = None

            for root, dirs, files in os.walk(
                MODELS_DIR
            ):

                if filename in files:

                    found_path = os.path.join(
                        root,
                        filename
                    )

                    break

            if found_path:

                if os.path.abspath(
                    found_path
                ) != os.path.abspath(local_path):

                    shutil.copy2(
                        found_path,
                        local_path
                    )

            else:

                raise FileNotFoundError(
                    f"Kaggle download completed, "
                    f"but {filename} was not found."
                )

        if not os.path.exists(local_path):

            raise FileNotFoundError(
                f"Model file does not exist after "
                f"Kaggle download: {local_path}"
            )

        print(
            f"[KAGGLE] Model ready: {local_path}"
        )

        return local_path

    except Exception as e:

        print(
            f"[KAGGLE ERROR] Failed to download "
            f"{filename}"
        )

        print(str(e))

        raise RuntimeError(
            f"Could not download model "
            f"{filename} from Kaggle. "
            f"Error: {str(e)}"
        )


# ============================================================
# LOAD MODEL
# ============================================================

def get_model(model_type):
    """
    Lazy-load a model.

    First request:
        1. Download model from Kaggle if necessary.
        2. Load model into RAM.
        3. Store it in MODELS cache.

    Later requests:
        Use the already loaded model.
    """

    if model_type not in MODEL_FILES:

        raise ValueError(
            f"Invalid model type: {model_type}"
        )

    # Fast path:
    # model is already loaded.
    if model_type in MODELS:

        return MODELS[model_type]

    # Prevent concurrent loading.
    with MODEL_LOCK:

        # Check again after acquiring lock.
        if model_type in MODELS:

            return MODELS[model_type]

        print("=" * 70)
        print(
            f"[MODEL] Initializing: {model_type}"
        )
        print("=" * 70)

        # Download if necessary.
        model_path = download_model_from_kaggle(
            model_type
        )

        # Load the REAL trained model.
        try:

            print(
                f"[MODEL] Loading: {model_path}"
            )

            model = load_model(
                model_path,
                compile=False
            )

            MODELS[model_type] = model

            print(
                f"[MODEL] Successfully loaded: "
                f"{model_type}"
            )

            return model

        except Exception as e:

            print(
                f"[MODEL ERROR] Could not load "
                f"{model_type}: {str(e)}"
            )

            # Do NOT create a fake/random replacement model.
            raise RuntimeError(
                f"Failed to load trained model "
                f"{model_type}: {str(e)}"
            )


# ============================================================
# OPTIONAL MODEL UNLOADING
# ============================================================

def unload_model(model_type):
    """
    Remove a model from memory.

    Useful on memory-limited hosting.
    """

    if model_type in MODELS:

        try:

            del MODELS[model_type]

            gc.collect()

            print(
                f"[MODEL] Unloaded: {model_type}"
            )

            return True

        except Exception as e:

            print(
                f"[MODEL] Failed to unload "
                f"{model_type}: {e}"
            )

    return False


# ============================================================
# COVID LABEL ENCODER
# ============================================================

COVID_LABEL_ENCODER_FILE = os.path.join(
    MODELS_DIR,
    "Label_encoder.pkl"
)

covid_le = None


def load_covid_label_encoder():
    """
    Download and load the COVID label encoder.
    """

    global covid_le

    if covid_le is not None:
        return covid_le

    if not os.path.exists(
        COVID_LABEL_ENCODER_FILE
    ):

        print(
            "[KAGGLE] Downloading "
            "Label_encoder.pkl..."
        )

        try:

            downloaded_path = (
                kagglehub.dataset_download(
                    KAGGLE_DATASET,
                    path="Label_encoder.pkl",
                    output_dir=MODELS_DIR
                )
            )

            if os.path.isfile(downloaded_path):

                if os.path.abspath(
                    downloaded_path
                ) != os.path.abspath(
                    COVID_LABEL_ENCODER_FILE
                ):

                    shutil.copy2(
                        downloaded_path,
                        COVID_LABEL_ENCODER_FILE
                    )

        except Exception as e:

            print(
                "[KAGGLE] Could not download "
                f"Label_encoder.pkl: {e}"
            )

    # Try loading the real encoder.
    if os.path.exists(
        COVID_LABEL_ENCODER_FILE
    ):

        try:

            with open(
                COVID_LABEL_ENCODER_FILE,
                "rb"
            ) as f:

                covid_le = pickle.load(f)

            print(
                "[COVID] Label encoder loaded."
            )

            print(
                "[COVID] Classes:",
                covid_le.classes_
            )

            return covid_le

        except Exception as e:

            print(
                "[COVID] Failed to load "
                f"label encoder: {e}"
            )

    # Fallback only for label decoding.
    # This does NOT replace the trained model.
    print(
        "[COVID] Creating default label encoder."
    )

    covid_le = LabelEncoder()

    covid_le.fit(
        [
            "Covid",
            "Viral Pneumonia",
            "Normal"
        ]
    )

    return covid_le


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

    "lung_cancer_xray":
        "uploads/lung_cancer_xray",

    "lung_cancer_histo":
        "uploads/lung_cancer_histo",

    "malaria":
        "uploads/malaria",

    "retinal_oct":
        "uploads/retinal_oct",

    "diabetic_retinopathy":
        "uploads/diabetic_retinopathy",

    "brain_tumor":
        "uploads/brain_tumor",

    "covid_xray":
        "uploads/covid_xray"
}


for folder in UPLOAD_FOLDERS.values():

    os.makedirs(
        folder,
        exist_ok=True
    )


app.config[
    "UPLOAD_FOLDERS"
] = UPLOAD_FOLDERS


# ============================================================
# PRINTER FOLDER
# ============================================================

PRINTER_FOLDER = "printer_scans"

os.makedirs(
    PRINTER_FOLDER,
    exist_ok=True
)

app.config[
    "PRINTER_FOLDER"
] = PRINTER_FOLDER


# ============================================================
# DATABASE
# ============================================================

def init_db():

    conn = sqlite3.connect(
        "users.db"
    )

    c = conn.cursor()

    # Users table
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

    # Password reset table
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
# DATABASE HELPERS
# ============================================================

def get_db_connection():

    conn = sqlite3.connect(
        "users.db"
    )

    conn.row_factory = sqlite3.Row

    return conn


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

    # First user becomes admin.
    user_count = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

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


def verify_user(
    email,
    password
):

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

    if (
        user
        and bcrypt.check_password_hash(
            user["password"],
            password
        )
    ):

        return user

    return None


def get_user_by_id(
    user_id
):

    conn = get_db_connection()

    user = conn.execute(
        """
        SELECT *
        FROM users
        WHERE id = ?
        """,
        (user_id,)
    ).fetchone()

    conn.close()

    return user


def update_user(
    user_id,
    **kwargs
):

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

        query = (
            query.rstrip(", ")
            + " WHERE id = ?"
        )

        params.append(user_id)

        conn.execute(
            query,
            params
        )

        conn.commit()

        return True

    except Exception as e:

        print(
            f"Error updating user: {e}"
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

        return f(
            *args,
            **kwargs
        )

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

            return f(
                *args,
                **kwargs
            )

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

        else:

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

        first_name = request.form[
            "first_name"
        ]

        last_name = request.form[
            "last_name"
        ]

        birth_date = request.form[
            "birth_date"
        ]

        hospital = request.form[
            "hospital"
        ]

        phone = request.form[
            "phone"
        ]

        user_type = request.form[
            "user_type"
        ]

        email = request.form[
            "email"
        ]

        password = request.form[
            "password"
        ]

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

        else:

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
def admin_update_user(
    user_id
):

    data = request.get_json()

    field = data.get("field")

    value = data.get("value")

    if field not in [
        "role",
        "is_active"
    ]:

        return jsonify(
            {
                "success": False,
                "message": "Invalid field"
            }
        )

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

        return jsonify(
            {
                "success": True
            }
        )

    except Exception as e:

        print(
            f"Error updating user: {e}"
        )

        return jsonify(
            {
                "success": False,
                "message": str(e)
            }
        )

    finally:

        conn.close()


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

            token = secrets.token_urlsafe(
                32
            )

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
                "Password reset link has "
                "been sent to your email",
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

    if (
        request.method == "POST"
        and valid_token
    ):

        new_password = request.form[
            "password"
        ]

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

        else:

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
# ADMIN EDIT USER
# ============================================================

@app.route(
    "/admin/edit-user/<int:user_id>",
    methods=["GET", "POST"]
)
@role_required("admin")
def admin_edit_user(
    user_id
):

    user = get_user_by_id(
        user_id
    )

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
                if request.form.get(
                    "is_active"
                ) == "on"
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
# ADMIN MODEL UPDATE
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

            temp_dir = "temp_uploads"

            os.makedirs(
                temp_dir,
                exist_ok=True
            )

            temp_path = os.path.join(
                temp_dir,
                filename
            )

            file.save(
                temp_path
            )

            # Verify model.
            try:

                test_model = load_model(
                    temp_path,
                    compile=False
                )

                # Basic shape test.
                if model_type == "covid_xray":

                    test_input = np.random.rand(
                        1,
                        150,
                        150,
                        3
                    )

                elif model_type == "lung_cancer_xray":

                    test_input = np.random.rand(
                        1,
                        256,
                        256,
                        3
                    )

                elif model_type == "malaria":

                    test_input = np.random.rand(
                        1,
                        100,
                        100,
                        3
                    )

                else:

                    test_input = np.random.rand(
                        1,
                        224,
                        224,
                        3
                    )

                test_model.predict(
                    test_input,
                    verbose=0
                )

                del test_model

                gc.collect()

            except Exception as e:

                if os.path.exists(
                    temp_path
                ):

                    os.remove(
                        temp_path
                    )

                flash(
                    f"File is not a valid model: "
                    f"{str(e)}",
                    "danger"
                )

                return redirect(
                    url_for("update_models")
                )

            # Replace old model.
            model_path = MODEL_PATHS[
                model_type
            ]

            shutil.move(
                temp_path,
                model_path
            )

            # Remove cached model.
            unload_model(
                model_type
            )

            flash(
                f"Model {model_type} "
                "updated successfully!",
                "success"
            )

            return redirect(
                url_for("update_models")
            )

        except Exception as e:

            flash(
                f"Error updating model: {str(e)}",
                "danger"
            )

            return redirect(
                url_for("update_models")
            )

    # Model update information.
    model_updates = {}

    for model_type, path in MODEL_PATHS.items():

        if os.path.exists(path):

            model_updates[
                model_type
            ] = datetime.fromtimestamp(
                os.path.getmtime(path)
            ).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        else:

            model_updates[
                model_type
            ] = "Not downloaded"

    return render_template(
        "admin/update_models.html",

        models=list(
            MODEL_PATHS.keys()
        ),

        model_names=CLASS_NAMES,

        # This contains only currently
        # loaded models.
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
        # Model is loaded only now.
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
                predictions[
                    0
                ][predicted_index]
            )

            label_encoder = (
                load_covid_label_encoder()
            )

            try:

                predicted_label = (
                    label_encoder
                    .inverse_transform(
                        [predicted_index]
                    )[0]
                )

            except Exception:

                predicted_label = (
                    classes[predicted_index]
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


        # ----------------------------------------------------
        # BINARY MODELS
        # ----------------------------------------------------

        if model_type in [
            "malaria",
            "diabetic_retinopathy"
        ]:

            probability = float(
                predictions[0][0]
            )

            predicted_index = int(
                probability > 0.5
            )

            if predicted_index == 1:

                confidence = probability

            else:

                confidence = (
                    1.0 - probability
                )


        # ----------------------------------------------------
        # MULTI-CLASS MODELS
        # ----------------------------------------------------

        else:

            predicted_index = int(
                np.argmax(
                    predictions[0]
                )
            )

            confidence = float(
                predictions[
                    0
                ][predicted_index]
            )


        return (
            classes[predicted_index],
            float(confidence)
        )

    except Exception as e:

        print(
            f"[PREDICTION ERROR] "
            f"{model_type}: {str(e)}"
        )

        raise


# ============================================================
# PRINTER IMAGE PROCESSING
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

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
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

            "prediction":
                result[0],

            "confidence":
                float(result[1]),

            "image_path":
                saved_path
        }

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }


# ============================================================
# PRINTER API
# ============================================================

@app.route(
    "/api/process_scan",
    methods=["POST"]
)
@login_required
def process_scan():

    if "file" not in request.files:

        return jsonify(
            {
                "error":
                    "No file provided"
            }
        ), 400

    file = request.files[
        "file"
    ]

    if file.filename == "":

        return jsonify(
            {
                "error":
                    "Filename is empty"
            }
        ), 400

    filename = secure_filename(
        file.filename
    )

    temp_path = os.path.join(
        app.config["PRINTER_FOLDER"],
        f"temp_{filename}"
    )

    file.save(
        temp_path
    )

    model_type = request.form.get(
        "model_type",
        "lung_cancer_xray"
    )

    if model_type not in MODEL_FILES:

        try:
            os.remove(temp_path)
        except Exception:
            pass

        return jsonify(
            {
                "error":
                    "Invalid model type"
            }
        ), 400

    result = process_printer_image(
        model_type,
        temp_path
    )

    try:

        os.remove(
            temp_path
        )

    except Exception:

        pass

    if result["status"] == "error":

        return jsonify(
            result
        ), 500

    return jsonify(
        result
    )


# ============================================================
# OPTIONAL MODEL BUILDERS
# ============================================================

def build_lung_cancer_xray_model():

    input_shape = (
        256,
        256,
        3
    )

    inputs = Input(
        shape=input_shape
    )

    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(inputs)

    x = MaxPooling2D(
        (2, 2)
    )(x)

    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = MaxPooling2D(
        (2, 2)
    )(x)

    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = MaxPooling2D(
        (2, 2)
    )(x)

    x = Flatten()(x)

    x = Dense(
        256,
        activation="relu"
    )(x)

    x = Dropout(
        0.5
    )(x)

    outputs = Dense(
        3,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.0001
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_brain_tumor_model():

    inputs = Input(
        shape=(
            224,
            224,
            3
        )
    )

    x = Conv2D(
        32,
        (3, 3),
        activation="relu"
    )(inputs)

    x = MaxPooling2D(
        (2, 2)
    )(x)

    x = Conv2D(
        64,
        (3, 3),
        activation="relu"
    )(x)

    x = MaxPooling2D(
        (2, 2)
    )(x)

    x = Flatten()(x)

    x = Dense(
        128,
        activation="relu"
    )(x)

    outputs = Dense(
        2,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_covid_xray_model():

    inputs = Input(
        shape=(
            150,
            150,
            3
        )
    )

    x = Conv2D(
        32,
        (3, 3),
        activation="relu"
    )(inputs)

    x = MaxPooling2D(
        (2, 2)
    )(x)

    x = Conv2D(
        64,
        (3, 3),
        activation="relu"
    )(x)

    x = MaxPooling2D(
        (2, 2)
    )(x)

    x = Flatten()(x)

    x = Dense(
        128,
        activation="relu"
    )(x)

    outputs = Dense(
        3,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# HOME
# ============================================================

@app.route("/")
@login_required
def home():

    model_updates = {}

    for model_type, model_path in MODEL_PATHS.items():

        if os.path.exists(
            model_path
        ):

            model_updates[
                model_type
            ] = datetime.fromtimestamp(
                os.path.getmtime(
                    model_path
                )
            ).strftime(
                "%Y-%m-%d %H:%M"
            )

        else:

            model_updates[
                model_type
            ] = "Not downloaded"

    return render_template(
        "index.html",
        model_updates=model_updates
    )


# ============================================================
# PRINTER PAGE
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

@app.route(
    "/<model_type>"
)
@login_required
def model_home(
    model_type
):

    if model_type not in MODEL_FILES:

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
def upload_file(
    model_type
):

    if model_type not in MODEL_FILES:

        return redirect(
            request.url
        )

    if "file" not in request.files:

        return render_template(
            "error.html",
            error="No file uploaded",
            model_type=model_type,
            now=datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ), 400

    file = request.files[
        "file"
    ]

    if file.filename == "":

        return redirect(
            request.url
        )

    filename = secure_filename(
        file.filename
    )

    upload_folder = (
        app.config[
            "UPLOAD_FOLDERS"
        ][model_type]
    )

    os.makedirs(
        upload_folder,
        exist_ok=True
    )

    file_path = os.path.join(
        upload_folder,
        filename
    )

    file.save(
        file_path
    )

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

        print(
            f"[UPLOAD ERROR] {str(e)}"
        )

        return render_template(
            "error.html",

            error=(
                f"Image processing error: "
                f"{str(e)}"
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
def camera(
    model_type
):

    if model_type not in MODEL_FILES:

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

    if model_type not in MODEL_FILES:

        return redirect(
            url_for("home")
        )

    folder = app.config[
        "UPLOAD_FOLDERS"
    ][model_type]

    return send_from_directory(
        folder,
        filename
    )


# ============================================================
# MODEL STATUS API
# ============================================================

@app.route(
    "/api/models/status"
)
@login_required
def models_status():

    result = {}

    for model_type in MODEL_FILES:

        result[model_type] = {

            "downloaded":
                os.path.exists(
                    MODEL_PATHS[
                        model_type
                    ]
                ),

            "loaded":
                model_type in MODELS,

            "filename":
                MODEL_FILES[
                    model_type
                ]
        }

    return jsonify(
        result
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
# APPLICATION START
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
