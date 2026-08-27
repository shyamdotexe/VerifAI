import os
import uuid

from flask import Flask, request, jsonify
from flask_cors import CORS

from app_detectors.text_detector import TextDetector
from app_detectors.video_detector import VideoDetector

from services.analyzer import UnifiedAnalyzer
from services.firebase_service import FirebaseService


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

CORS(app)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER


ALLOWED_VIDEO_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm"
}


# =========================================================
# LOAD DETECTORS
# =========================================================

print("\n========================================")
print("        STARTING VERIFAI BACKEND")
print("========================================")

print("\nLoading Text Detector...")

text_detector = TextDetector()


print("\nLoading Video Detector...")

video_detector = VideoDetector()


# =========================================================
# UNIFIED ANALYZER
# =========================================================

analyzer = UnifiedAnalyzer(

    text_detector=text_detector,

    video_detector=video_detector,

    # Teammates will be added later
    image_detector=None,
    audio_detector=None
)


# =========================================================
# FIREBASE
# =========================================================

firebase_service = FirebaseService()


print("\n========================================")
print("        VERIFAI BACKEND READY")
print("========================================\n")


# =========================================================
# HOME
# =========================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "status": "online",

        "service": "VerifAI API",

        "message":
            "VerifAI backend is running."

    })


# =========================================================
# HEALTH
# =========================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "healthy",

        "service": "VerifAI",

        "detectors": {

            "text":
                text_detector is not None,

            "video":
                video_detector is not None,

            "image":
                False,

            "audio":
                False
        }

    })


# =========================================================
# TEXT ANALYSIS
# =========================================================

@app.route(
    "/api/analyze/text",
    methods=["POST"]
)
def analyze_text():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "error":
                    "Request body is required."
            }), 400

        text = data.get(
            "text"
        )

        if not text or not text.strip():

            return jsonify({
                "error":
                    "Text cannot be empty."
            }), 400

        # ---------------------------------------------
        # Unified Analyzer
        # ---------------------------------------------

        result = analyzer.analyze(
            "text",
            text
        )

        # ---------------------------------------------
        # Save result
        # ---------------------------------------------

        firebase_service.save_analysis(
            result
        )

        return jsonify(
            result
        ), 200

    except Exception as error:

        print(
            "Text analysis error:",
            error
        )

        return jsonify({

            "error":
                str(error)

        }), 500


# =========================================================
# VIDEO ANALYSIS
# =========================================================

@app.route(
    "/api/analyze/video",
    methods=["POST"]
)
def analyze_video():

    video_path = None

    try:

        # ---------------------------------------------
        # Check upload
        # ---------------------------------------------

        if "video" not in request.files:

            return jsonify({

                "error":
                    "No video file provided."

            }), 400

        video = request.files[
            "video"
        ]

        if not video.filename:

            return jsonify({

                "error":
                    "No video selected."

            }), 400

        # ---------------------------------------------
        # Validate extension
        # ---------------------------------------------

        extension = (
            video.filename
            .rsplit(".", 1)[-1]
            .lower()
        )

        if extension not in (
            ALLOWED_VIDEO_EXTENSIONS
        ):

            return jsonify({

                "error":
                    f"Unsupported video format: "
                    f".{extension}"

            }), 400

        # ---------------------------------------------
        # Generate safe filename
        # ---------------------------------------------

        filename = (
            f"{uuid.uuid4().hex}"
            f".{extension}"
        )

        video_path = os.path.join(

            app.config[
                "UPLOAD_FOLDER"
            ],

            filename
        )

        video.save(
            video_path
        )

        print(
            f"Video uploaded: {video_path}"
        )

        # ---------------------------------------------
        # Analyze
        # ---------------------------------------------

        result = analyzer.analyze(

            "video",

            video_path
        )

        # ---------------------------------------------
        # Save to Firebase
        # ---------------------------------------------

        firebase_service.save_analysis(
            result
        )

        return jsonify(
            result
        ), 200

    except Exception as error:

        print(
            "Video analysis error:",
            error
        )

        return jsonify({

            "error":
                str(error)

        }), 500

    finally:

        # ---------------------------------------------
        # Delete temporary video
        # ---------------------------------------------

        if (
            video_path
            and os.path.exists(video_path)
        ):

            try:

                os.remove(
                    video_path
                )

            except OSError:

                pass


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )