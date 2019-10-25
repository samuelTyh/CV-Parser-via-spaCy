class HyperParameter:
    test_size = 0.3
    n_iter = 300
    early_stopping = 20


class FlaskConfig:
    UPLOAD_FOLDER = "/app/uploaded"
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024
    JSONIFY_PRETTYPRINT_REGULAR = True
