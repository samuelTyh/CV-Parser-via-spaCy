import os
from flask import Flask
from werkzeug.middleware.shared_data import SharedDataMiddleware
from app import bp, config


app = Flask(__name__)
app.config.from_object(config.FlaskConfig)
app.add_url_rule('/uploads/<filename>', 'uploaded_file', build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  os.getcwd() + config.FlaskConfig.UPLOAD_FOLDER
})

app.register_blueprint(bp)

if __name__ == '__main__':
    app.run(host="0.0.0.0", use_reloader=False)
