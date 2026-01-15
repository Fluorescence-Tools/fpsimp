#!/usr/bin/env python3
"""
FPSIMP: Fluorescent Protein Simulation Pipeline
Main application entry point - Orchestrates modular components
"""
import os
import redis
from flask import Flask
from jinja2 import Environment, FileSystemLoader
from flask_cors import CORS
from celery import Celery

# 1. Configuration import
from config import config

# 2. Flask App Initialization
app = Flask(__name__, static_folder='../static', static_url_path='/static')
CORS(app)

# Ensure proper MIME types for ES6 modules
@app.after_request
def after_request(response):
    if response.path.endswith('.js'):
        response.mimetype = 'application/javascript'
    return response

# 3. Jinja2 Environment for static templates
template_env = Environment(loader=FileSystemLoader('/app/static'))

def render_static_template(template_name, **context):
    """Render a template from the static folder"""
    from flask import url_for as flask_url_for
    
    def url_for(endpoint, **kwargs):
        if endpoint == 'static' and 'filename' in kwargs:
            return f"/static/{kwargs['filename']}"
        return flask_url_for(endpoint, **kwargs)
        
    template = template_env.get_template(template_name)
    return template.render(url_for=url_for, **context)

# 4. Celery Initialization
celery = Celery(app.import_name)
app.config.from_object(config)
celery.config_from_object(config)

# 5. Redis Initialization
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'redis') if not os.getenv('REDIS_URL') else None,
    port=6379 if not os.getenv('REDIS_URL') else None,
    url=config.broker_url,
    db=1,
    decode_responses=True
)

# 6. Custom Jinja2 filters
from utils.general import format_file_size
app.jinja_env.filters['get_file_size'] = format_file_size

# 7. Import tasks to register them with Celery
# Important: this must happen after celery is initialized
import tasks

# 8. Register Blueprints
from routes.views import views_bp
from routes.jobs import jobs_bp
from routes.files import files_bp

app.register_blueprint(views_bp)
app.register_blueprint(jobs_bp)
app.register_blueprint(files_bp)

# 9. Additional Configuration
app.config['DISABLE_COLABFOLD'] = os.environ.get('DISABLE_COLABFOLD', 'true').lower() == 'true'

if __name__ == '__main__':
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
