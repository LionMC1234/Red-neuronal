from flask import Flask
from flask_jwt_extended import JWTManager
from .endpoints import chat, training
from .middleware import auth, rate_limiter
from core.utilities import config_manager, metrics

def create_app():
    app = Flask(__name__)
    
    # Configuración
    app.config['JWT_SECRET_KEY'] = config_manager.get('auth/jwt_secret')
    app.config['JWT_ALGORITHM'] = config_manager.get('auth/algorithm')
    
    # Inicialización de componentes
    JWTManager(app)
    metrics.start_exporter(port=9090)
    
    # Registro de endpoints
    chat.init_chat_endpoints(app)
    training.init_training_endpoints(app)
    
    # Middleware
    app.before_request(auth.auth_required)
    app.before_request(rate_limiter.check_limit)
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=5000)