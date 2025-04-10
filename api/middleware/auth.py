from functools import wraps
from flask import request, jsonify
import jwt
from core.utilities import config_manager, advanced_logger

logger = advanced_logger.AiraLogger(__name__)

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        config = config_manager.load_config()
        token = request.headers.get('Authorization')
        
        if not token:
            logger.warning("Missing auth token")
            return jsonify({"error": "Authorization required"}), 401
            
        try:
            token = token.split()[1]
            decoded = jwt.decode(
                token,
                config['auth']['jwt_secret'],
                algorithms=[config['auth']['algorithm']]
            )
            request.user = decoded['sub']
        except jwt.ExpiredSignatureError:
            logger.warning("Expired token")
            return jsonify({"error": "Token expired"}), 401
        except Exception as e:
            logger.error(f"Invalid token: {str(e)}")
            return jsonify({"error": "Invalid token"}), 401
            
        return f(*args, **kwargs)
    return decorated