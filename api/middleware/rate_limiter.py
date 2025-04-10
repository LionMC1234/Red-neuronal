from flask import request, jsonify
from datetime import datetime, timedelta
from collections import defaultdict
from core.utilities import advanced_logger

logger = advanced_logger.AiraLogger(__name__)

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.config = {
            'limit': 100,
            'window': 60  # segundos
        }
        
    def check_limit(self):
        client_ip = request.remote_addr
        now = datetime.now()
        
        # Limpieza de registros antiguos
        self.requests[client_ip] = [
            t for t in self.requests[client_ip]
            if t > now - timedelta(seconds=self.config['window'])
        ]
        
        if len(self.requests[client_ip]) >= self.config['limit']:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return False
            
        self.requests[client_ip].append(now)
        return True