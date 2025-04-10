from flask import request, Response, jsonify
from flask_jwt_extended import jwt_required
from core.neural_engine import HybridLearner
from core.utilities import config_manager, advanced_logger
from ..middleware import auth, rate_limiter
import json

logger = advanced_logger.AiraLogger(__name__)
limiter = rate_limiter.RateLimiter()
learner = HybridLearner()

def init_chat_endpoints(app):
    @app.route('/v1/chat', methods=['POST'])
    @jwt_required()
    @limiter.check_limit
    def chat_handler():
        try:
            data = request.get_json()
            prompt = data['prompt']
            stream = data.get('stream', False)
            
            if stream:
                return Response(stream_generator(prompt), mimetype='text/event-stream')
            else:
                response = learner.process_request(prompt)
                log_interaction(prompt, response)
                return jsonify(build_response(prompt, response))
                
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return jsonify({"error": "Processing failed"}), 500

    def stream_generator(prompt):
        try:
            queue = learner.generate_stream(prompt)
            while not queue.empty():
                yield f"data: {json.dumps({'token': queue.get()})}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield "event: error\ndata: Streaming failed\n\n"

    def log_interaction(prompt, response):
        learner.interaction_logger.log_interaction(
            input_text=prompt,
            model_output=response,
            chatgpt_output=response if config_manager.get('model/active_model') == 'chatgpt' else None
        )

    def build_response(prompt, response):
        return {
            "response": response,
            "metadata": {
                "model": config_manager.get('model/active_model'),
                "tokens": learner.aira_model.tokenizer(response, return_tensors='pt').input_ids.shape[1]
            }
        }