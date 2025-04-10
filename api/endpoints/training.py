from flask import request, jsonify
from flask_jwt_extended import jwt_required
from core.data_engine import TrainingBatcher
from core.neural_engine import AiraModel
from core.utilities import config_manager, advanced_logger
import threading

logger = advanced_logger.AiraLogger(__name__)
batcher = TrainingBatcher()
aira_model = AiraModel()

def init_training_endpoints(app):
    @app.route('/v1/train', methods=['POST'])
    @jwt_required()
    def train_model():
        try:
            thread = threading.Thread(target=start_training)
            thread.start()
            return jsonify({"status": "Training started"}), 202
        except Exception as e:
            logger.error(f"Training init failed: {str(e)}")
            return jsonify({"error": "Training failed to start"}), 500

    def start_training():
        try:
            aira_model.initialize()
            dataset = batcher.prepare_dataset(
                config_manager.get('training/batch_size'),
                config_manager.get('training/validation_split')
            )
            aira_model.train(dataset)
            aira_model.save_model('latest')
            logger.info("Training completed successfully")
        except Exception as e:
            logger.critical(f"Training failed: {str(e)}")