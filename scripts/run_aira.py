from core.neural_engine import AiraModel, HybridLearner
from core.data_engine import InteractionLogger
from core.utilities import config_manager

def main():
    # Inicialización del sistema
    config = config_manager.load_config()
    
    # Modelo principal
    aira = AiraModel()
    aira.initialize()
    
    # Sistema de aprendizaje híbrido
    learner = HybridLearner()
    
    # Logger de interacciones
    interaction_logger = InteractionLogger(config['database']['url'])
    
    # Bucle principal de ejecución
    while True:
        try:
            user_input = get_user_input()
            response = learner.process_request(user_input)
            
            interaction_logger.log_interaction(
                input_text=user_input,
                model_output=response if config['model']['active_model'] == 'local' else None,
                chatgpt_output=response if config['model']['active_model'] == 'chatgpt' else None
            )
            
            display_response(response)
            
            if config['learning']['online_learning']:
                learner.aira_model.online_learn_from_buffer()
                
        except KeyboardInterrupt:
            aira.save_model('latest')
            break

if __name__ == "__main__":
    main()