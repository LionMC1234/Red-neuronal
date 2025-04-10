from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from core.utilities import config_manager, advanced_logger

class ChatGPTAdapter:
    def __init__(self):
        self.config = config_manager.load_config()
        self.logger = advanced_logger.AiraLogger(self.__class__.__name__)
        openai.api_key = config_manager.get_secret('CHATGPT_API_KEY')
        self.model_version = "gpt-4o-mini"
        self.client = OpenAI(base_url="", api_key="")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_response(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model_version,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config['chatgpt']['temperature'],
                max_tokens=self.config['chatgpt']['max_tokens']
            )
            return response.choices[0].message.content
        except openai.error.OpenAIError as e:
            self.logger.error(f"ChatGPT API Error: {str(e)}")
            raise
        except Exception as e:
            self.logger.critical(f"Unexpected error: {str(e)}")
            raise

class HybridLearner:
    def __init__(self):
        self.aira_model = AiraModel()
        self.chatgpt = ChatGPTAdapter()
        self.logger = advanced_logger.AiraLogger(self.__class__.__name__)

    def process_request(self, prompt):
        config = config_manager.load_config()
        
        if config['model']['active_model'] == 'chatgpt':
            response = self.chatgpt.generate_response(prompt)
            self._learn_from_external(response, prompt)
            return response
        else:
            return self.aira_model.generate(prompt)

    def _learn_from_external(self, response, prompt):
        if self.config['learning']['online_learning']:
            try:
                training_data = self._prepare_training_pair(prompt, response)
                loss = self.aira_model.online_learn(training_data)
                self.logger.info(f"Online learning completed. Loss: {loss}")
            except Exception as e:
                self.logger.error(f"Learning from ChatGPT failed: {str(e)}")

    def _prepare_training_pair(self, prompt, response):
        return f"USER: {prompt}\nASSISTANT: {response}"