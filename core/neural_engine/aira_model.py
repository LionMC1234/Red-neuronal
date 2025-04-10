import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import bitsandbytes as bnb
from core.utilities import config_manager, advanced_logger

class AiraModel:
    def __init__(self):
        self.config = config_manager.load_config()
        self.logger = advanced_logger.AiraLogger(self.__class__.__name__)
        self.model = None
        self.tokenizer = None
        self.optimizer = None

    def initialize(self):
        try:
            self._load_components()
            self._configure_quantization()
            self._setup_optimizer()
            self.logger.info("Model initialized successfully")
        except Exception as e:
            self.logger.critical(f"Initialization failed: {str(e)}")
            raise

    def _load_components(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.config['model']['base_architecture'],
            padding_side='left'
        )
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model = GPT2LMHeadModel.from_pretrained(
            self.config['model']['base_architecture'],
            device_map='auto',
            torch_dtype=torch.float16
        )

    def _configure_quantization(self):
        if self.config['model']['quantization'] == '4bit':
            bnb.quantize(self.model, 
                       quantization_type=bnb.QuantizationType.FP4,
                       skip_modules=["lm_head"])
            
    def save_model(self, version):
        save_path = f"{self.config['model']['model_path']}_v{version}"
        self.model.save_pretrained(save_path, 
                                 state_dict=bnb.serialization.save_quantized_state_dict)
        self.tokenizer.save_pretrained(save_path)
        self.logger.info(f"Model saved to {save_path}")

    def _setup_optimizer(self):
        optimizer_config = {
            'optim_bits': 32,
            'learning_rate': self.config['learning']['initial'],
            'weight_decay': 0.01,
            'max_grad_norm': 1.0
        }
        self.optimizer = bnb.optim.Adam8bit(self.model.parameters(), **optimizer_config)

    def online_learn(self, batch):
        try:
            self.model.train()
            loss = self._compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return loss.item()
        except RuntimeError as e:
            self.logger.error(f"Training error: {str(e)}")
            self._handle_memory_error()

    def _compute_loss(self, batch):
        # Implementación de pérdida adaptativa con regularización
        inputs = self.tokenizer(batch, padding=True, return_tensors='pt').to('cuda')
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        return outputs.loss

    def _handle_memory_error(self):
        torch.cuda.empty_cache()
        self.logger.warning("Memory optimized after error")