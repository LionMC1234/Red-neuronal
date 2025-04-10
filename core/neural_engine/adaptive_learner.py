import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from core.utilities import advanced_logger, config_manager

class MetaLearner:
    def __init__(self, model):
        self.config = config_manager.load_config()
        self.logger = advanced_logger.AiraLogger(self.__class__.__name__)
        self.model = model
        self.optimizer = self._configure_optimizer()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000)
        self.loss_buffer = []
        
    def _configure_optimizer(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning']['initial'],
            weight_decay=0.01,
            fused=True
        )
    
    def adaptive_learning_step(self, batch):
        try:
            self.model.train()
            loss = self._compute_adaptive_loss(batch)
            loss.backward()
            self._gradient_management()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            return loss.item()
        except RuntimeError as e:
            self.logger.error(f"Learning step failed: {str(e)}")
            self._handle_oom()
            return None

    def _compute_adaptive_loss(self, batch):
        # Implementación de pérdida con regularización adaptativa
        inputs = self.model.tokenizer(batch, return_tensors='pt', padding=True).to('cuda')
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        return outputs.loss * self._dynamic_loss_scale()

    def _dynamic_loss_scale(self):
        # Escalado dinámico basado en historial de pérdidas
        if len(self.loss_buffer) < 10:
            return 1.0
        return min(1.0, sum(self.loss_buffer[-10:]) / 10)

    def _gradient_management(self):
        # Norma de gradiente con escalado dinámico
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True
        )

    def _handle_oom(self):
        torch.cuda.empty_cache()
        self.logger.warning("OOM handled, cache cleared")