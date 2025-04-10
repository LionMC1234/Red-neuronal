from prometheus_client import start_http_server, Counter, Histogram, Gauge
from core.utilities import advanced_logger

logger = advanced_logger.AiraLogger(__name__)

class AiraMetrics:
    def __init__(self):
        # Métricas de API
        self.api_requests = Counter(
            'aira_api_requests_total',
            'Total API requests',
            ['endpoint', 'method']
        )
        
        self.response_times = Histogram(
            'aira_response_time_seconds',
            'API response times',
            ['endpoint'],
            buckets=[0.1, 0.5, 1, 2, 5]
        )
        
        # Métricas del Modelo
        self.training_loss = Gauge(
            'aira_training_loss',
            'Current training loss'
        )
        
        self.inference_latency = Histogram(
            'aira_inference_latency_seconds',
            'Model inference latency',
            buckets=[0.01, 0.05, 0.1, 0.5, 1]
        )
        
        # Métricas del Sistema
        self.memory_usage = Gauge(
            'aira_memory_usage_bytes',
            'Current memory usage'
        )
        
        self.gpu_utilization = Gauge(
            'aira_gpu_utilization_percent',
            'GPU utilization percentage'
        )
        
    def start_exporter(self, port=9090):
        start_http_server(port)
        logger.info(f"Metrics exporter started on port {port}")
        
    def track_api_request(self, endpoint, method):
        self.api_requests.labels(endpoint, method).inc()
        
    def track_training_loss(self, loss):
        self.training_loss.set(loss)
        
    def track_inference(self, latency):
        self.inference_latency.observe(latency)
        
    def update_system_metrics(self):
        # Implementar lógica de monitoreo del sistema
        pass

# Singleton metrics instance
metrics = AiraMetrics()