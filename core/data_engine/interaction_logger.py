import sqlalchemy as sa
from sqlalchemy.orm import declarative_base
from contextlib import contextmanager
from core.utilities.advanced_logger import AiraLogger

Base = declarative_base()
logger = AiraLogger(__name__)

class Interaction(Base):
    __tablename__ = 'interactions'
    
    id = sa.Column(sa.Integer, primary_key=True)
    input_text = sa.Column(sa.Text, nullable=False)
    model_output = sa.Column(sa.Text)
    chatgpt_output = sa.Column(sa.Text)
    timestamp = sa.Column(sa.DateTime, server_default=sa.func.now())
    processed = sa.Column(sa.Boolean, default=False)

class InteractionLogger:
    def __init__(self, db_url):
        self.engine = sa.create_engine(db_url)
        self.Session = sa.orm.sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self.buffer = []
        self.buffer_size = 100

    @contextmanager
    def session_scope(self):
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()

    def log_interaction(self, input_text, model_output=None, chatgpt_output=None):
        interaction = Interaction(
            input_text=input_text,
            model_output=model_output,
            chatgpt_output=chatgpt_output
        )
        self.buffer.append(interaction)
        
        if len(self.buffer) >= self.buffer_size:
            self.flush_buffer()

    def flush_buffer(self):
        with self.session_scope() as session:
            session.bulk_save_objects(self.buffer)
            self.buffer.clear()
            logger.info(f"Saved {len(self.buffer)} interactions to DB")

    def get_training_batch(self, batch_size):
        with self.session_scope() as session:
            query = session.query(Interaction).filter_by(processed=False).limit(batch_size)
            batch = query.all()
            query.update({Interaction.processed: True}, synchronize_session=False)
            return batch