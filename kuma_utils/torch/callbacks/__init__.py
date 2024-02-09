from .base import CallbackTemplate
from .stopping import SaveEveryEpoch, EarlyStopping, CollectTopK
from .logger import TorchLogger, DummyLogger
from .snapshot import SaveSnapshot, SaveAllSnapshots, SaveAverageSnapshot