from logging import getLogger

from .database import SampleBufferDatabase, cleanup_sample_buffer_databases
from .filestore import SampleBufferFilestore, cleanup_sample_buffer_filestores
from .types import SampleBuffer

logger = getLogger(__name__)


def sample_buffer(location: str) -> SampleBuffer:
    # the .eval.sample format is its own durable buffer — the viewer reads
    # running and completed samples from the directory through one path
    from ..eval_sample.store import SampleDirStore, is_eval_sample_log

    if is_eval_sample_log(location):
        return SampleDirStore(location, create=False)

    try:
        return SampleBufferDatabase(location, create=False)
    except FileNotFoundError:
        return SampleBufferFilestore(location, create=False)


def running_tasks(log_dir: str) -> list[str]:
    tasks = SampleBufferDatabase.running_tasks(log_dir)
    if tasks is not None:
        return tasks
    else:
        return SampleBufferFilestore.running_tasks(log_dir) or []


def cleanup_sample_buffers(log_dir: str) -> None:
    try:
        cleanup_sample_buffer_databases()
        cleanup_sample_buffer_filestores(log_dir)
    except Exception as ex:
        logger.warning(f"Unexpected error cleaning up sample buffers: {ex}")
