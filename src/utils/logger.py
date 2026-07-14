import queue
import threading
from functools import partial

import jax
from streamlet.utils.typing import PyTree


class Async:
    def __init__(self, logger):
        self.logger = logger
        self.queue = queue.Queue()

        self.worker = threading.Thread(target=self.dispatch, daemon=False)
        self.worker.start()

    def dispatch(self):
        while True:
            task = self.queue.get()

            if task is None:
                self.queue.task_done()
                break

            task()
            self.queue.task_done()

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        data = jax.device_get(data)
        steps = jax.device_get(steps)
        self.queue.put(partial(self.logger.log, data, steps, **kwargs))

    def log_summary(self, data: PyTree, **kwargs) -> None:
        data = jax.device_get(data)
        self.queue.put(partial(self.logger.log_summary, data, **kwargs))

    def log_artifact(self, state: PyTree, step: int, **kwargs) -> None:
        state = jax.device_get(state)
        self.queue.put(partial(self.logger.log_artifact, state, step, **kwargs))

    def finish(self) -> None:
        self.queue.put(None)
        self.queue.join()
        self.worker.join()
        self.logger.finish()
