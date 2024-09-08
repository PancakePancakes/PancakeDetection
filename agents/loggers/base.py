import time
import logging
from enum import Enum
from typing import Optional, Dict, List, Callable
from datetime import datetime, timezone, timedelta


class TrainingEvent(Enum):
    """
    An enumeration of events that occur during the training process.

    Attributes:
        EPOCH_START (int): Event triggered at the start of each epoch.
        EPOCH_END (int): Event triggered at the end of each epoch.
        BATCH_START (int): Event triggered at the start of each batch.
        BATCH_END (int): Event triggered at the end of each batch.
        FORWARD_PASS (int): Event triggered after the forward pass.
        BACKWARD_PASS (int): Event triggered after the backward pass.
        OPTIMIZATION_STEP (int): Event triggered after the optimization step.
        VALIDATION_START (int): Event triggered at the start of validation.
        VALIDATION_BATCH_END (int): Event triggered at the end of each validation batch.
        VALIDATION_END (int): Event triggered at the end of validation.
        CHECKPOINT_SAVE (int): Event triggered when a checkpoint is saved.
        CHECKPOINT_LOAD (int): Event triggered when a checkpoint is loaded.
        EARLY_STOPPING (int): Event triggered when early stopping is activated.
        LEARNING_RATE_SCHEDULE (int): Event triggered when the learning rate is updated.
    """
    EPOCH_START = 101
    EPOCH_END = 102
    BATCH_START = 103
    BATCH_END = 104
    FORWARD_PASS = 105
    BACKWARD_PASS = 106
    OPTIMIZATION_STEP = 107
    VALIDATION_START = 108
    VALIDATION_BATCH_END = 109
    VALIDATION_END = 110
    CHECKPOINT_SAVE = 111
    CHECKPOINT_LOAD = 112
    EARLY_STOPPING = 113
    LEARNING_RATE_SCHEDULE = 114


class TrainLogger:
    """
    A logger for handling training events. Each event has a corresponding method
    that gets called when the event occurs.
    """

    def __init__(
            self,
            max_epochs: Optional[int],
            logger: logging.Logger = logging.getLogger(__name__),
            level: int = logging.INFO,
            handlers: Optional[list[logging.Handler]] = None,
            time_format: str = "%Y-%m-%d %H:%M:%S.%milliseconds",
            time_delta: Optional[int] = None,
    ):
        # Register event handlers
        self.event_handlers = {
            TrainingEvent.EPOCH_START: self.on_epoch_start,
            TrainingEvent.EPOCH_END: self.on_epoch_end,
            TrainingEvent.BATCH_START: self.on_batch_start,
            TrainingEvent.BATCH_END: self.on_batch_end,
            TrainingEvent.FORWARD_PASS: self.on_forward_pass,
            TrainingEvent.BACKWARD_PASS: self.on_backward_pass,
            TrainingEvent.OPTIMIZATION_STEP: self.on_optimization_step,
            TrainingEvent.VALIDATION_START: self.on_validation_start,
            TrainingEvent.VALIDATION_BATCH_END: self.on_validation_batch_end,
            TrainingEvent.VALIDATION_END: self.on_validation_end,
            TrainingEvent.CHECKPOINT_SAVE: self.on_checkpoint_save,
            TrainingEvent.CHECKPOINT_LOAD: self.on_checkpoint_load,
            TrainingEvent.EARLY_STOPPING: self.on_early_stopping,
            TrainingEvent.LEARNING_RATE_SCHEDULE: self.on_learning_rate_schedule,
        }
        self.event_callbacks: Dict[TrainingEvent, List[Callable]] = {}

        # Set up logger
        self.logger = logger
        self.logger.setLevel(level)

        if handlers:
            for handler in handlers:
                self.logger.addHandler(handler)

        # Set up time
        self.time_format = time_format
        self.time_delta = time_delta

        # Copy metadata
        self.max_epochs = max_epochs

        # Records
        self.epoch_start_times: List[time.time] = []  # eg: [e1, e2, ...]
        self.epoch_end_times: List[time.time] = []
        self.batches_start_times: List[List[time.time]] = []  # eg: [[e1b1, e1b2], [e2b1, e2b2, ...], ...]
        self.batches_end_times: List[List[time.time]] = []
        self.forward_pass_times: List[List[time.time]] = []

        # Current epoch state
        self.current_epoch: Optional[int] = None
        self.current_training_loader_len: Optional[int] = None
        self.current_training_batch_size: Optional[int] = None
        self.current_training_batches: Optional[int] = None
        self.current_validation_loader_len: Optional[int] = None
        self.current_validation_batch_size: Optional[int] = None
        self.current_validation_batches: Optional[int] = None

        # Current batch state
        self.current_batch: Optional[int] = None

    def __call__(self, event: TrainingEvent, *args, **kwargs):
        """
        Calls the event handler for the given event.

        Args:
            event (TrainingEvent): The event to handle.
            *args: Additional positional arguments for the event handler.
            **kwargs: Additional keyword arguments for the event handler.
        """
        handler = self.event_handlers.get(event)
        if handler:
            handler(*args, **kwargs)
        else:
            print(f"No handler for event: {event}")

    def on_event(self, event: TrainingEvent, *args, **kwargs):
        """
        An alias for __call__. Handles the given event.

        Args:
            event (TrainingEvent): The event to handle.
            *args: Additional positional arguments for the event handler.
            **kwargs: Additional keyword arguments for the event handler.
        """
        self(event, *args, **kwargs)

    def format_time(self, timestamp: float) -> str:
        """
        Formats a timestamp according to the specified time format and time delta.

        This method supports a custom format specifier `%milliseconds` to include
        milliseconds in the formatted time string.

        Args:
            timestamp (float): The timestamp to format, typically from time.time().

        Returns:
            str: The formatted timestamp string, including milliseconds if specified.
        """
        if "%milliseconds" in self.time_format:
            ms = int((timestamp - int(timestamp)) * 1000)
            time_format = self.time_format.replace("%milliseconds", f"{ms:03d}")
        else:
            time_format = self.time_format

        dt = (
            datetime.fromtimestamp(timestamp, tz=timezone(timedelta(hours=self.time_delta)))
            if self.time_delta is not None
            else datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone()
        )
        return dt.strftime(time_format)

    def register_callback(self, event: TrainingEvent, callback: Callable):
        if not callable(callback):
            raise ValueError("Callback must be callable")
        if event not in self.event_callbacks:
            self.event_callbacks[event] = []
        self.event_callbacks[event].append(callback)
        self.logger.debug(f"Registered callback for event: {event}")

    def on_epoch_start(
            self,
            epoch: int,
            train_batches: Optional[int] = None,
            valid_batches: Optional[int] = None,
            start_time: Optional[time.time] = None,
            train_loader_length: Optional[int] = None,
            train_batch_size: Optional[int] = None,
            valid_loader_length: Optional[int] = None,
            valid_batch_size: Optional[int] = None,
            *args,
            **kwargs
    ):
        # Check validation parameters
        if epoch < 0:
            raise ValueError("Epoch must be a non-negative integer.")

        start_time = time.time() if start_time is None else start_time
        train_batches = (
            train_loader_length // train_batch_size + 1
            if train_batches and train_loader_length and train_batch_size
            else train_batches
        )
        valid_batches = (
            valid_loader_length // valid_batch_size + 1
            if valid_batches and valid_loader_length and valid_batch_size
            else valid_batches
        )

        # Set current epoch state
        self.current_epoch = epoch
        self.current_training_loader_len = train_loader_length
        self.current_training_batch_size = train_batch_size
        self.current_training_batches = train_batches
        self.current_validation_loader_len = valid_loader_length
        self.current_validation_batch_size = valid_batch_size
        self.current_validation_batches = valid_batches

        # Set epoch start time
        if len(self.epoch_start_times) <= epoch:
            self.epoch_start_times.extend([-1] * (epoch - len(self.epoch_start_times) + 1))
        self.epoch_start_times[epoch] = start_time

        # Log
        self.logger.info(f"Epoch {epoch} (of {self.max_epochs}) started at {self.format_time(start_time)} "
                         f"with {(train_batches or '[unknown]')} training batches "
                         f"and {(valid_batches or '[unknown]')} validation batches.")

        # Call callbacks
        callbacks = self.event_callbacks.get(TrainingEvent.EPOCH_START, None)
        if callbacks is None or len(callbacks) == 0:
            return
        for callback in self.event_callbacks.get(TrainingEvent.EPOCH_START, []):
            self.logger.debug(f"Calling callback \"{callback.__name__}\"")
            callback(epoch=epoch, start_time=start_time, train_loader_length=train_loader_length,
                     train_batch_size=train_batch_size, train_batches=train_batches,
                     valid_loader_length=valid_loader_length, valid_batch_size=valid_batch_size,
                     valid_batches=valid_batches, *args, **kwargs)

        return

    def on_batch_start(
            self,
            batch: int,
            epoch: Optional[int] = None,
            batch_size: Optional[int] = None,
            start_time: Optional[time.time] = None,
            *args,
            **kwargs
    ):
        # check validation
        if batch < 0:
            raise ValueError("Batch must be a non-negative integer.")

        # initialize parameters
        start_time = start_time or time.time()
        batch_size = batch_size or self.current_training_batch_size
        if epoch is not None and epoch != self.current_epoch:
            self.current_epoch = epoch
        elif epoch is None and self.current_epoch is None:
            epoch = 0
            self.current_epoch = 0
        elif epoch is None:
            epoch = self.current_epoch

        # save start time
        if len(self.batches_start_times) <= epoch:
            self.batches_start_times.extend([[]] * (epoch - len(self.batches_start_times) + 1))
            self.batches_start_times[epoch] = []
        if len(self.batches_start_times[epoch]) <= batch:
            self.batches_start_times[epoch].extend([-1] * (batch - len(self.batches_start_times[epoch]) + 1))
        self.batches_start_times[epoch][batch] = start_time

        # log
        self.logger.info(f"Batch {batch}/{(self.current_training_batches or '[unknown]')}) "
                         f"started at {self.format_time(start_time)} "
                         f"with {(batch_size or '[unknown]')} samples")

        # call callbacks
        callbacks = self.event_callbacks.get(TrainingEvent.BATCH_START, None)
        if callbacks is None or len(callbacks) == 0:
            return
        for callback in self.event_callbacks.get(TrainingEvent.BATCH_START, []):
            self.logger.debug(f"Calling callback \"{callback.__name__}\"")
            callback(batch=batch, epoch=epoch, batch_size=batch_size, start_time=start_time, *args, **kwargs)

        # Set current batch state
        self.current_batch = batch

    def on_forward_pass(
            self,
            loss: float,
            batch: Optional[int] = None,
            epoch: Optional[int] = None,
            *args,
            **kwargs
    ):
        pass

    def on_backward_pass(self, *args, **kwargs):
        pass

    def on_optimization_step(self, *args, **kwargs):
        pass

    def on_validation_start(self, *args, **kwargs):
        pass

    def on_validation_batch_end(self, *args, **kwargs):
        pass

    def on_validation_end(self, *args, **kwargs):
        pass

    def on_checkpoint_save(self, *args, **kwargs):
        pass

    def on_batch_end(self, *args, **kwargs):
        pass

    def on_epoch_end(
            self,
            *args,
            **kwargs
    ):
        pass

    def on_checkpoint_load(self, *args, **kwargs):
        pass

    def on_early_stopping(
            self,
            reason: Optional[str] = None,
            *args,
            **kwargs
    ):
        # Log
        self.logger.warning(f"Training stopped due to {(reason or '[unknown]')}.")

        # Call callbacks
        callbacks = self.event_callbacks.get(TrainingEvent.EARLY_STOPPING, None)
        if callbacks is None or len(callbacks) == 0:
            return
        for callback in self.event_callbacks.get(TrainingEvent.EARLY_STOPPING, []):
            self.logger.debug(f"Calling callback \"{callback.__name__}\"")
            callback(reason=reason, *args, **kwargs)

    def on_learning_rate_schedule(self, *args, **kwargs):
        pass
