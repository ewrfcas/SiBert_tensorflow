from tensorflow.python.training import session_run_hook
import numpy as np
import time
from tensorflow.python.framework import ops
import six
import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunArgs


class _HookTimer(object):
    """Base timer for determining when Hooks should trigger.

    Should not be instantiated directly.
    """

    def __init__(self):
        pass

    def reset(self):
        """Resets the timer."""
        pass

    def should_trigger_for_step(self, step):
        """Return true if the timer should trigger for the specified step."""
        raise NotImplementedError

    def update_last_triggered_step(self, step):
        """Update the last triggered time and step number.

        Args:
          step: The current step.

        Returns:
          A pair `(elapsed_time, elapsed_steps)`, where `elapsed_time` is the number
          of seconds between the current trigger and the last one (a float), and
          `elapsed_steps` is the number of steps between the current trigger and
          the last one. Both values will be set to `None` on the first trigger.
        """
        raise NotImplementedError

    def last_triggered_step(self):
        """Returns the last triggered time step or None if never triggered."""
        raise NotImplementedError


class NeverTriggerTimer(_HookTimer):
    """Timer that never triggers."""

    def should_trigger_for_step(self, step):
        _ = step
        return False

    def update_last_triggered_step(self, step):
        _ = step
        return (None, None)

    def last_triggered_step(self):
        return None


class SecondOrStepTimer(_HookTimer):
    """Timer that triggers at most once every N seconds or once every N steps.
    """

    def __init__(self, every_secs=None, every_steps=None):
        self.reset()
        self._every_secs = every_secs
        self._every_steps = every_steps

        if self._every_secs is None and self._every_steps is None:
            raise ValueError("Either every_secs or every_steps should be provided.")
        if (self._every_secs is not None) and (self._every_steps is not None):
            raise ValueError("Can not provide both every_secs and every_steps.")

        super(SecondOrStepTimer, self).__init__()

    def reset(self):
        self._last_triggered_step = None
        self._last_triggered_time = None

    def should_trigger_for_step(self, step):
        """Return true if the timer should trigger for the specified step.

        Args:
          step: Training step to trigger on.

        Returns:
          True if the difference between the current time and the time of the last
          trigger exceeds `every_secs`, or if the difference between the current
          step and the last triggered step exceeds `every_steps`. False otherwise.
        """
        if self._last_triggered_step is None:
            return True

        if self._last_triggered_step == step:
            return False

        if self._every_secs is not None:
            if time.time() >= self._last_triggered_time + self._every_secs:
                return True

        if self._every_steps is not None:
            if step >= self._last_triggered_step + self._every_steps:
                return True

        return False

    def update_last_triggered_step(self, step):
        current_time = time.time()
        if self._last_triggered_time is None:
            elapsed_secs = None
            elapsed_steps = None
        else:
            elapsed_secs = current_time - self._last_triggered_time
            elapsed_steps = step - self._last_triggered_step

        self._last_triggered_time = current_time
        self._last_triggered_step = step
        return (elapsed_secs, elapsed_steps)

    def last_triggered_step(self):
        return self._last_triggered_step


class LoggingTensorHook(session_run_hook.SessionRunHook):
    """Prints the given tensors every N local steps, every N seconds, or at end.

    The tensors will be printed to the log, with `INFO` severity. If you are not
    seeing the logs, you might want to add the following line after your imports:

    ```python
      tf.logging.set_verbosity(tf.logging.INFO)
    ```

    Note that if `at_end` is True, `tensors` should not include any tensor
    whose evaluation produces a side effect such as consuming additional inputs.
    """

    def __init__(self, tensors, every_n_iter=None, every_n_secs=None,
                 at_end=False, formatter=None, save_file=None):
        """Initializes a `LoggingTensorHook`.

        Args:
          tensors: `dict` that maps string-valued tags to tensors/tensor names,
              or `iterable` of tensors/tensor names.
          every_n_iter: `int`, print the values of `tensors` once every N local
              steps taken on the current worker.
          every_n_secs: `int` or `float`, print the values of `tensors` once every N
              seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
              provided.
          at_end: `bool` specifying whether to print the values of `tensors` at the
              end of the run.
          formatter: function, takes dict of `tag`->`Tensor` and returns a string.
              If `None` uses default printing all tensors.

        Raises:
          ValueError: if `every_n_iter` is non-positive.
        """
        only_log_at_end = (
                at_end and (every_n_iter is None) and (every_n_secs is None))
        if (not only_log_at_end and
                (every_n_iter is None) == (every_n_secs is None)):
            raise ValueError(
                "either at_end and/or exactly one of every_n_iter and every_n_secs "
                "must be provided.")
        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError("invalid every_n_iter=%s." % every_n_iter)
        if not isinstance(tensors, dict):
            self._tag_order = tensors
            tensors = {item: item for item in tensors}
        else:
            self._tag_order = sorted(tensors.keys())
        self._tensors = tensors
        self._formatter = formatter
        self._timer = (NeverTriggerTimer() if only_log_at_end else
                       SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter))
        self._log_at_end = at_end

        self.save_file = save_file

    def begin(self):
        self._timer.reset()
        self._iter_count = 0
        # Convert names to tensors if given
        self._current_tensors = {tag: _as_graph_element(tensor)
                                 for (tag, tensor) in self._tensors.items()}

    def before_run(self, run_context):  # pylint: disable=unused-argument
        if run_context._original_args.feed_dict is None:
            return None
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            return SessionRunArgs(self._current_tensors)
        else:
            return None

    def _log_tensors(self, tensor_values):
        original = np.get_printoptions()
        np.set_printoptions(suppress=True)
        elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
        if self._formatter:
            print(self._formatter(tensor_values))
        else:
            stats = []
            for tag in self._tag_order:
                if tag == 'lr':
                    show_str_ = "%s = %e" % (tag, tensor_values[tag])
                elif 'Loss' in tag and 'scale' not in tag:
                    show_str_ = "%s = %.4f" % (tag, tensor_values[tag])
                elif 'Acc' in tag:
                    show_str_ = "%s = %.2f" % (tag, tensor_values[tag])
                else:
                    show_str_ = "%s = %s" % (tag, tensor_values[tag])
                stats.append(show_str_)
            if elapsed_secs is not None:
                stats.append("(%.3f sec)" % elapsed_secs)
            print(" ".join(stats))
            if self.save_file:
                with open(self.save_file, 'a') as f:
                    f.write(str(time.strftime("%H:%M:%S")) + " " + (" ".join(stats)) + '\n')
        np.set_printoptions(**original)

    def after_run(self, run_context, run_values):
        if run_context._original_args.feed_dict is not None:
            _ = run_context
            if self._should_trigger:
                self._log_tensors(run_values.results)
            self._iter_count += 1

    def end(self, session):
        if self._log_at_end:
            values = session.run(self._current_tensors)
            self._log_tensors(values)


def _as_graph_element(obj):
    """Retrieves Graph element."""
    graph = ops.get_default_graph()
    if not isinstance(obj, six.string_types):
        if not hasattr(obj, "graph") or obj.graph != graph:
            raise ValueError("Passed %s should have graph attribute that is equal "
                             "to current graph %s." % (obj, graph))
        return obj
    if ":" in obj:
        element = graph.as_graph_element(obj)
    else:
        element = graph.as_graph_element(obj + ":0")
        # Check that there is no :1 (e.g. it's single output).
        try:
            graph.as_graph_element(obj + ":1")
        except (KeyError, ValueError):
            pass
        else:
            raise ValueError("Name %s is ambiguous, "
                             "as this `Operation` has multiple outputs "
                             "(at least 2)." % obj)
    return element
