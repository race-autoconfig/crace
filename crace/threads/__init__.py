import threading
import multiprocessing


def _process_wrapped(queue, func, args):
    """
    runs a function and gives the output to a multiprocessing queue

    :param queue:   execution queue given by parent thread
    :param args:    arguments for the function
    """
    ret = func(*args)
    queue.put(ret)


class JobThread(threading.Thread):

    def __init__(self, target_fn, callback_fn, args=()):
        """
        Thread implementation that runs a parallel process with a given
        method. When the process finishes runs a callback giving as parameter
        the output of the process.

        :param target_fn:   Function that runs in a parallel process. It must be
                            picklable,
        :param callback_fn: Callable that is invoked after target process. It is
                            executed in a thread and is able to edit data in the
                            main program.
        :param args:        Arguments to target, Must be an iterable
        """
        self._stopEvent = multiprocessing.Event()
        self._func = target_fn
        self._callback = callback_fn
        self._args = args
        self._result = multiprocessing.Queue()
        self._process = multiprocessing.Process(
            target=_process_wrapped, daemon=True, args=[
                self._result, self._func, self._args])

        threading.Thread.__init__(self, daemon=True)

    def run(self):
        """
        tries to start a parallel process with the target function. once
        started joins it, so the thread is suspended until the process
        finishes or is terminated by the main process. After that tries to
        run the callback callable if it was given. the return value is given to
        the callback function (assuming there is one), if there is no return
        value or the process ended abruptly the callback is invoked with None
        as parameter.
        """
        try:
            import os
            if self._process:
                self._process.start()
                self._process.join()
                if self._result.empty:
                    self._result.put(None)
                if self._callback:
                    self._callback(self._result.get())
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            # del self._func, self._args, self._process, self._callback, \
            #     self._result
            pass

    def join(self, timeout=None):
        threading.Thread.join(self, timeout)

    def terminate(self):
        """
        Tries to end the parallel process if it is in execution. after the
        thread is canceled the callback function is called and executed with
        None as parameter. If finishing the process is considered in the
        execution, then the callback function should be able to accept None
        as parameter.
        """
        if hasattr(self, '_process'):
            self._process.terminate()
            threading.Thread.join(self, None)


def target():
    import time
    import os
    print('Child: PID: ', os.getpid())
    print("Child: en proceso paralelo. sleeping 10s")
    time.sleep(10)
    print('Child: saliendo de proceso paralelo')


def callback(result):
    print('dentro de callback')
    import time
    import os
    time.sleep(5)
    print('Cback: PID: ', os.getpid())
    x = 0
    while x < 30:
        print(x)
        x += 1
    print('saliendo de callback')


if __name__ == "__main__":
    import os

    print('Main PID: ', os.getpid())
    thread = JobThread(target_fn=target, callback_fn=callback)
    print(thread.is_alive())
    thread.start()
    print(thread.is_alive())
    import time

    time.sleep(2)
    print('Main: killing process')
    thread.terminate()
    print('Main: estado de thread.is_alive: ', thread.is_alive())
    print('Main: saliendo del proceso principal')
