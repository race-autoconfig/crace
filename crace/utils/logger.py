import os
import re
import logging
from crace.errors import FileError


def setup_race_loggers(options, check: bool=False):
    recovery_folder = options.recoveryDir.value
    setup_logger('asyncio', options.logDir.value + '/asyncio_log.log', level=options.debugLevel.value, recovery_folder=recovery_folder)

    if not check:
        setup_logger('slice', options.logDir.value + '/slice.log', level=options.debugLevel.value, recovery_folder=recovery_folder)

        if options.logLevel.value > 0:
            setup_logger('execution', options.logDir.value + '/execution_log.log', level=options.debugLevel.value, recovery_folder=recovery_folder)
            setup_logger('race_log', options.logDir.value + '/race_log.log', level=options.debugLevel.value, recovery_folder=recovery_folder)

        if not options.readlogs.value:
            print("# Creating log with log_level {}, debug_level {} {}".format(options.logLevel.value, options.debugLevel.value, 50-(options.debugLevel.value*10)))

        if options.mpi.value and options.logLevel.value >= 4:
            setup_logger('mpi', options.logDir.value + '/mpi_log.log', level=options.debugLevel.value, recovery_folder=recovery_folder)

def setup_logger(logger_name, log_file, level, recovery_folder=None):
    l = logging.getLogger(logger_name)

    if l.handlers: return

    l.propagate = False

    formatter = logging.Formatter('%(asctime)s - %(message)s')
    clean_formatter = CleanFormatter('%(asctime)s - %(message)s')

    if recovery_folder is None:
        filehandler = logging.FileHandler(log_file, mode='w')
    else:
        filehandler = logging.FileHandler(log_file, mode='a')
    if logger_name != 'slice':
        if logger_name == 'asyncio':
            filehandler.setFormatter(clean_formatter)
        else:
            filehandler.setFormatter(formatter)
    filehandler.setLevel(0)

    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    streamhandler.setLevel(50-(level*10))

    l.setLevel(logging.DEBUG)
    l.addHandler(filehandler)
    l.addHandler(streamhandler)

    # if recovery_folder is not None:
    #     l.info(f"Recovery from {recovery_folder}")


ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
OSC_ESCAPE = re.compile(r'\x1b\]8;;')
ESC_BACK = re.compile(r'\x1b\\')
BROKEN_ANSI = re.compile(r'\[[0-9;]*m')

def clean_text(text):
    text = ANSI_ESCAPE.sub('', text)
    text = OSC_ESCAPE.sub('', text)
    text = ESC_BACK.sub('', text)
    text = BROKEN_ANSI.sub('', text)
    return text

class CleanFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        return clean_text(msg)


"""
debug level equivalences from crace and logging

debug lv crace | logging lv
---------------|-----------
5              | 0   notset
4              | 10  debug
3              | 20  info
2              | 30  warning
1              | 40  error
0              | 50  critical
"""

def get_logger(dir: str) -> str:
    """
    Check the correct path including all necessary log files when loading
    from a provided folder.
    
    :param dir: Path provided by user
    :return: Folder including all log files
    :rtype: str
    """
    from pathlib import Path

    logger = []
    required = ['parameters', 'config', 'exps_fin']
    for root, dirs, files in os.walk(dir):
        l = [Path(f).stem for f in files]
        if all(n in l for n in required):
            logger.append(root)

    if len(logger) == 0:
        raise FileError(f"One crace run including all {', '.join(required)} log files must be provided.")
    elif len(logger) == 1:
        return logger[0]
    else:
        raise FileError(f"The provided path including log folders from at least two crace runs, \n"
                        f"please specify the correct run you want to load.")

