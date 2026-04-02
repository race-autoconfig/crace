
"""def create_log(path,logger_name,debug_lv):
	#create the class using a logger, and the path when log file will be saved
	logger = logging.getLogger()
	logger.setLevel(0)
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	file_handler = logging.FileHandler(path)
	file_handler.setLevel(0)# minimun level to save,must be higher or equal than logger's level
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	handler = logging.StreamHandler()
	handler.setLevel(50-(debug_lv*10))#minimun level to show, must be higher or equal than logger's level
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger"""

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
