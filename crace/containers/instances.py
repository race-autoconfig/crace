import os
import csv
import sys
import random
import collections
import pandas as pd

from random import shuffle

from crace.errors import CraceError
from crace.utils import ConditionalReturn, Reader


"""
Instance object tuple(3). 
 (instance_id, path, seed)
"""
InstanceEntry = collections.namedtuple("InstanceEntry", ["instance_id", "path", "seed"])


class Instances(Reader):
    """
    Class Instances contains a set of Instance objects. Each instance has an
        ID, a path and a seed related. The seeds are generated with a random
        generator that gives an int between 0 and sys.maxsize.

    :ivar names: List of instance names (file paths or strings).
    :ivar instances: List of Instance objects that are used already in crace.
    :ivar instances_id: List of instance ids that are used already in crace.
    :ivar instances_all: List of instance objects initialized for training.
    :ivar instances_ids_all: List of instance ids initialized for training.
    :ivar instances_num: Number of instance objects initialized for training.
    :ivar fixsize: Size of provided training instances.
    :ivar size: Size of instances in current stream.
    :ivar _next_instance: ID of the next instance to be used in crace.
    :ivar tnames: List of test instance names (file paths or strings)
    :ivar tinstances: List of Instance objects used for testing.
    :ivar tinstances_id: List of test instance ids.
    :ivar tsize: Number of test instances in the set.
    """

    def __init__(self, instances_list: list = None, instances_dir: str = None, instances_file: str = None,
                 tinstances_list: list = None, tinstances_dir: str = None, tinstances_file: str = None,
                 exec_dir: str = None, deterministic: bool = False, recovery_folder: str = None,
                 debug_level: int = 0, seed: int = None, shuffle_ins: bool = False, sample_ins: bool = True, 
                 silent: bool = False, load_folder = None, log_base_name: str = None):
        """
        Creates an instance set.
        :param instances_list: List of instances
        :param instances_dir: Directory path where instance files can be found
        :param instances_file: File path to the file where instances are listed
        :param tinstances_list: List of test instances
        :param tinstances_dir: Directory path where test instance files can be found
        :param tinstances_file: File path to the file where test instances are listed
        :param exec_dir: String of the path to the folder where logs should be created
        :param deterministic: Boolean that indicated if the scenario is deterministic
        :param recovery_folder: Folder path where recovery files can be found
        :param debug_level: Debug level to output reports
        """
        log_folder = None
        if not load_folder:
            if exec_dir:
                log_folder = os.path.join(exec_dir, log_base_name)
            else:
                print('Notice: set current directory as execDir. (name here: exec_dir)')
                exec_dir = os.path.abspath('./')
                log_folder = os.path.join(exec_dir, log_base_name)
            if not os.path.exists(log_folder): os.makedirs(log_folder, exist_ok=True)
        else:
            recovery_folder = load_folder

        self.seed = seed
        self.shuffle_ins = shuffle_ins
        self.sample_ins = sample_ins

        # training instance data
        self.names = []
        # list of instances (for non deterministic scenarios this is instances + seed)
        self.instances = []
        self.instances_ids = []
        self.instances_all = []
        self.instances_ids_all = []
        self.instances_num = 0  # the size of sampled instances
        self.fixsize = 0    # the size of provided instances
        self.size = 0       # the size of current instance stream
        self._next_instance = 0
        self.instances_log = None
        self.next_log = None

        # test instance data
        self.tnames = []
        self.tinstances = []
        self.tinstances_ids = []
        self.tsize = 0
        self.tinstances_log = None

        # counter for each instance
        self.instances_config_num = {}

        self.debug_level = debug_level
        self.exec_dir = exec_dir

        dirs = [instances_dir, instances_file, tinstances_dir, tinstances_file]
        for i, item in enumerate(dirs):
            if item and not os.path.isabs(item):
                dirs[i] = os.path.abspath(item)
        instances_dir, instances_file, tinstances_dir, tinstances_file = dirs

        if log_folder:
            self.instances_log = log_folder + "/instances.log"
            self.next_log = log_folder + "/instances_next.log"

        try:
            if recovery_folder is not None:
                # if debug_level >= 2:
                #     print("# Recovering instances")
                self.load_from_log(recovery_folder)
                if debug_level >= 2:
                    print(f"#  {self.size} instances recovered")
            else:
                if not silent:
                    self._read_training_instances(instances_list=instances_list,
                                                instances_file=instances_file,
                                                instances_dir=instances_dir,
                                                log_folder=log_folder)
                    self._create_training_log(log_folder)
                    self._add_all_training_instances()

                if tinstances_list is not None or tinstances_file is not None or tinstances_dir is not None:
                    if not silent:
                        self._read_test_instances(instances_list=tinstances_list,
                                                instances_file=tinstances_file,
                                                instances_dir=tinstances_dir)
                        self._create_test_log(log_folder)
                        self._add_all_test_instances()

        except Exception as e:
            print(f"\nERROR: There was an error while loading instances")
            print(e)
            sys.tracebacklimit = 0
            raise e
        
        self.instances_dir = instances_dir
        self.instances_file = instances_file
        self.tinstances_dir = tinstances_dir
        self.tinstances_file = tinstances_file


    ############################################################
    ##               initialization functions                 ##
    ############################################################

    def _read_training_instances(self, instances_list= None, instances_file:str =None, instances_dir: str= None, log_folder: str= None):
        """
        Reads the training instances and creates variables required to handle them

        :param instances_list: List of instances provided by the user
        :param instances_file: String of the path to a file where instances are listed
        :param instances_dir: String of the path to a directory where instances are locates

        :return: None
        """
        # add instances directly from list
        if instances_list is not None:
            assert isinstance(instances_list, list), \
                "Instances must be provided in a list when using argument instances_list"
            self.names = instances_list

        elif instances_file is not None:
            if self.debug_level >= 2:
                print("# Reading instances from file: " + instances_file)

            # read instances from file
            if self.check_readable(instances_file):
                lines = self.get_readable_lines(instances_file)
            else:
                raise CraceError("Cannot read instance file:" + instances_file)

            if instances_dir is not None:
                # append directory to instances
                lines = [instances_dir + "/" + x for x in lines]

            self.names = self.names + lines

        elif instances_dir is not None:
            if self.debug_level >= 2:
                print("# Reading instances from folder: " + instances_dir)

            # read instances from directory
            if self.check_dir_readable(instances_dir):
                files = os.listdir(instances_dir)
                path_files = [os.path.join(instances_dir, x) for x in files]
                self.names = self.names + path_files

        else:
            if log_folder:
                instances_dir = os.path.join(os.path.dirname(log_folder), './Instances')
                if not os.path.exists(instances_dir):
                    sys.tracebacklimit = 0
                    raise CraceError("The default './Instances' is not exist in current directory.\n"
                                     "instances_list, instances_dir or instances_file must be provided to build instances object.")
                self._read_training_instances(instances_dir=instances_dir)

            else:
                raise CraceError(
                    "instances_list, instances_dir or instances_file must be provided to build instances object.")

        self.fixsize = len(self.names)

    def _create_training_log(self, log_folder: str):
        """
        Creates and initializes the training instances log

        :param log_folder: String of the path of a directory where logs should be created
        :return: None
        """
        assert self.names is not None, "Training instances are nor initialized"
        assert log_folder is not None, "log_folder is not provided"

        # Instance log
        # FIXME: add also deterministic variable log!
        self.instances_log = log_folder + "/instances.log"
        df = pd.DataFrame([self.names])
        df.to_csv(self.instances_log, mode='w', index=False, header=False)

        self.next_log = log_folder + "/instances_next.log"
        with open(self.next_log, mode='w') as f:
            f.write(str(self._next_instance))

    def _add_all_training_instances(self):
        """
        Creates training instances by adding each of them with an associated seed

        :return: None
        """
        assert len(self.names) > 0, "Training instances are not loaded"

        # add one version of the instance initially
        # new_names = random.sample(self.names, len(self.names))
        new_names = self.names[:]
        if self.sample_ins is True:
            random.shuffle(new_names)

        for name in new_names:
            instance = InstanceEntry(self.instances_num+1, name, random.randint(0, 9999999))
            if self.instances_log: self.print_to_log(self.instances_log, instance)
            self.instances_all.append(instance)
            self.instances_ids_all.append(instance.instance_id)
            self.instances_num += 1

    def _add_new_training_instances(self):
        """
        Adds training instances by adding each of them with an associated seed

        :return: None
        """
        index = self.size + 1
        new_names = self.names[:]
        if self.sample_ins is True:
            random.shuffle(new_names)

        for name in new_names:
            instance = InstanceEntry(index, name, random.randint(0, 9999999))
            self.print_to_log(self.instances_log, instance)
            self.instances_all.append(instance)
            self.instances_ids_all.append(instance.instance_id)
            index += 1

    def _read_test_instances(self, instances_list=None, instances_file: str = None, instances_dir: str = None):
        """
        Reads the test instances and creates variables required to handle them

        :param instances_list: List of instances provided by the user
        :param instances_file: String of the path to a file where instances are listed
        :param instances_dir: String of the path to a directory where instances are locates

        :return: None
        """
        # add instances directly from list
        if instances_list is not None:
            assert isinstance(instances_list, list), \
                "Test instances must be provided in a list when using argument instances_list"
            self.tnames = instances_list

        elif instances_file is not None:
            if self.debug_level >= 2:
                print("# Reading test instances from file: " + instances_file)

            # read instances from file
            if self.check_readable(instances_file):
                lines = self.get_readable_lines(instances_file)
            else:
                CraceError("Cannot read test instance file:" + instances_file)

            if instances_dir is not None:
                # append directory to instances
                lines = [instances_dir + "/" + x for x in lines]
            self.tnames = self.tnames + lines

        elif instances_dir is not None:
            if self.debug_level >= 2:
                print("# Reading test instances from folder: " + instances_dir)

            # read instances from directory
            if self.check_dir_readable(instances_dir):
                files = os.listdir(instances_dir)
                path_files = [os.path.join(instances_dir, x) for x in files]
                self.tnames = self.tnames + path_files

        else:
            raise CraceError(
                "Test instance list, instance directory or instances file must be provided to build instances object.")


    def _create_test_log(self, log_folder: str):
        """
        Creates and initializes the test instances log

        :param log_folder: String of the path of a directory where logs should be created
        :return: None
        """
        assert self.tnames is not None, "Test instances are nor initialized"
        assert log_folder is not None, "log_folder is not provided"

        # Instance log
        # FIXME: add also deterministic variable log!
        self.tinstances_log = log_folder + "/tinstances.log"
        df = pd.DataFrame([self.tnames])
        df.to_csv(self.tinstances_log, mode='w', index=False, header=False)

    def _add_all_test_instances(self):
        """
        Creates test instances by adding each of them with an associated seed

        :return: None
        """
        assert len(self.tnames) > 0, "Test instances are not loaded"
        # assert self.tinstances_log is not None, "Test instance log must be initialized to add instances"

        # add one version of the instance initially
        for name in self.tnames:
            instance = InstanceEntry(self.tsize+1, name, random.randint(0, 9999999))
            if self.tinstances_log: self.print_to_log(self.tinstances_log, instance)
            self.tinstances.append(instance)
            self.tinstances_ids.append(instance.instance_id)
            self.tsize += 1

    ############################################################
    ##             Training instance functions                ##
    ############################################################
    def get(self, instance_ids: list):
        """
        Returns an Instance entry

        :param instance_id: Index/ID of the instance
        :return: a list of Instance tuple with the corresponding ID
        """
        selected = []
        # assuming instance_id is index+1
        for id in instance_ids:
            index = self.instances_ids.index(id)
            selected.append(self.instances[index])
            assert selected[-1].instance_id == id, "Could not locate instance " + id
        return selected

    def new_instance(self):
        """
        Creates a new random instance (adding a seed if needed) and updates the size of the container
        """
        # selected training instances from that sampled already
        if self.size+1 not in self.instances_ids_all:
            i = int((self.size+1) / self.instances_num)
            if self.seed: random.seed((self.seed * i + 1234567) % 7654321)
            self._add_new_training_instances()

        instance = self.instances_all[self.size]

        if self.debug_level >=5:
            print("# New instance {}: {}, {}".format(instance.instance_id, instance.path, instance.seed))

        return self._add_instance(instance)
    
    def check_instances(self):
        """
        creates a new random instance from training set and test set, respectively.
        """
        name = random.choice(self.names)
        instance = InstanceEntry(1, name, random.randint(0, 9999999))
        self.instances.append(instance)
        self.instances_ids.append(instance.instance_id)

        if self.has_test():
            tname = random.choice(self.tnames)
            tinstance = InstanceEntry(2, tname, random.randint(0, 9999999))
            self.instances.append(tinstance)
            self.instances_ids.append(tinstance.instance_id)
        
    def _add_instance(self, instance):
        """
        Adds an instance to the set

        :param instance: Instance tuple
        :return: Instance ID
        """
        self.size += 1
        self._next_instance += 1
        self.instances.append(instance)
        self.instances_ids.append(instance.instance_id)
        self.instances_config_num[instance.instance_id] = 0

        # write log for next instance
        with open(self.next_log, mode='w') as f:
            f.write(str(self._next_instance))
        return instance.instance_id

    def shuffle_instances(self):
        """
        Shuffle instances already generated
        """
        shuffle(self.instances)
        self.instances_ids = [x.instance_id for x in self.instances]

    def update_config_num(self, instance_id):
        """
        Update the counter of configuration for instance in the stream
        """
        self.instances_config_num[instance_id] += 1

    def update_instances_stream(self, instance_ids):
        """
        Update the instance stream
        """
        index = [self.instances_ids.index(x) for x in instance_ids]
        for i in sorted(index, reverse=True):
            del self.instances[i]
            del self.instances_ids[i]

    def print_all(self, as_dataframe: bool=True):
        """
        Print all instances
        """
        if not as_dataframe:
            print("# Training instances:")
            print(self.instances)
            print("# Test instances:")
            print(self.tinstances)

        else:
            df1 = pd.DataFrame([{'instance_id': x.instance_id, 'instance': x.path, 'seed': x.seed} for x in self.instances])
            df1.attrs['title'] = "Training instances"
            df2 = pd.DataFrame([{'instance_id': x.instance_id, 'instance': x.path, 'seed': x.seed} for x in self.tinstances])
            df2.attrs['title'] = "Test instances"
            return ConditionalReturn(train=df1, test=df2)

    def as_dict(self, instance):
        """
        Gets an Instance entry as dictionary

        :return: Dictionary with keys: id_instance, instance_path, instance_seed
        """
        return {"id_instance": instance[0], "path": instance[1], "seed": instance[2]}

    def print_to_log(self, log_file, instance):
        """
        Print a instance to a log file

        :param log_file: File path to write log
        :param instance: Instance tuple entry
        """
        # create dataframe
        instance_df = pd.DataFrame([self.as_dict(instance)])
        instance_df.to_csv(log_file, mode='a', index=False, header=False)

    def load_from_log(self, recovery_folder):
        """
        Load a set of instances from a log file
        :param recovery_folder: File path of the log
        """
        log_file = recovery_folder + "/instances.log"
        next_log_file = recovery_folder + "/instances_next.log"
        test_log_file = recovery_folder + "/tinstances.log"

        if os.path.exists(next_log_file):
            with open(next_log_file, newline='') as f:
                data = f.readline()
                self._next_instance = int(data)

        if os.path.exists(log_file):
            with open(log_file, newline='') as f:
                reader = csv.reader(f)
                data = [tuple(row) for row in reader]

                #first line should be instances names
                self.names = list(data[0])
                self.fixsize = len(self.names)

                # add new instances
                data = data[1:]
                for i in data:
                    instance = InstanceEntry(int(i[0]), i[1], int(i[2]))
                    self.instances_all.append(instance)
                    self.instances_ids_all.append(int(i[0]))
                    self.instances_num += 1

                    if int(i[0]) <= self._next_instance:
                        self.size += 1
                        self.instances.append(instance)
                        self.instances_ids.append(instance.instance_id)
                        self.instances_config_num[instance.instance_id] = 0

        if os.path.exists(test_log_file):
            with open(test_log_file, newline='') as f:
                reader = csv.reader(f)
                data = [tuple(row) for row in reader]

                #first line should be instances names
                self.tnames = list(data[0])

                # add new instances
                data = data[1:]
                for i in data:
                    instance = InstanceEntry(int(i[0]), i[1], int(i[2]))
                    self.tinstances.append(instance)
                    self.tinstances_ids.append(int(i[0]))
                    self.tsize += 1


    ############################################################
    ##                Test instance functions                 ##
    ############################################################

    def has_test(self):
        return self.tsize>0

    ############################################################
    ##                load instance data                      ##
    ############################################################

    def get_used_instances(self, as_dataframe=True):
        """
        return used instance ids
        """
        if as_dataframe:
            ins = [self.instances_all[x-1] for x in self.instances_ids]
            df = pd.DataFrame([x._asdict() for x in ins])
            return df

        else:
            ins_dict = {x: self.instances_all[x] for x in self.instances_ids}
            return ins_dict

    def get_instance_id_seed_pairs(self, as_dataframe=True):
        """
        return instance - seed pairs
        """
        if as_dataframe:
            ins = [self.instances_all[x-1] for x in self.instances_ids]
            df = pd.DataFrame([{'instance_id': x.instance_id, 'seed': x.seed} for x in ins])
            return df

        else:
            pairs = {x: self.instances_all[x-1].seed for x in self.instances_ids}
            return pairs
