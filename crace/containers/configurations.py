import io
import os
import csv
import json
import copy
import pandas as pd

from crace.errors import FileError, CraceError
from crace.utils import Reader, ConditionalReturn, pandas_future_ctx
from crace.containers.parameters import Parameters
from crace.containers.forbidden_expressions import ForbiddenExpressions


class ConfigurationEntry:
    """
    Class that represents a configuration
    :ivar id: configuration ID
    :ivar parent_id: parent configuration ID
    :ivar model_id: model ID
    :ivar param_values: Dictionary of parameter values (keys are parameter names)
    :ivar alive: Boolean, indicates if the configuration is alive
    :ivar cmd: Command line string of the configuration
    """
    def __init__(self, config_id, parent_id, model_id, param_values, cmd, parameters: Parameters = None):
        """
        Store the information of a configuration. Ths class should be
        created only through the Configurations class.

        :param config_id:    Unique ID of the configuration
        :param parent_id:    ID of the parent configuration to the
                                current if it has.
        :param param_values: Dictionary of parameter values (keys are parameter names)
        :param cmd:          Contains the switch of each parameter
                                associated to its value.
        :param parameters:   Contains the name and value of each argument of
                                the configuration

        """
        self.id = config_id
        self.parent_id = parent_id
        self.model_id = model_id
        self.param_values = param_values
        self.alive = True
        self.cmd = cmd

    def get_hash(self):
        """
        Determine a hash for the configuration based on the parameters given
        in the initialization

        :return:    Value of the hash for the Configuration
        """
        hashing = 0
        counter = 1
        for key, value in self.param_values.items():
            hashing += hash(value) * counter + 1
            counter += 1
        return hashing
    
    def __hash__(self) -> int:
        return self.get_hash()

    def __eq__(self, other):
        """
        Comparison method overwritten for the class. It is responsible for
        when comparing two classes of this type by means of their parameters

        :param other:   Object to be compared with the current one

        :return:    True if both objects are of the same class and their
                    parameters have the same values
        """
        if not isinstance(other, ConfigurationEntry):
            return False
        return self.cmd == other.cmd

    def get_command(self):
        """
        Get the command line format of a configuration
        :return: command line string
        """
        return self.cmd

    def get_id(self):
        """
        Get ID of a configuration
        :return: integer ID
        """
        return self.id

    def set_id(self, new_id: int):
        """
        Set the ID  of a configuration
        :param new_id: integer ID
        """
        self.id = new_id

    def get_parent_id(self):
        """
        Get the ID of the parent configuration
        :return: Integer ID of parent
        """
        return self.parent_id

    def set_model_id(self, new_id: int):
        """
        Get the ID of the parent configuration
        :return: Integer ID of parent
        """
        self.model_id = new_id

    def get_model_id(self):
        """
        Get the ID of the parent configuration
        :return: Integer ID of parent
        """
        return self.model_id

    def get_values(self, add_metadata=True):
        """
        Get configuration information (id, parameter values, parent id)
        :param add_metadata: Boolean, indicates if metadata (.ID, .PARENT) fields should be added
        :return: Dictionary of parameter values with parameter names as keys, plus .ID and .PARENT if required
        """
        row = copy.deepcopy(self.param_values)
        if add_metadata:
            row[".ID"] = self.id
            row[".PARENT"] = self.parent_id
        return row

    def is_alive(self):
        """
        Indicates if a configuration is alive
        """
        return self.alive

    def set_discarded(self):
        """
        Set the configuration as discarded
        """
        self.alive = False

    def print_cmd(self):
        print(self.cmd)

    def get_parameter_value(self, name):
        return self.param_values[name]

class Configurations:
    """
    Class configurations manages the set of configurations
    :ivar parameters: object Parameters.
    :ivar all_configurations: Dictionary of Configuration objects, keys are configuration ids
    :ivar n_config: Number of configurations in all_configurations
    :ivar alive_ids: List of ids which are considered alive
    :ivar hash_configurations: list of configuration hashes used maninly to find repeated configurations
    """

    def __init__(self, parameters: Parameters, exec_dir: str = None, recovery_folder: str = None,
                 conf_repair: str = None, debug_level=0, load_folder: str = None, log_base_name: str = None):
        """
        Class containing and generating configurations. It allows to create
        configurations in a random way and store them together with a
        identifier and a hash. Deliver an object that is not of the type
        Parameters will raise an exception of type ValueError.

        :param parameters: Parameter Object with its parameters loaded
        :param exec_dir: Folder where configuration logs should be created
        :param recovery_folder: File from which configurations should be recovered
        """
        if not isinstance(parameters, Parameters):
            raise ValueError('Must receive a Parameters object')
        log_folder = None
        if not load_folder:
            if exec_dir:
                log_folder = os.path.join(exec_dir, log_base_name)
            else:
                print('Notice: set current directory as execDir. (name here: exec_dir)')
                exec_dir = os.path.abspath('./')
                log_folder = os.path.join(exec_dir, log_base_name)
            if not os.path.exists(log_folder): os.makedirs(log_folder, exist_ok=True)

        self.parameters = parameters
        # create configurations dictionary, keys are ids
        self.all_configurations = {}
        self.n_config = 0
        self.debug_level = debug_level
        self.exec_dir = exec_dir

        # alive configurations data frame
        self.alive_ids = []

        # configurations hash
        self.hash_configurations = []

        # parameters expressions with infinity value
        self.forbidden_expressions = ForbiddenExpressions([])
        self.forbidden_ids = []
        self.conf_repair = None

        if log_folder:
            self.log_file_conf = log_folder + "/config.log"
            self.log_file_alive = log_folder + "/config_alive.log"
        # recover data if needed
        if recovery_folder is not None:
            if debug_level >= 3:
                print("# Recovering configurations ")
            log_file_conf = recovery_folder + "/config.log"
            log_file_al = recovery_folder + "/config_alive.log"
            self._load_from_log(log_file_conf, log_file_al)
            if debug_level >= 3:
                print("#  ", self.n_config, " configurations recovered, alive: ", len(self.alive_ids))
        elif load_folder is not None:
            load_file_conf = load_folder + '/config.log'
            load_file_al = load_folder + '/config_alive.log'
            self._load_from_log(load_file_conf, load_file_al)
        else:
            # initialize logs
            file = open(self.log_file_conf, 'w')
            file.close()
            file = open(self.log_file_alive, 'w')
            file.close()
        if conf_repair is not None:
            # TODO: check this!!
            exec(conf_repair, globals())
            # self.conf_repair = FrepairConfiguration
            self.conf_repair = conf_repair

    def add_from_file(self, configurations_file, size=None):
        """
        Add configurations from a configuration file.
        Impotant: Configurations should match the parameter definition provided by crace

        :param configurations_file: File name where configurations should be obtained
        :param size: Maximum number of configurations that should be obtained from the file
        """
        count_total = count_sel = 0
        config_ids = parent_ids = []

        if Reader().check_readable(configurations_file):
            list_file = Reader().get_readable_lines(configurations_file)

            # csv file
            with pandas_future_ctx():
                if any(',' in s for s in list_file):
                    reader = pd.read_csv(io.StringIO("\n".join(list_file)), dtype=str, 
                                        ).replace(["", "NULL", "none", "missing"], pd.NA).fillna("NA")
                else:
                    reader = pd.read_csv(io.StringIO("\n".join(list_file)), dtype=str, sep=r"\s+"
                                    ).replace(["", "NULL", "none", "missing"], pd.NA).fillna("NA")

            if ".ID" in reader.keys():
                config_ids = reader[".ID"].tolist()
                reader = reader.drop(columns=[".ID"])
            if ".PARENT" in reader.keys():
                parent_ids = reader[".PARENT"].tolist()
                reader = reader.drop(columns=[".PARENT"])

            param_names = reader.columns.values.tolist()
            values_list = reader.values.tolist()

            for idx, values in enumerate(values_list):
                param_values = dict(zip(param_names, values))
                count_total = count_total + 1
                param_values, cmd_line = self.parameters.parse_values(param_values, count_total)

                if self.parameters.is_forbidden(param_values):
                    print("# Discarding the provided configuration " + str(count_total) + " as it is forbidden")

                else:
                    if not config_ids:
                        config = ConfigurationEntry(self.n_config + 1, 0, 0, param_values, cmd_line, parameters=self.parameters)
                    else:
                        config = ConfigurationEntry(int(config_ids[idx]), int(parent_ids[idx]), 0, param_values, cmd_line, parameters=self.parameters)
                    count_sel = count_sel + 1
                    if not self._add_configuration(config):
                        raise CraceError("Configurations provided in configurations file should not be repeated.")

                if (size is not None) and self.n_config == size:
                    if self.debug_level >= 2:
                        print("#   configuration file has more configurations than required. Keeping the first " + str(size))
                    break

            if self.debug_level >= 2:
                print("#   adding " + str(count_sel) + "/" + str(count_total) + " initial configurations")

        else:
            raise FileError("Error reading configuration file: " + configurations_file)

        self.configurations_file = configurations_file

    def add_random(self, size=1):
        """
        Add configurations created uniformly at random over the parameter domains.

        :param size:    How many configurations are going to be added
        """
        for i in range(size):
            # FIXME: should we add a maximum number of attempts?
            # sample parameter values
            param_values, cmd_line = self.parameters.sample_uniform()

            while (self.parameters.is_forbidden(param_values) == True) or (self.is_forbidden(param_values) == True):
                param_values, cmd_line = self.parameters.sample_uniform()

            # generate the new configuration
            config = ConfigurationEntry(self.n_config + 1, 0, 0, param_values, cmd_line, parameters=self.parameters)
            self.repair_configuration_execution(config)

            # check if configuration was added successfully
            if not self._add_configuration(config):
                i -= 1

    def add_from_model(self, new_configurations: list):
        """
        Add configurations sampled from the model of another configuration.
        This function assigns an ID to each configuration and sets them as alive configurations.
        All configurations must be valid (not forbidden).

        :param new_configurations: list of configuration object instances
        :return: list of the IDs of the new configurations
        """
        if isinstance(new_configurations, ConfigurationEntry): 
            new_configurations = [new_configurations]

        new_ids = []
        for config in new_configurations:
            self.repair_configuration_execution(config)
            # assign ID to the configuration
            config.set_id(self.n_config + 1)
            # attempt to add the configuration
            #   if the configuration already exists: not added (-2)
            if self._add_configuration(config):
                new_ids.append(config.get_id())
            else:
                new_ids.append(-2)

        return new_ids

    def _add_configuration(self, config: ConfigurationEntry, add_to_log: bool = True):
        """
        Add a new configuration to the configuration set

        :param config:   Configuration object instance
        :param add_to_log: Boolean that indicates if a configuration should be
                           added to the log
        :return:    True if the configuration could be added or False if already
                    is there and thus the configuration is not added
        """

        # check if configuration already exists
        if self.configuration_exist(config):
            return False

        # add it to the configuration list
        self.all_configurations[config.get_id()] = config

        # set it as alive
        # FIXME: also set the configuration status as alive
        self.alive_ids.append(config.get_id())

        # add configuration hash
        self.hash_configurations.append(config.get_hash())

        # increment the configuration counter
        self.n_config += 1

        # log the configuration if required
        if add_to_log:
            self.new_log(self.log_file_conf, self.log_file_alive, config)

        return True

    def configuration_exist(self, configuration):
        """
        Check if a Configuration with the same parameters as the one given is
        already added in the set

        :param configuration: Configuration object instance
        :return:    True if there is a configuration with the same
                    parameters stored. False otherwise
        """
        if configuration.get_hash() in self.hash_configurations:
            return True
        return False

    def is_alive(self, id_config):
        """
        Check if a configuration is alive given its ID

        :param id_config: ID of the configuration
        :return:    True if the configuration is alive or False otherwise
        """
        return self.all_configurations[id_config].is_alive() and (id_config in self.alive_ids)

    def get_alive_ids(self):
        """
        Gets a list of IDs that corresponds to the alive configurations

        :return: list of IDs
        """
        return self.alive_ids

    def get_alive(self, as_list=True):
        """
        Get alive configurations object instances

        :param as_list: obtain the configurations as a list
        :return: Dictionary of alive configurations object instances where the key is the ID
        """
        if as_list:
            conf = [self.all_configurations[key] for key in self.alive_ids]
        else:
            conf = {key: self.all_configurations[key] for key in self.alive_ids}
        return conf

    def get_alive_model_ids(self, as_list=True):
        """
        Get alive configurations object instances

        :param as_list: obtain the configurations as a list
        :return: Dictionary of alive configurations object instances where the key is the ID
        """
        if as_list:
            model_ids = [self.all_configurations[key].model_id for key in self.alive_ids]
            model_ids = list(set(model_ids))
        else:
            model_ids = {key: self.all_configurations[key].model_id for key in self.alive_ids}
        return model_ids

    def get_configurations(self, ids_config, as_list=True):
        """
        Get a set of configurations based on the IDs

        :param ids_config: list of configuration IDs
        :param as_list: Boolean that indicates if the configurations should be provided as a list
        :param as_list: is related to the return
        :return: List or Dictionary  of selected configurations where the key is the ID
        :return: List[Configurations] or Configurations
        """

        if as_list:
            conf = [self.all_configurations[x] for x in ids_config]
        else:
            conf = {x: self.all_configurations[x] for x in ids_config}
        return conf

    def get_configuration(self, id_config):
        """
        Get a configuration based on the ID

        :param id_config: configuration ID
        :return: Selected configuration where the key is the ID
        """
        assert id_config in self.all_configurations.keys(), "Attempt to get unknown configuration(id): " + str(id_config)
        return self.all_configurations[id_config]

    def discard_configuration(self, id_config):
        """
        Set the configuration status alive as False

        :param id_config: ID of the configuration to set as not alive
        """

        if id_config not in self.alive_ids and self.debug_level >= 2:
            print("# Configuration " + id_config + " was already discarded. Ignoring action.")

        # set own state as not alive
        self.all_configurations[id_config].set_discarded()

        # get index in the list of alive configurations
        index = self.alive_ids.index(id_config)

        # delete its entry from alive list
        del self.alive_ids[index]

        # log as discarded
        self.log_update_alive(self.log_file_alive)

    def discard_configurations(self, ids_config):
        """
        Set the configurations status alive as False

        :param ids_config: ID of the configuration to set as not alive
        """
        if isinstance(ids_config, int):
            ids_config = [ids_config]

        # discard all configurations
        for id_config in ids_config:
            self.discard_configuration(id_config)

    def print_all(self):
        """
        Shows the information of the configurations created in table format.
        Indicating the values of the parameters, ID, ID of the parent
        configuration and hash of the configuration
        """
        config_list = [x.get_values() for x in self.all_configurations.values()]
        df = pd.DataFrame(config_list)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            return ConditionalReturn(all=df)

    def print_alive(self):
        """
        Shows the information of the configurations created in table format.
        Indicating the values of the parameters, ID, ID of the parent
        configuration and hash of the configuration
        """
        config_list = [self.all_configurations[x].get_values() for x in self.alive_ids]
        df = pd.DataFrame(config_list)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
            return ConditionalReturn(alive=df)

    def print_to_file(self, ids: list, ofile: str):
        """
        Prints a set of condigurations to a file
        :param ids: List of ids to print
        :param ofile: String of the path to a file where configurations should printed
        """
        config_list = [self.all_configurations[x].get_values(add_metadata=True) for x in ids]
        df = pd.DataFrame(config_list)
        df.to_csv(ofile, mode='w', index=False, header=True)

    def new_log(self, log_file: str, log_alive_file: str, config: ConfigurationEntry):
        """
        Add a configuration to a csv log file

        :param log_file: File where the log should be written
        :param log_alive_file: File where the alive IDs will be logged
        :param config: Configuration object instance that should logged
        """
        # check if csv exist
        if not self.log_exist(log_file):
            raise FileError("Log file " + log_file + " was not found")

        # create a dataframe from configuration
        param_dict = {key: config.param_values[key] for key in self.parameters.sorted_parameters}
        config_row = {"configuration_id": config.get_id(),
                      "parent_id": config.get_parent_id(),
                      "model_id": config.get_model_id(),
                      **param_dict,
                      "cmd": config.cmd,
                      "is_alive": config.alive}

        # write log of the configuration
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=config_row.keys())
            if os.path.getsize(log_file) == 0:
                writer.writeheader()
            writer.writerow(config_row)

        # re-write alive ids
        #FIXME: this could just append the id at the end
        file = open(log_alive_file, 'w')
        file.writelines(",".join([str(x) for x in self.alive_ids]))
        file.close()

    def log_exist(self, path):
        """
        Check if a log file exists
        :param path: Log file path
        """
        # FIXME: check if readable too
        if not os.path.isfile(path):
            return False
        return True

    def _load_from_log(self, log_file: str, log_alive_file: str = None):
        """
        Loads all configurations from a log File
        :param log_file: Log configurations file path
        :param log_alive_file: Log of alive configurations id
        """
        param_keys = None
        dtype_mapping = {"configuration_id": int,
                         "parent_id": int,
                         "model_id": int,
                         "cmd": str,
                         "is_alive": bool}

        for k in self.parameters.sorted_parameters:
            param_type = self.parameters.all_parameters[k].type
            if param_type in ("i", "i,log", "r", "r,log"):
                dtype_mapping[k] = float
            elif param_type in ("c", "o"):
                dtype_mapping[k] = str

        if self.log_exist(log_file):
            with open(log_file, newline='') as f:
                _int = int
                try:
                    # load the header from the provided file
                    # NOTE: rewind the pointer
                    header = pd.read_csv(f, nrows=0).columns.values.tolist()
                    f.seek(0)

                    if header[0] != 'configuration_id':
                        # the provided file is not in csv format
                        # goto except
                        raise FileError('Provided file is not in csv format.')

                    # load configurations from csv file
                    # load log file and fill nan with None
                    # NOTE: csv format of configurations
                    #   configuration_id,parent_id,model_id,...,cmd,is_alive
                    reader = pd.read_csv(f, dtype=dtype_mapping, na_values=["", "NULL", "none", "NA", "missing", None])
                    param_keys = list(reader.columns)[3:-2] # parameters of the provided scenario

                    int_keys = {k for k in param_keys if self.parameters.all_parameters[k].type in ("i", "i,log")}
                    # NaN -> None
                    reader = reader.where(pd.notna(reader), None)

                    for dr in reader.itertuples(index=False):
                        # transfer values based on the type
                        param_dict = {k: None if pd.isna(v) else _int(v) if k in int_keys else v
                                      for k in param_keys for v in (getattr(dr,k),)}
                        configuration = ConfigurationEntry(config_id=_int(dr.configuration_id),
                                                           parent_id=_int(dr.parent_id),
                                                           model_id=_int(dr.model_id),
                                                           param_values=param_dict,
                                                           cmd=dr.cmd)
                        self._add_configuration(configuration, False)

                except FileError:
                    # load configurations from the dict file
                    reader = csv.reader(f)
                    _json_loads = json.loads
                    for dr in reader:
                        # create configuration
                        try:
                            model_id = _int(dr[2])
                            param_idx = 3
                            cmd_idx = 4
                        except ValueError:
                            model_id = _int(dr[0])
                            param_idx = 2
                            cmd_idx = 3

                        configuration = ConfigurationEntry(config_id=_int(dr[0]),
                                                    parent_id=_int(dr[1]),
                                                    model_id=model_id,
                                                    param_values=_json_loads(dr[param_idx]),
                                                    cmd=dr[cmd_idx])
                        self._add_configuration(configuration, False)

                except pd.errors.EmptyDataError:
                    # empty file
                    pass

        else:
            raise CraceError("Attempt to load configurations from a non-existent file")

        if (log_alive_file is not None) and self.log_exist(log_alive_file):
            # if alive log is provided then we update alive configurations
            with open(log_alive_file, newline='') as f:
                reader = csv.reader(f, delimiter=",")
                # first line should have alive ids separated by ,
                self.alive_ids = [int(x) for x in list(reader)[0]]
            for config_id, config in self.all_configurations.items():
                if config_id not in self.alive_ids:
                    config.set_discarded()

    def log_update_alive(self, log_alive_file: str):
        """
        Change a configuration state to discarded and add it to the discarded log

        :param log_alive_file: Log file path
        """
        if self.log_exist(log_alive_file):
            # re-write alive ids which should be already
            file = open(log_alive_file, 'w')
            file.writelines(",".join([str(x) for x in self.alive_ids]))
            file.close()

    def discard_forbidden(self, configuration_id):
        """
        If the value obtained from the experiment is infinite, the configuration of that experiment is deleted. 
        Finally the values of the configuration parameters are saved

        :param configuration_id: Experiment configuration id
        :return: none
        """
        self.forbidden_ids.append(configuration_id)
        self.discard_configuration(configuration_id)

    def repair_configuration_execution(self, configuration):
        """
        check is a function to repair configuration exist and execute in that case
        otherwise return the configuration unchanged
        
        :param configuration : Configuration to repair
        : return: the configuration repaired, if the function exists, 
        in any other the configuration unchanged
        """
        if self.conf_repair is not None:
            return self.conf_repair(configuration)
        return configuration

    def get_cmd_from_configuration_id(self, id):
        return self.all_configurations[id].get_command()

    def update_status_for_recovering(self, scheduled: list, model_config: dict):
        """
        update parameter values for recovering

        :param scheduled: list of configuration ids provided by scheduled jobs
        """
        self.alive_ids = list(set(scheduled + self.get_alive_ids()))
        for x in self.get_alive_ids():
            self.all_configurations[x].alive = True

        # update model id based on model_config from models
        for model_id, configs in model_config.items():
            if len(configs) > 0:
                for id in configs:
                    config = self.get_configuration(id)
                    config.set_model_id(int(model_id))
