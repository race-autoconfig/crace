import os
import copy
import math
import json
import random
import logging

import pandas as pd
from typing import Dict, List
from scipy.stats import truncnorm
from abc import ABC, abstractmethod

from crace.utils import truncated_normal
from crace.utils.const import DELTA, EPSILON
from crace.containers.parameters import Parameters
from crace.errors import ModelError, CraceExecutionError
from crace.containers.forbidden_expressions import ForbiddenExpressions
from crace.containers.configurations import Configurations, ConfigurationEntry


class ParameterModel(ABC):
    """
    Abstract class ParameterModel
    :var name: Parameter name
    :var model_type: Model type
    :var domain: Parameter domain
    """
    def __init__(self, name, model_type, domain, dependent=False):
        """
        Initialization of the model for selected parameter
        
        :param name: name of selected parameter
        :param model_type: type of selected parameter
        :param domain: domain of selected parameter
        :param dependent: if selected parameter is dependent on other parameter
        """
        self.name = name
        self.model_type = model_type
        self.domain = domain
        # is_dependent: default False -- is_dependent() in parameters.py
        #               True: the domain of parameter depends on the value of another parameter
        self.is_dependent = dependent

    @abstractmethod
    def sample_configuration(self, mean, domain):
        """
        Sample value from the model. The parameter value will be within the provided domain

        :param mean: Model mean
        :param domain: Current parameter domain (this might change when using dependent domains)
        :return: sampled parameter value
        """
        pass

    @abstractmethod
    def sample_uniform(self, domain):
        """
        Sample parameter value uniformly within the domain
        :param domain: Parameter domain (list [lower_bound, upper_bound])
        :return: Sampled parameter value
        """
        pass

    @abstractmethod
    def update_model(self, n_sampled, nb_new_configs, nb_model_updated, total_model_update, value, by_sampling):
        """
        Update parameter model (updates the standard deviation)

        :param n_sampled: number of configurations sampled from this model
        :param nb_new_configs: number of newly sample configurations bewteen two updates
        :param nb_model_updated: counter for model updates in current loop (separated by restart)
        :param total_model_update: counter for total model updates
        :param value: value of parameter from the selected configuration
        """
        pass
    
    @abstractmethod
    def soft_restart(self):
        """
        Restart the model for selected parameter
        Default: restart to the initial version
        """
        # FIXME: implement soft restart method
        pass

    @abstractmethod
    def print(self):
        pass

    @abstractmethod
    def as_dict(self):
        pass


class IntegerModel(ParameterModel):
    """
    IntegerModel class
    Model implementation for integer parameters

    :ivar model_std: Float standard deviation value
    :ivar nb_variable: Number of variable parameters to be tuned
    """
    def __init__(self, name, domain, dependent=False, model_std=None, nb_variable=1):
        """
        Creates an IntegerModel object
        :param name: Parameter name
        :param domain: Parameter domain (list [lower_value, upper_value])
        :param model_mean: Parameter value assigned as the distribution mean
        :param model_std: Standard deviation for the model
        :param nb_variable:
        """
        super().__init__(name, "i", domain, dependent)

        if model_std is None:
            self.model_std = 0.5
        else:
            if model_std < 0.0 or model_std > 1.0:
                raise ModelError("Provided sd " + str(model_std) + " for model of parameter " + self.name + " must be within [0,1]")
            self.model_std = model_std

        self.nb_variable = nb_variable

    @staticmethod
    def from_log(recovery_dict, nb_variable):
        model = IntegerModel(name=recovery_dict["name"],
                             domain=recovery_dict["domain"],
                             model_std=recovery_dict["sd"])
        model.model_std = recovery_dict["sd"]
        model.nb_variable = nb_variable
        return model

    def sample_configuration(self, mean=None, domain=None):
        if domain is None:
            current_domain = self.domain
        else:
            current_domain = domain

        if mean is not None:
            current_mean = mean
        else:
            current_mean = int((current_domain[1]+current_domain[0])/2)

        # if self.is_dependent: FIXME
        domain_std = (current_domain[1] - current_domain[0]) * self.model_std
        lower = current_domain[0]
        upper = current_domain[1] + 1
        current_mean = current_mean + 0.5
        sampled_value = truncated_normal(current_mean, sd=domain_std, low=lower, upp=upper)
        new_value = math.floor(sampled_value)
        assert (current_domain[0] <= new_value) and (new_value <= current_domain[1]), "Sampled wrong value " + \
                                                                                      str(new_value) + \
                                                                                      " for parameter " + self.name + \
                                                                                      "with domain (" + str(current_domain[0]) + \
                                                                                      "," + str(current_domain[1])
        return new_value

    def sample_uniform(domain):
        #FIXME: add proper rounding
        value = random.uniform(domain[0], domain[1])
        return int(value)

    def update_model(self, n_sampled, nb_new_configs, nb_model_updated, total_model_update, value=None, by_sampling=False):
        #FIXME: this does not make sense for the new version, change
        # self.model_std = self.model_std * math.sqrt(pow((2.0 / (n_sampled + 1.0)), (1.0 /self.nb_variable)))
        # pow((1.0 / (n_sampled + 1.0)), (1.0 /self.nb_variable))
        # math.exp(-(1.0 /self.nb_variable) * n_sampled) 

        self.model_std = self.model_std * pow((1/(nb_new_configs+1)), (1/self.nb_variable))
        self.model_std = round(self.model_std + DELTA, 4)
            
    def soft_restart(self):
        self.model_std = 0.5

    def print(self):
        print("model: " + self.name + ", type: i" + ", sd: " + str(self.model_std))

    def as_dict(self):
        return {"name": self.name, "type": "i",
                "sd": self.model_std,
                "domain": self.domain}


class IntegerLogModel(ParameterModel):
    """
    IntegerLogModel class
    Model implementation for integer parameters sampling on a logarithmic scale
    :ivar model_mean: Integer mean value
    :ivar model_std: Float standard deviation value
    :ivar nb_variable: Number of variable parameters to be tuned
    """
    def __init__(self, name, domain, dependent=False, model_std=None, nb_variable=1):
        """
        Creates an IntegerLogModel object
        :param name: Parameter name
        :param domain: Parameter domain (list [lower_value, upper_value])
        :param model_mean: Parameter value assigned as the distribution mean
        :param model_std: Standard deviation for the model
        :param nb_variable:
        """
        super().__init__(name, "i", domain, dependent)

        if model_std is None:
            self.model_std = 0.5
        else:
            if model_std < 0.0 or model_std > 1.0:
                raise ModelError("Provided sd " + str(model_std) + " for model of parameter " + self.name + " must be within [0,1]")
            self.model_std = model_std
        self.nb_variable = nb_variable

    @staticmethod
    def from_log(recovery_dict, nb_variable):
        model = IntegerLogModel(name=recovery_dict["name"],
                                domain=recovery_dict["domain"],
                                model_std=recovery_dict["sd"])
        model.model_std = recovery_dict["sd"]
        model.nb_variable = nb_variable
        return model

    def sample_configuration(self, mean=None, domain=None):
        if domain is None:
            current_domain = self.domain
        else:
            current_domain = domain

        if mean is None:
            current_mean = int((current_domain[1]+current_domain[0])/2)
        else:
            current_mean = mean

        # if current_mean >= current_domain[0] and current_mean <= current_domain[1]:
        #     current_mean = self.sample_uniform(current_domain)

        upper = current_domain[1] + 1
        current_mean = current_mean + 0.5

        t_lower = math.log(current_domain[0])
        t_upper = math.log(upper)

        t_mean = self._to_log(current_mean, lower=t_lower, upper=t_upper)
        sampled_value = truncated_normal(t_mean, sd=self.model_std, low=0, upp=1)
        new_value = self._to_value(sampled_value, lower=t_lower, upper=t_upper)

        new_value = math.floor(new_value)
        assert (current_domain[0] <= new_value) and (new_value <= current_domain[1]), "Sampled wrong value " + str(new_value) + \
                                                                    " for parameter " + self.name
        return new_value

    def sample_uniform(self, domain):
        #FIXME: add proper rounding
        value = random.uniform(domain[0], domain[1])
        return int(value)

    def update_model(self, n_sampled, nb_new_configs, nb_model_updated, total_model_update, value=None, by_sampling=False):
        #FIXME: this does not make sense for the new version, change
        # self.model_std = self.model_std * math.exp(-(1.0/self.nb_variable) * n_sampled) #pow((1 / (n_sampled + 1)), )

        self.model_std = self.model_std * pow((1/(nb_new_configs+1)), (1/self.nb_variable))
        self.model_std = round(self.model_std + DELTA, 4)

    def soft_restart(self):
        self.model_std = 0.5

    @staticmethod
    def _to_log(value, lower, upper):
        t_value = (math.log(value) - lower) / (upper - lower)
        return t_value

    @staticmethod
    def _to_value(t_value, lower, upper):
        value = math.exp(lower + (upper - lower) * t_value)
        return value

    def print(self):
        print("model: " + self.name + ", type: ilog" + ", sd: " + str(self.model_std))

    def as_dict(self):
        return {"name": self.name, "type": "ilog",
                "sd": self.model_std,
                "domain": self.domain}


class ContinuousModel(ParameterModel):
    """
    ContinuousModel class
    Model implementation for real parameters
    :ivar model_mean: Integer mean value
    :ivar model_std: Float standard deviation value
    :ivar digits: Number of digits for precision
    :ivar nb_variable: Number of variable parameters to be tuned
    """
    def __init__(self, name, domain, dependent=False, model_std=None, digits: int = 4, nb_variable=1):
        """
        Creates an ContinuousModel object
        :param name: Parameter name
        :param domain: Parameter domain (list [lower_value, upper_value])
        :param model_mean: Parameter value assigned as the distribution mean
        :param model_std: Standard deviation for the model
        :param digits: Number of digits for precision
        :param nb_variable:
        """
        super().__init__(name, "r", domain, dependent)

        if model_std is None:
            self.model_std = 0.5
        else:
            if model_std < 0.0 or model_std > 1.0:
                raise ModelError("Provided sd " + str(model_std) + " for model of parameter " + self.name + " must be within [0,1]")
            self.model_std = model_std

        self.digits = digits
        self.nb_variable = nb_variable

    @staticmethod
    def from_log(recovery_dict, nb_variable):
        model = ContinuousModel(name=recovery_dict["name"],
                                domain=recovery_dict["domain"],
                                model_std=recovery_dict["sd"],
                                digits=recovery_dict["digits"])
        model.model_std = recovery_dict["sd"]
        model.nb_variable = nb_variable
        return model

    def sample_configuration(self, mean=None, domain=None):
        if domain is None:
            current_domain = self.domain
        else:
            current_domain = domain

        if mean is None:
            current_mean = (current_domain[1]+current_domain[0])/2
        else:
            current_mean = mean

        # if current_mean >= current_domain[0] and current_mean <= current_domain[1]:
        #     current_mean = self.sample_uniform(current_domain)

        # if self.is_dependent: FIXME
        domain_std = (current_domain[1] - current_domain[0]) * self.model_std
        sample_value = truncated_normal(current_mean, sd=domain_std, low=current_domain[0], upp=current_domain[1])
        new_value = round(sample_value, self.digits)
        assert current_domain[0] <= new_value <= current_domain[1], "Sampled wrong value " + str(new_value) + \
                                                                    " for parameter " + self.name
        return new_value

    def sample_uniform(self, domain):
        # FIXME: add proper rounding
        value = random.uniform(domain[0], domain[1])
        return round(value, self.digits)

    def update_model(self, n_sampled, nb_new_configs, nb_model_updated, total_model_update, value=None, by_sampling=False):
        # FIXME: this does not make sense for the new version, change
        # self.model_std = self.model_std * math.sqrt(pow((2.0 / (n_sampled + 1.0)), (1.0 /self.nb_variable)))
        #math.exp(-(1.0/self.nb_variable) * n_sampled) #

        self.model_std = self.model_std * pow((1/(nb_new_configs+1)), (1/self.nb_variable))
        self.model_std = round(self.model_std + DELTA, self.digits)

    def soft_restart(self):
        self.model_std = 0.5

    def print(self):
        print("model: " + self.name + ", type: r" + ", sd: " + str(self.model_std))

    def as_dict(self):
        return {"name": self.name, "type": "r",
                "sd": self.model_std,
                "domain": self.domain, "digits": self.digits}


class ContinuousLogModel(ParameterModel):
    """
    ContinuousLogModel class
    Model implementation for real parameters sampled in a logarithmic scale
    :ivar model_mean: Integer mean value
    :ivar model_std: Float standard deviation value
    :ivar digits: Number of digits for precision
    :ivar nb_variable: Number of variable parameters to be tuned
    """
    def __init__(self, name, domain, dependent=False, model_std=None, digits: int = 4, nb_variable=1):
        """
        Creates an ContinuousLogModel object
        :param name: Parameter name
        :param domain: Parameter domain (list [lower_value, upper_value])
        :param model_mean: Parameter value assigned as the distribution mean
        :param model_std: Standard deviation for the model
        :param digits: Number of digits for precision
        :param nb_variable:
        """
        super().__init__(name, "r", domain, dependent)

        if model_std is None:
            self.model_std = 0.5
        else:
            if model_std < 0.0 or model_std > 1.0:
                raise ModelError("Provided sd " + str(model_std) + " for model of parameter " + self.name + " must be within [0,1]")
            self.model_std = model_std

        self.digits = digits
        self.nb_variable = nb_variable

    @staticmethod
    def from_log(recovery_dict, nb_variable):
        model = ContinuousLogModel(name=recovery_dict["name"],
                                   domain=recovery_dict["domain"],
                                   model_std=recovery_dict["sd"],
                                   digits=recovery_dict["digits"])
        model.model_std = recovery_dict["sd"]
        model.nb_variable = nb_variable
        return model

    def sample_configuration(self, mean=None, domain=None):
        if domain is None:
            current_domain = self.domain
        else:
            current_domain = domain

        if mean is None:
            current_mean = (current_domain[1]+current_domain[0])/2
        else:
            current_mean = mean

        # if current_mean >= current_domain[0] and current_mean <= current_domain[1]:
        #     current_mean = self.random_uniform(current_domain)

        t_lower = math.log(current_domain[0])
        t_upper = math.log(current_domain[1])
        t_mean = self._to_log(current_mean, lower=t_lower, upper=t_upper)
        sampled_value = truncated_normal(t_mean, sd=self.model_std, low=0, upp=1)
        new_value = self._to_value(sampled_value, lower=t_lower, upper=t_upper)
        new_value = round(sampled_value.rvs(), self.digits)

        assert current_domain[0] <= new_value <= current_domain[1], "Sampled wrong value " + str(new_value) + \
                                                                    " for parameter " + self.name
        return new_value

    def sample_uniform(self, domain):
        value = random.uniform(domain[0], domain[1])
        return round(value, self.digits)

    def update_model(self, n_sampled, nb_new_configs, nb_model_updated, total_model_update, value=None, by_sampling=False):
        # self.model_std = self.model_std * pow((1 / (n_sampled + 1)), (1.0/self.nb_variable))

        self.model_std = self.model_std * pow((1/(nb_new_configs+1)), (1/self.nb_variable))
        self.model_std = round(self.model_std + DELTA, self.digits)

    def soft_restart(self):
        self.model_std = 0.5

    @staticmethod
    def _to_log(value, domain):
        t_value = (math.log(value) - domain[0]) / (domain[1] - domain[0])
        return t_value

    @staticmethod
    def _to_value(t_value, domain):
        value = math.exp(domain[0] + (domain[1] - domain[0]) * t_value)
        return value

    def print(self):
        print("model: " + self.name + ", type: rlog" + ", sd: " + str(self.model_std))

    def as_dict(self):
        return {"name": self.name, "type": "rlog",
                "sd": self.model_std,
                "domain": self.domain, "digits": self.digits}


class OrdinalModel(ParameterModel):
    """
    OrdinalModel class
    Model implementation for ordered parameters
    :ivar model_value: String value assigned as mean
    :ivar model_std: Float standard deviation value
    :ivar nb_variable: Number of variable parameters to be tuned
    """
    def __init__(self, name, domain, dependent=False, model_std=None, nb_variable=1):
        """
        Creates an OrdinalModel object
        :param name: Parameter name
        :param domain: Parameter domain (list [value1, value2, value3, ...])
        :param model_value: Parameter value assigned as the mean
        :param model_std: Standard deviation for the model
        :param nb_variable:
        """
        super().__init__(name, "o", domain, dependent)

        if model_std is None:
            self.model_std = (len(self.domain) - 1.0)/ 2.0
        else:
            if model_std < 0.0 or model_std > 1.0:
                raise ModelError("Provided sd " + str(model_std) + " for model of parameter " + self.name + " must be within [0,1]")
            self.model_std = model_std
        self.nb_variable = nb_variable

    @staticmethod
    def from_log(recovery_dict, nb_variable):
        model = OrdinalModel(name=recovery_dict["name"],
                             domain=recovery_dict["domain"],
                             model_std=recovery_dict["sd"])
        model.model_std = recovery_dict["sd"]
        model.nb_variable = nb_variable
        return model

    def sample_configuration(self, mean=None, domain=None):
        if domain is None:
            current_domain = self.domain
        else:
            current_domain = domain
            if current_domain != self.domain:
                raise ModelError("Dynamic ordinal domains are nor supported.")

        if mean is None:
            current_value = self.sample_uniform(current_domain)
        else:
            current_value = mean

        # get index in the domain
        current_mean = domain.index(current_value)

        #FIXME check this sampling
        sampled_value = math.floor(truncnorm(0, len(current_domain), loc=current_mean+0.5, scale=1))
        new_value = domain[sampled_value]
        return new_value

    def sample_uniform(self, domain):
        if domain is None:
            domain = self.domain
        if domain != self.domain:
            raise ModelError("Domain provided for sampling must be the same as the one in the parameter space")
        new_value = random.choice(domain)
        return new_value

    def update_model(self, n_sampled, nb_new_configs, nb_model_updated, total_model_update, value=None, by_sampling=False):
        # self.model_std = self.model_std *  math.sqrt(pow((2.0 / (n_sampled + 1.0)), (1.0 /self.nb_variable)))

        self.model_std = self.model_std * pow((1/(nb_new_configs+1)), (1/self.nb_variable))
        self.model_std = round(self.model_std + DELTA, self.digits)

    def soft_restart(self):
        self.model_std = 0.5

    def print(self):
        print("model: " + self.name + ", type: o" + ", sd: " + self.model_std)

    def as_dict(self):
        return {"name": self.name, "type": "o",
                "sd": self.model_std, "domain": self.domain}


class CategoricalModel(ParameterModel):
    """
    CategoricalModel class
    Model implementation for categorical parameters
    :ivar model_value: String value assigned as mean
    :ivar probabilities: List of probabilities assigned to each value in the domain
    :ivar max_probabilities: Maximum probability possible in probabilities vector
    :ivar nb_variable: Number of variable parameters to be tuned
    """
    def __init__(self, name, domain, dependent=False, nb_variable=1):
        """
        Creates an OrdinalModel object
        :param name: Parameter name
        :param domain: Parameter domain (list [value1, value2, value3, ...])
        :param model_value: Parameter value assigned as the mean
        :param max_prob: Maximum probability assigned
        """
        super().__init__(name, "c", domain, dependent)

        self.domain = domain
        self.name = name
        self.probabilities = [1.0 / len(domain)] * len(domain)
        self.nb_variable = nb_variable
        self.max_probability = pow(0.2, 1/nb_variable)

        self.check_max = False

    @staticmethod
    def from_log(recovery_dict, nb_variable):
        model = CategoricalModel(name=recovery_dict["name"], 
                                 domain=recovery_dict["domain"])
        model.probabilities = recovery_dict["probabilities"]
        model.nb_variable = nb_variable
        return model

    def sample_configuration(self, mean=None, domain=None):
        if domain is None:
            domain = self.domain
        if domain != self.domain:
            raise ModelError("Domain provided for sampling must be the same as the one in the parameter space")
        new_value = random.choices(domain, weights=self.probabilities, k=1)
        return new_value[0]

    def sample_uniform(self, domain):
        if domain is None:
            current_domain = self.domain
        else:
            current_domain = domain
            if current_domain != self.domain:
                raise ModelError("Domain provided for sampling must be the same as the one in the parameter space")

        new_value = random.choice(current_domain)
        return new_value

    def update_model(self, n_sampled, nb_new_configs, nb_model_updated, total_model_update, value=None, by_sampling=False):
        # dependent parameter
        if not value:
            return

        if not by_sampling:
            # factor is related to the times of model updated
            # total_model_update is related to the number of parameters of the provided scenario
            self.total_model_update = total_model_update
            factor = float(nb_model_updated/(total_model_update+1))
            delta_p = []
            index = self.domain.index(value)

            # To guarantee the new max_p be greater than the old max_p
            diff = (max(self.probabilities) - self.probabilities[index]) / (1 - self.probabilities[index]) + DELTA
            factor = min(max(diff, factor), 1.0-EPSILON)

            for i in range(0, len(self.domain)):
                if i == index:
                    delta_p.append(factor)
                else:
                    delta_p.append(0)
                self.probabilities[i] = self.probabilities[i] * (1 - factor) + delta_p[i]

            # check max probability and normalize the probabilities
            # gurantee the maximum probability
            if self.check_max and max(self.probabilities) > self.max_probability:
                self.probabilities = [min(x, self.max_probability) for x in self.probabilities]
                sum_prob = sum(self.probabilities) - max(self.probabilities)
                for i,x in enumerate(self.probabilities):
                    if x != max(self.probabilities):
                        self.probabilities[i] = x/sum_prob*(1-max(self.probabilities))

            # normalize the probabilities
            # gurantee the maimum and minimum probabilities
            self.probabilities = [max(EPSILON, p) for p in self.probabilities]
            s = sum(self.probabilities)
            self.probabilities = [p/s for p in self.probabilities]

            for x in self.probabilities:
                assert 0.0 < x < 1.0, f"Sampled wrong probabilitie {x} for parameter {self.name}"

    def soft_restart(self):
        self.probabilities = [1.0 / len(self.domain)] * len(self.domain)

    def print(self):
        print("model: " + self.name + ", type: c" + ", prob: " + str(self.probabilities))

    def as_dict(self):
        return {"name": self.name, "type": "c",
                "domain": self.domain,
                "probabilities": self.probabilities}


class LocalModel:
    """
    Class LocalModel
    Manages a collection of parameter models associated to a configuration

    :ivar config_id: Configuration ID
    :ivar models: Dictionary of ParameterModel objects, parameter names as keys
    :ivar n_sampled: Number of configurations sampled
    """
    def __init__(self, model_id, parameters: Parameters):
        self.models: Dict[str, ParameterModel] = {}
        self.n_sampled = 1
        self.n_updated = 0
        if parameters is not None or param_values is not None:
            for name, parameter in parameters.get_parameters().items():
                if parameter.get_type() == "i":
                    if parameter.get_transform() == "log":
                        self.models[name] = IntegerLogModel(name, parameter.get_domain(),
                                                            parameter.is_dependent(),
                                                            nb_variable=parameters.nb_variable)
                    else:
                        self.models[name] = IntegerModel(name, parameter.get_domain(),
                                                         parameter.is_dependent(),
                                                         nb_variable=parameters.nb_variable)
                elif parameter.get_type() == "r":
                    if parameter.get_transform() == "log":
                        self.models[name] = ContinuousLogModel(name, parameter.get_domain(),
                                                               parameter.is_dependent(),
                                                               digits=parameter.get_digits(),
                                                               nb_variable=parameters.nb_variable)
                    else:
                        self.models[name] = ContinuousModel(name, parameter.get_domain(),
                                                            parameter.is_dependent(),
                                                            digits=parameter.get_digits(),
                                                            nb_variable=parameters.nb_variable)
                elif parameter.get_type() == "c":
                    self.models[name] = CategoricalModel(name, parameter.get_domain(),
                                                         parameter.is_dependent(),
                                                         nb_variable=parameters.nb_variable)
                elif parameter.get_type() == "o":
                    self.models[name] = OrdinalModel(name, parameter.get_domain(),
                                                     parameter.is_dependent(),
                                                     nb_variable=parameters.nb_variable)
                else:
                    raise ModelError("Parameter type not recognized by model")

    @staticmethod
    def from_log(recovery_dict, model_id, parameters):
        model = LocalModel(model_id, parameters)
        model.models_from_log(recovery_dict=recovery_dict, nb_variable=parameters.nb_variable)
        return model

    def models_from_log(self, recovery_dict, nb_variable):
        for pname, pmodel in recovery_dict.items():
            if pmodel["type"] == "i":
                self.models[pname] = IntegerModel.from_log(pmodel, nb_variable)
            elif pmodel["type"] == "ilog":
                self.models[pname] = IntegerLogModel.from_log(pmodel, nb_variable)
            elif pmodel["type"] == "r":
                self.models[pname] = ContinuousModel.from_log(pmodel, nb_variable)
            elif pmodel["type"] == "rlog":
                self.models[pname] = ContinuousLogModel.from_log(pmodel, nb_variable)
            elif pmodel["type"] == "c":
                self.models[pname] = CategoricalModel.from_log(pmodel, nb_variable)
            elif pmodel["type"] == "o":
                self.models[pname] = OrdinalModel.from_log(pmodel, nb_variable)
            else:
                raise ModelError("Error in parameter type in the log")

    def sample_parameter(self, name, mean, domain=None):
        assert name in self.models.keys(), "Cannot find parameter model"
        value = self.models[name].sample_configuration(domain=domain, mean=mean)
        return value

    def increment_sampled(self):
        """
        Increments sampled configurations counter
        """
        self.n_sampled = self.n_sampled + 1

    def increment_updated(self):
        """
        Increments updated model counter
        """
        self.n_updated = self.n_updated + 1

    def update(self, config: ConfigurationEntry, nb_new_configs, nb_model_updated, total_model_update, by_sampling):
        """
        Update the parameter models

        :param config: the configuration whose model should be updated (configuration provides parameter values)
        :param nb_new_configs: number of newly sample configurations bewteen two updates
        :param nb_model_updated: counter for model updates in current loop (separated by restart)
        :param total_model_update: counter for total model updates
        """
        param_values = config.get_values(add_metadata=False)
        for name, model in self.models.items():
            value = param_values[name]
            model.update_model(n_sampled=self.n_sampled, nb_new_configs=nb_new_configs,
                               nb_model_updated=nb_model_updated, total_model_update=total_model_update, value=value,
                               by_sampling=by_sampling)

    def soft_restart(self):
        for _, model in self.models.items():
            model.soft_restart()

    def print(self):
        for name, model in self.models.items():
            model.print()

    def as_dict(self):
        models_dict = {}
        for pname, model in self.models.items():
            models_dict[pname] = model.as_dict()
        return models_dict

    def get_hash(self):
        """
        Determine a hash for the configuration based on the parameters given
        in the initialization

        :return:    Value of the hash for the Configuration
        """
        counter = 1
        hashing = 0

        for _, contents in self.models.items():
            model = contents.as_dict()
            for key, value in model.items():
                if key == "sd":
                    hashing += hash(value) * counter + 1
                    counter += 1
                elif key == "probabilities":
                    hashing += hash(tuple(value)) * counter + 1
                    counter += 1
        
        return hashing
    
    def __hash__(self) -> int:
        return self.get_hash()

class ProbabilisticModel:
    """
    ProbabilisticModel class

    :ivar models: Dictionary of LocalModel objects with model IDs as keys
    :ivar alive_models: Dictionary of models used by alive configurations with model IDs as keys.
    :ivar alive_model_config: Dictionary of alive configuration IDs with model IDs as keys.
    :ivar disc_models: Dictionary of discarded models with model IDs as keys
    :ivar parameters: Parameters object
    :ivar size: Number of models in the object (equal to number of configurations)
    :ivar hash_models: List of LocalModel objects hashes used maninly to find repeated models.
    """
    def __init__(self, parameters: Parameters, log_folder: str, log_level: int, recovery_folder=None, read_folder=None, global_model=False):
        """
        Creates a ProbabilisticModel object
        :param parameters: Parameters object
        :param log_folder:
        """
        self.models: Dict[int, LocalModel] = {} # int is the index of model, from 0
        self.alive_models: Dict[int, LocalModel] = {}
        self.alive_model_config: Dict[int, list] = {}
        self.disc_models: Dict[int, LocalModel] = {}
        # self.weights = {}
        self.parameters = parameters
        self.size = 0

        # models hash
        self.hash_models = []

        # global model
        self.global_model = global_model
        self.global_model_id = None

        # forbidden_expressions here tracks the configurations that have been marked as forbidden
        # during the experiment. parameters.forbidden_expressions on the other hand only tracks those
        # expressions that were defined with the parameters
        self.forbidden_expressions = ForbiddenExpressions([])

        # create log files
        self.log_level = log_level
        self.model_log = log_folder + "/models.log"
        if recovery_folder is None and read_folder is None:
            file = open(self.model_log, "w")
            file.close()
        self.disc_log = None
        if log_level >= 5:
            self.disc_log = log_folder + "/model_disc.log"
            if recovery_folder is None and read_folder is None:
                file = open(self.disc_log, "w")
                file.close()

    @staticmethod
    def from_log(configurations: Configurations, parameters: Parameters, log_folder: str, log_level: int, recovery_folder=None, read_folder=None, global_model=False, onlytest=False):
        if onlytest: return
        model = ProbabilisticModel(parameters=parameters, log_folder=log_folder,
                                   log_level=log_level, recovery_folder=recovery_folder, read_folder=read_folder,
                                   global_model=global_model,)
        rec_folder = recovery_folder if recovery_folder else read_folder
        model.add_models_from_log(rec_folder, parameters)
        if not read_folder:
            print("# Recovering sampling models from log files")
            print("#  ", model.size, " models recovered")
        return model

    def add_models_from_log(self, recovery_folder, parameters: Parameters):
        model_log_file = recovery_folder + "/models.log"
        if os.path.isfile(model_log_file):
            with open(model_log_file, "r") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                data = json.loads(line)
                if i+1 < len(lines):
                    for mid, model in data.items():
                        new_model = LocalModel.from_log(recovery_dict=model,
                                                        model_id=int(mid),
                                                        parameters=parameters)
                        if not self.model_exist(new_model):
                            self.hash_models.append(new_model.get_hash())
                            self.models[int(mid)] = copy.deepcopy(new_model)
                    self.size = len(self.models)
                else:
                    self.alive_model_config = copy.deepcopy(data)
            self.alive_models = copy.deepcopy(self.models)
        else:
            # raise ModelError("Attempt to recover model from non-existent log file: %s " % (model_log_file))
            print(f"# Note: attempt to recover model from non-existent log file: "
                  f"{model_log_file.split('/')[-1]}")

        disc_log_file = recovery_folder + "/model_disc.log"
        if os.path.isfile(disc_log_file):
            with open(disc_log_file, "r") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                data = json.loads(line)
                for mid, model in data.items():
                        new_model = LocalModel.from_log(recovery_dict=model,
                                                        model_id=int(mid),
                                                        parameters=parameters)
                        if not self.model_exist(new_model):
                            self.hash_models.append(new_model.get_hash())
                            self.models[int(mid)] = copy.deepcopy(new_model)
                self.disc_models[int(mid)] = self.models[int(mid)]
            self.size = len(self.models)

        if self.global_model: self.global_model_id = self.size - 1

    def model_exist(self, model: LocalModel):
        """
        check if a model is already added in the set
        """
        if model.get_hash() in self.hash_models:
            return True
        return False
    
    def _add_model(self, model: LocalModel):
        if self.model_exist(model):
            if self.global_model: self.global_model_id = self.hash_models.index(model.get_hash())
            return False
        
        self.models[self.size] = model
        # self.weights[self.size] = 1

        self.hash_models.append(model.get_hash())

        self.size = len(self.models)

        if self.global_model: self.global_model_id = self.size - 1

        return True

    def add_models(self, mid: int):
        """
        Create and add model
        :param mid: the index of model
        """
        if mid == 0:
            new_model = LocalModel(mid, self.parameters)
            if self._add_model(new_model):
                self.models[0] = new_model
                self.alive_models[0] = new_model
                self.print_to_log()

    def add_forbidden_configuration(self, configuration: ConfigurationEntry):
        """
        Add a configuration to the list of forbidden configurations.

        This does not affect the forbidden_expressions of the Parameters class.

        :param configuration: The configuration to be forbidden
        """
        expression_list = []
        for i, value in configuration.param_values.items():
            if value is not None:
                expression_list.append(self.parameters.get_parameter_forbidden(i, value))
            else:
                expression_list.append(str(i) + ' is None')
        expression = " and ".join(expression_list)
        self.forbidden_expressions.add_forbidden_expression(expression)

    def add_forbidden_configurations(self, configurations: List[ConfigurationEntry]):
        """
        Add a list of configurations to the list of forbidden configurations.

        This does not affect the forbidden_expressions of the Parameters class.

        :param configuration: The configuration to be forbidden
        """
        expression_list = []
        for config in configurations:
            for i, value in config.param_values.items():
                if value is not None:
                    expression_list.append(self.parameters.get_parameter_forbidden(i, value))
                else:
                    expression_list.append(str(i) + ' is None')
            expression = " and ".join(expression_list)
        self.forbidden_expressions.add_forbidden_expression(expression)

    def is_forbidden(self, param_values: dict):
        """
        Check if a set of parameter values are forbidden.

        This method only checks the forbidden_expressions that were generated as a result of configurations
        that were forbidden by the target-runner. For parameter_values that were forbidden by the user, check
        Parameters.is_forbidden().

        :param param_values: Dictionary of values (keys are parameter names)
        :return: True if the param_values are forbidden, False otherwise
        """
        return self.forbidden_expressions.is_forbidden(param_values)

    def _create_sampled_configuration(self, parent_config) -> ConfigurationEntry:
        """
        Samples a configuration from the model of the configuration parent_id.

        This configuration might contain forbidden parameter combinations.

        :param parent_id: The ID of the parent configuration
        :return: A configuration, sampled around the configuration described by parent_ID.
        Might contain forbidden parameter combinations.
        """
        new_values = {}
        for name in self.parameters.sorted_parameters:
            if self.parameters.get_parameter(name).is_active(new_values):
                # get domain in case the parameter has dependent domains
                domain = self.parameters.get_parameter(name).get_domain(new_values)
                mean = parent_config.get_parameter_value(name)
                parent_model = parent_config.get_model_id() if parent_config.id > 0 else 0
                if self.global_model: parent_model = self.global_model_id
                new_values[name] = self.models[parent_model].sample_parameter(name, mean, domain)
                if not self.parameters.get_parameter(name).check_value(new_values[name]):
                    raise ModelError(
                        "Error sampling parameter " + name + " model delivered value " + str(new_values[name]))
            else:
                new_values[name] = None
        cmd_line = self.parameters.get_cmdline(new_values)
        # set configuration id -1 in the meantime
        config = ConfigurationEntry(-1, parent_config.get_id(), parent_model, new_values, cmd_line, self.parameters)
        return config

    def sample_configuration(self, parent_config):
        """
        Samples a configuration from a parent model.
        The configuration will have no ID assigned.

        :param parent_config: Configuration of the parent

        :return: Configuration object (sampled from parent ID model)
        """
        config = self._create_sampled_configuration(parent_config)
        tries = 100
        while (tries >= 0
               and self.parameters.is_forbidden(config.param_values)
               or self.is_forbidden(config.param_values)):
            if tries != 0:
                tries -= 1
                config = self._create_sampled_configuration(parent_config)
            else:
                raise CraceExecutionError("ERROR: crace tried 100 times to sample from the model a configuration not forbidden without success, perhaps your constraints are too strict?")
        # report a newly sampled configuration
        self.models[config.get_model_id()].increment_sampled()
        return config

    def sample_random_parameters(self, size: int) -> list:
        """
        Create a list of length size, containing randomly sampled configurations.

        :param size: How many randomly sampled configurations should be created.
        :return: A list of randomly sampled configurations
        """
        configurations = []
        for _ in range(size):
            # FIXME: should we add a maximum number of attempts?
            # sample parameter values
            param_values, cmd_line = self.parameters.sample_uniform()
            while self.parameters.is_forbidden(param_values) or self.is_forbidden(param_values):
                param_values, cmd_line = self.parameters.sample_uniform()

            # generate the new configuration
            config = ConfigurationEntry(-1, 0, 0, param_values, cmd_line, self.parameters)
            configurations.append(config)
        return configurations

    def sample_from_random_parents(self, size, configurations: Configurations):
        """
        Sample size configurations from randomly selected parents.
        New configurations do no hay an ID assigned.

        :param size: Number of configurations to be samples
        :param configurations: Configurations object used to verify if
        the generated configurations are not repeated in it.
        :return: List of new Configuration objects
        """
        new_configs = []
        for i in range(size):
            #FIXME: add selection by weight
            #FIXME: add repair configurations
            #FIXME: add forbidden parameters
            configs = list(configurations.all_configurations.values())
            parent_config = random.choice(configs)
            generated = False
            while not generated:
                config = self.sample_configuration(parent_config)
                if not configurations.configuration_exist(config):
                    generated = True
            new_configs.append(config)
        return new_configs

    def sample_by_weight(self, size, configurations: Configurations):
        #FIXME: DO THIS
        pass

    def update_weights(self):
        #FIXME: DO THIS
        pass

    def update_old(self, nb_configs, nb_iterations, iteration):
        for config_id, model in self.models.items():
            model.update(nb_configs, nb_iterations, iteration)

    def update(self, configs, alive_configs, nb_new_configs, nb_model_updated, total_model_update, by_sampling=False):
        """
        update model(s)

        :param configs: a list of configurations whose models should be updated
        :param alive_configs: current alive configurations that are to update alive models
        :param nb_new_configs: number of newly sampled configurations bewteen two updates
        :param nb_model_updated: counter for model updates in current loop (separated by restart)
        :param total_model_update: counter for total model updates

        """
        # TODO: the argument of update is currently not used, it could be set as the
        #  number of instances, maybe this can be a good idea
        if not isinstance(configs, list): configs = [configs]
        if not self.global_model:
            for config in configs:
                new_model = copy.deepcopy(self.models[config.model_id])
                new_model.update(config=config, nb_new_configs=nb_new_configs,
                                nb_model_updated=nb_model_updated,
                                total_model_update=total_model_update,
                                by_sampling=by_sampling)
                if self._add_model(new_model):
                    config.set_model_id(self.size-1)
                    logger_1 = logging.getLogger("race_log")
                    logger_1.info(f"# Model {self.size} is updated.")
        else:
            new_model = copy.deepcopy(self.models[self.global_model_id])
            config = configs[0]
            new_model.update(config=config, nb_new_configs=nb_new_configs,
                            nb_model_updated=nb_model_updated,
                            total_model_update=total_model_update*2,
                            by_sampling=by_sampling)
            if self._add_model(new_model):
                config.set_model_id(self.size-1)
                logger_1 = logging.getLogger("race_log")
                logger_1.info(f"# Model {self.size} is updated.")

        # update alive models and print to log file
        self.alive_models.clear()
        self.disc_models.clear()
        old_model = copy.deepcopy(self.alive_model_config)
        self.alive_model_config.clear()
        alive_ids = sorted([x.model_id for x in alive_configs])
        for id in alive_ids:
            if id not in self.alive_models.keys():
                self.alive_models[id] = self.models[id]
        for config in alive_configs:
            if config.model_id not in self.alive_model_config.keys():
                self.alive_model_config[config.model_id] = []
            self.alive_model_config[config.model_id].append(config.id)
        keys_to_delete = [k for k in old_model.keys() if k in self.alive_model_config.keys()]
        for k in keys_to_delete: del old_model[k]
        for id in old_model:
            if id not in self.disc_models.keys():
                self.disc_models[int(id)] = self.models[int(id)]
        self.print_to_log()

    def soft_restart(self, configs: List[ConfigurationEntry], elites: List[int], model_id: int=0, factor: float=None):
        """
        Restart the model of alive configurations

        param: configs: a list of ConfigurationEntry objects to be updated
        param: elites: a list of elitist configurations, print to the log file (model update)
        param: factor: the times of updated model for current configuration (nb_model_updated)
                       used to control the level of restart (default 0: restart to the initial)
        """
        logger_1 = logging.getLogger("race_log")
        logger_1.info(f"# Model of configs soft restart: {[x.get_id() for x in configs]}")

        # FIXME: different method to restart
        #   1. ✅ restart to specific model based on model id
        #   2. restart each parameter model
        for config in configs:
            config.set_model_id(model_id)

        if self.global_model: self.global_model_id = 0

        if model_id == 0 and model_id not in self.models.keys():
            self.add_models(0)

        if model_id not in self.alive_models.keys():
            self.alive_models[model_id] = self.models[model_id]

        self.print_to_log()

    def print(self):
        for model_id, model in self.models.items():
            print(f"model {model_id}:")
            model.print()

    def print_to_log(self):
        """
        Print current models to log file
        :param log_file: File path for the log
        """
        # print to models.log
        log_file = self.model_log
        models = self.alive_models if self.log_level < 3 else self.models

        if os.path.isfile(log_file):
            with open(log_file, 'w') as f:
                # Write the rest of the models
                for key, model in models.items():
                    json.dump({key: model.as_dict()}, f)
                    f.write('\n')
                json.dump(self.alive_model_config, f)

        # print to model_disc.log
        if self.disc_log and os.path.isfile(self.disc_log):
            with open(self.disc_log, 'a') as f:
                # Write the rest of the models
                for key, model in self.disc_models.items():
                    json.dump({key: model.as_dict()}, f)
                    f.write('\n')


    def add_disc_log(self, log_file, config_id):
        """
        Print a model as a line to log file
        :param log_file: File path for the log
        :param config_id: Configuration ID
        """
        model = {}
        assert config_id in self.models.keys(), "Can't find parameter models for {} configuration".format(config_id)
        if os.path.isfile(log_file):
            model[str(config_id)] = self.models[config_id].as_dict()
            json.dump(model, open(log_file, 'a'))
            with open(log_file, "a") as f:
                f.write("\n")


class ModelInfo:
    def __init__(self, type: str, n_all_updating: int, idx_updating: int, last_config_id: int, used_budget, best_so_far: int, best_mean_so_far, elites: list, alive_model_ids: list, end_time: float):
        self.type = type
        self.n_all_updating = n_all_updating
        self.idx_updating = idx_updating
        self.last_config_id = last_config_id
        self.used_budget = used_budget
        self.best_so_far = best_so_far
        self.best_mean_so_far = best_mean_so_far
        self.elites = elites
        self.alive_model_ids = alive_model_ids
        self.end_time = end_time

    def as_dict(self):
        """
        Gets a dictionary of class variables
        """
        return {"type": self.type,
                "n_all_updating": self.n_all_updating,
                "idx_updating": self.idx_updating,
                "last_config_id": self.last_config_id,
                "used_budget": self.used_budget,
                "best_so_far": self.best_so_far,
                "best_mean_so_far": self.best_mean_so_far,
                "elites": self.elites,
                "alive_model_ids": self.alive_model_ids,
                "end_time": self.end_time}

    @staticmethod
    def load_from_log(file):
        slice = pd.DataFrame()
        if os.path.exists(file):
            with open(file, newline='') as f:
                try:
                    data = [json.loads(line) for line in f]
                except:
                    import re
                    from datetime import datetime
                    data= {
                        'total_nb_model_update': [],
                        'nb_model_update': [],
                        'last_config_id': [],
                        'end_time': [],
                    }
                    f.seek(0)
                    for line in f:
                        time = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line).group(1)
                        total, current = map(int, re.search(r'Model update\s+(\d+)\s*/\s*(\d+)', line).groups())
                        config_id = re.search(r'current config id\s+(\d+)', line).group(1)
                        if time:
                            end_time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S,%f")
                            data['end_time'].append(end_time.timestamp())
                        if total:
                            data["total_nb_model_update"].append(total)
                        if current:
                            data['nb_model_update'].append(current)
                        if config_id:
                            data['last_config_id'].append(config_id)
                slice = pd.DataFrame(data)
        return slice
