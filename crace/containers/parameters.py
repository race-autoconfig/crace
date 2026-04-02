import os
import json
import sys
import types
import random
import numpy as np

from abc import ABC, abstractmethod
from ast import literal_eval, parse

from crace.utils import Reader
from crace.containers.forbidden_expressions import ForbiddenExpressions
from crace.errors import CraceError, ParameterDefinitionError, ParameterValueError


""" 
Variable that defines parameter types
  "i": integer 
  "r": real
  "c": categorical
  "o": ordered
  "i,log": logarithmic integer  
  "r,log": logarithmic real
"""
_parameter_types = ["i", "r", "c", "o", "i,log", "r,log"]


class ParameterEntry(ABC):
    """
    Class Parameter implements basic functionality for handling parameters
    :ivar name: Parameter name
    :ivar switch: Parameter switch
    :ivar type: Parameter type
    :ivar domain: List of parameter domain
    :ivar condition: Parameter condition (logical expression)
    :ivar priority: Parameter priority
    :ivar depends: List of parameter names from which a parameter is conditional
    """
    def __init__(self, name, switch, param_type, domain, condition=None):
        """
        Contains the information needed to create the parameter of a
        configuration space

        :param name:        Name of the parameter
        :param switch:      Switch used in the command line to the program
        :param param_type:  Data type of the argument, is a value that
                            belongs to the accepted types. "c", "i", "r", "o"
        :param domain:      Domain of the parameter. gives a range of
                            values for the argument
        :param condition:   Logical expresion to determine if the
                            parameter belongs or not to a configuration. if
                            no condition is given set the default value as True
        """
        if condition is None:
            self.condition = True
        self.name = name
        self.switch = switch
        self.type = param_type
        self.domain = domain
        self.condition = condition
        self.priority = 0
        self.depends = 0

    @abstractmethod
    def get_datastring(self):
        """
        Returns the attributes of the parameters as a string of the form
        "name:		{}
        switch:		{}
        type:		{} {}
        domain:		{}
        conditions:	{}
        priority:	{}
        depends:    {}
        isFixed:	{}
        """

    def get_cmdline(self, value):
        """
        Gets the command line string of a parameter value
        :param value: Value to be assigned to the parameter
        :return: string with the format [flagvalue] (no space between them)
        """
        return self.switch + str(value)

    @staticmethod
    def get_parameter(full_line, n_line, digits, debug_level):
        """
        Reads a string with a parameter data and initializes a Parameter
        object with the attributes from the string

        :param full_line:   String without line skip
        :param n_line:      Number of the current line of the valid text
        :param digits:      Number of digits for the precision of real-valued parameters
        :param debug_level: Debug level (0,1,2,3,4) to print information
        :return:    Initialized Parameter object
        """

        # get parameter name
        name, line = Reader.get_matched_line(full_line, ["^[a-zA-Z0-9_]+$"])
        if not name:
            ParameterEntry.raise_parameter_error(n_line, line, "parameter name must be alphanumeric", ParameterDefinitionError)

        # get switch
        switch, line = Reader.get_matched_line(line, ["\"[^\"]*\""], "( |\\\".*?\\\"|'.*?')", True)
        if not switch:
            ParameterEntry.raise_parameter_error(n_line, line, "parameter switch must be a double-quoted string", ParameterDefinitionError)

        # get type
        transform = None
        param_type, line = Reader.get_matched_line(line, _parameter_types)
        if not param_type:
            ParameterEntry.raise_parameter_error(n_line, line,
                                        "parameter type must be a single character {'c','i',"
                                        "'r','o'}, with 'i', 'r' optionally followed by ',log' (no "
                                        "spaces in between) to sample using a logarithmic scale",
                                       ParameterDefinitionError)

        elif param_type == "i,log" or param_type == "r,log":
            param_type, transform = param_type.split(",")

        # get domain
        domain, line = Reader.get_matched_line(line=line, patterns=[r"\([^)]+\)"],
                                               separator=r"\[[^\]]*\]|\([^\)]*\)|\"[^\"]*\"|\S+", delimited=True)
        if not domain:
            ParameterEntry.raise_parameter_error(n_line, line, "Allowed values must be a list within parenthesis", ParameterDefinitionError)
        try:
            if param_type == "i":
                domain = IntegerParameter.parse_domain(domain)
            elif param_type == "r":
                domain = RealParameter.parse_domain(domain, digits)
            elif param_type == "c":
                domain = CategoricalParameter.parse_domain(domain)
            elif param_type == "o":
                domain = OrderedParameter.parse_domain(domain)
            else:
                raise ParameterDefinitionError("Parameter type error")
        except ParameterDefinitionError as err:
            ParameterEntry.raise_parameter_error(n_line, line, err, ParameterDefinitionError)

        # get conditions
        if len(line) > 1 and not line.strip().startswith("#"):
            value, line = Reader.get_matched_line(line, ["|"])
            if value is None:
                ParameterEntry.raise_parameter_error(n_line, name, "expected '|' before condition", ParameterDefinitionError)
            condition = ParameterEntry.parse_condition(line, n_line, full_line)
        else:
            condition = ParameterEntry.parse_condition("True", n_line, full_line)

        # create parameter object
        param = None
        if param_type == "i":
            param = IntegerParameter(name, switch, param_type, transform, domain, condition)
        elif param_type == "r":
            param = RealParameter(name, switch, param_type, transform, domain, condition, digits)
        elif param_type == "c":
            param = CategoricalParameter(name, switch, param_type, domain, condition)
        elif param_type == "o":
            param = OrderedParameter(name, switch, param_type, domain, condition)
        else:
            ParameterEntry.raise_parameter_error(n_line, name, "error when reading parameter line", ParameterDefinitionError)
        return param

    @staticmethod
    def get_from_log(obj):
        """
        Get parameter from object

        :param obj: List of parameter definition
        :return: a dict of parameter class        
        """
        param = None
        param_type = obj["type"]
        if param_type == "i":
            param = IntegerParameter.from_dict(obj)
        elif param_type == "r":
            param = RealParameter.from_dict(obj)
        elif param_type == "c":
            param = CategoricalParameter.from_dict(obj)
        elif param_type == "o":
            param = OrderedParameter.from_dict(obj)
        else:
            ParameterEntry.raise_parameter_error("Error when loading parameter from log")
        return param

    @staticmethod
    def raise_parameter_error(nb, line, msg, error=Exception):
        """
        Raise a given ParameterException
        """
        raise error("On line {}: '{}'\n{}".format(nb+1, line, msg))

    def is_active(self, partial_config):
        """
        Check that the higher hierarchy parameters satisfy their conditions
        needed to assign a value to the current parameter

        :param partial_config: Previously created parameters
        :return: True if the condition is satisfied, False otherwise
        """
        assert self.condition is not None, "Empty condition"
        
        variables = {}
        for var in self.condition.co_names:
            value = partial_config.get(var)
            if value is None:  # FIXME: Same issue as for ForbiddenExpressions
                return False  # if at least one variable is None, the condition is not active
            if isinstance(value, str):
                variables[var] = value
            else:
                variables[var] = str(value)
        return self.eval(self, self.condition, variables)

    @staticmethod
    def parse_condition(condition, n_line, line):
        """
        check the syntax of a condition, compile and return it.
        works with python and R syntax for simple tasks.
        valid operators ! == > < <= >= %in% in & and | or also R vectors
        """
        if condition == '':
            ParameterEntry.raise_parameter_error(n_line, line, "expected condition after '|'")
        try:
            condition = condition.split("#")[0]
            condition = Reader.replace_logical_operator(condition)
            return compile(parse(condition, mode='eval'), condition, 'eval')
        except SyntaxError:
            ParameterEntry.raise_parameter_error(n_line, line, "invalid condition after '|'")

    def check_set_condition(self, param_names):
        """
        Check, test and set the dependency of the parameter.
        This function check that all variables in the condition are parameter
        names. The variable self.depends is set with the names of the variables
        in the condition.

        :param param_names: Dictionary of parameter values (keys are parameter names)
        """
        assert self.condition is not None, "Empty condition"

        for var in self.condition.co_names:
            if var not in param_names:
                raise ParameterDefinitionError("Condition " + str(self.condition) + " includes parameter " + var + " that is not defined")
        self.depends = self.condition.co_names

    def get_type(self):
        """
        Get parameter type.

        Types are defined in variable _parameter_types
        """
        return self.type

    def get_domain(self, partial_config=None):
        """
        Get active parameter domain

        :param partial_config: Dictionary of parameter values
            already assigned in the configuration (partial or complete)

        :return: domain
        """
        if partial_config is None:
            return self.domain
        else:
            # dependent domains
            # FIXME: implement dependent domains!
            return self.domain

    def is_dependent(self):
        # FIXME: implement dependent domains!
        return False #len(self.depends) > 0
    
    def eval(self, o, a=None, b=None):
        try:
            if a and b: return eval(a,b)
            if a and not b: return eval(a)
        except Exception as e:
            print(f"\nERROR: There was an error while loading parameter {o.name}: {a.co_filename if isinstance(a, types.CodeType) else a}:")
            print(e)
            sys.exit(1)


class IntegerParameter(ParameterEntry):
    """
    class that implements integer parameters
    :param transform: Boolean that indicates if the parameter value is log transformed
    :param is_fixed: Boolean that indicates that a parameters has a fixed value
    """
    def __init__(self, name, switch, param_type, transform, domain, condition=None):
        """
        Builds an integer parameter object
        :param name: Parameter name (string)
        :param switch: Parameter switch (string)
        :param param_type: Parameter type from _parameter_types variable
        :param transform: Boolean that indicates if the parameter value is log transformed
        :param domain: Domain list of a parameter
        :param condition: Activation condition of a parameter
        """
        super().__init__(name, switch, param_type, domain, condition)
        self.transform = transform
        self.is_fixed = (self.domain[0] == self.domain[1])

    @staticmethod
    def from_dict(obj):
        """
        Creates an Integer parameter from a dictionary
        :return: IntegerParameter object
        """
        return IntegerParameter(obj["name"], obj["switch"], obj["type"], obj["transform"], obj["domain"],
                                compile(parse(obj["condition"], mode='eval'), obj["condition"], 'eval'))

    def get_datastring(self):
        """
        Returns the attributes of the parameters as a string of the form
        "name:		{}
        switch:		{}
        type:		{} {}
        domain:		{}
        conditions:	{}
        priority:	{}
        depends:    {}
        isFixed:	{}
        """
        return (
            "#   name:\t\t{}\n#   switch:\t\t{}\n#   type:\t\t{} {}\n#   domain:\t\t{}\n"
            "#   conditions:\t{}\n#   priority:\t{}\n#   depends:\t{}\n#   isFixed:\t{}\n".format(
                self.name, self.switch, self.type, self.transform,
                self.domain, self.condition, self.priority, self.depends,
                self.is_fixed))

    def as_dict(self):
        """
        Generate a dictionary with the parameter object data
        :returns: Dictionary of parameter class variables
            name
            switch
            type
            transform
            domain
            condition
            priority
            depends
            is_fixed
        """
        return {"name": self.name, "switch": self.switch, "type": self.type, "transform": self.transform,
                "domain": self.domain, "condition": self.condition.co_filename, "priority": self.priority,
                "depends": self.depends, "is_fixed": self.is_fixed}

    def check_value(self, value):
        """
        Check if a value has the correct type and within the parameter domain

        :param value:   Possible value for the parameter
        :return:     True if the value is in the domain, False in other case
        """
        if not isinstance(value, int):
            return False
        return self.domain[0] <= value and value <= self.domain[1]

    def parse_value(self, value):
        """
        Parse and check a parameter value from a string or the right type for the parameter.
        The parameter value is check to be within a valid domain.

        :param value: String of parameter value
        :return: Parsed parameter value
        """
        if value == "NA":
            return None
        if isinstance(value, str):
            value = eval(value)
        if not self.check_value(value):
            raise ParameterValueError("Value " + str(value) + " for parameter " + self.name +
                                      " is not valid. Parameter " + self.name + " is integer, values must in " +
                                      str(self.domain))
        return value

    def random_value(self, partial_config=None):
        """
        Generates a random value for the parameter. If parameter is not active value returned is None.

        :param partial_config: Partial parameter value assignment (dictionary with parameter name keys)
        :return:    The value of the parameter obtained by a random function.
        """        
        if partial_config is None or self.is_active(partial_config):
            if self.is_fixed:
                return self.domain[0]
            value = random.uniform(self.domain[0], self.domain[1])
            return int(value)
        else:
            return None

    @staticmethod
    def parse_domain(domain_str):
        """
        Parse a parameter domain from a string

        :param domain_str: Domain string lower_bound, upper_bound
        :return: Domain list [lower_bound, upper_bound]
        """
        # parse domain
        try:
            domain = literal_eval(domain_str)
            domain = list(domain)
            domain = [round(x, 0) for x in domain]
        except SyntaxError:
            raise ParameterDefinitionError("incorrect data type ({}) in domain ".format(domain_str))
        except Exception as error:
            raise error

        if len(domain) != 2:
            raise ParameterDefinitionError("incorrect numeric range ({})".format(domain_str))

        if domain[0] > domain[1]:
            raise ParameterDefinitionError("lower bound must be smaller than upper bound in numeric range {}".format(domain_str))

        if not isinstance(domain[0], int) or not isinstance(domain[1], int):
            raise ParameterDefinitionError("for parameter type 'i' values must be  integers {} ".format(domain_str))

        if domain[0] == domain[1]:
            raise ParameterDefinitionError("lower and upper bounds are the same in numeric range {}".format(domain_str))

        return domain

    def get_transform(self):
        """
        Get flag value that indicates if the parameter is log transformed
        """
        return self.transform

    def get_forbidden_condition(self, value):
        s = self.name + " == " + str(value)
        return s

class RealParameter(ParameterEntry):
    """
    class that implements real parameters
    :ivar transform: Boolean that indicates if the parameter value is log transformed
    :ivar digits: Number of digits for precision
    """
    def __init__(self, name, switch, param_type, transform, domain, condition=None, digits=4):
        """
        Builds a real parameter object, set default digits for the real parameters as 4
        :param name: Parameter name (string)
        :param switch: Parameter switch (string)
        :param param_type: Parameter type from _parameter_types variable
        :param transform: Boolean that indicates if the parameter value is transformed
        :param domain: Domain list of a parameter
        :param condition: Activation condition of a parameter
        :param digits: Number of digits for precision
        """
        super().__init__(name, switch, param_type, domain, condition)
        self.transform = transform
        self.digits = digits
        self.is_fixed = (self.domain[0] == self.domain[1])

    @staticmethod
    def from_dict(obj):
        """
        Creates an Real parameter from a dictionary

        :return: RealParameter object
        """
        return RealParameter(obj["name"], obj["switch"], obj["type"], obj["transform"], obj["domain"],
                             compile(parse(obj["condition"], mode='eval'), obj["condition"], 'eval'),
                             obj["digits"])

    def as_dict(self):
        """
        Generate a dictionary with the parameter object data

        :returns: Dictionary of parameter class variables
            name
            switch
            type
            transform
            domain
            condition
            priority
            depends
            is_fixed
            digits
        """
        return {"name": self.name, "switch": self.switch, "type": self.type, "transform": self.transform,
                "domain": self.domain, "condition": self.condition.co_filename, "priority": self.priority,
                "depends": self.depends, "is_fixed": self.is_fixed, "digits": self.digits}

    def get_datastring(self):
        """
        Returns the attributes of the parameters as a string of the form
        "name:		{}
        switch:		{}
        type:		{} {}
        domain:		{}
        conditions:	{}
        priority:	{}
        depends:    {}
        isFixed:	{}
        """
        return (
            "#   name:\t\t{}\n#   switch:\t\t{}\n#   type:\t\t{} {}\n#   domain:\t\t{}\n"
            "#   conditions:\t{}\n#   priority:\t{}\n#   depends:\t{}\n#   isFixed:\t{}\n".format(
                self.name, self.switch, self.type, self.transform,
                self.domain, self.condition, self.priority, self.depends,
                self.is_fixed))

    def check_value(self, value):
        """
        Check if a value has the correct type and within the parameter domain

        :param value:   Possible value for the parameter
        :return:     True if the value is in the domain, False in other case
        """
        if not isinstance(value, float) and not isinstance(value, int):
            return False
        value = round(value, self.digits)
        return self.domain[0] <= value and value <= self.domain[1]

    def parse_value(self, value):
        """
        Parse and check a parameter value from a string or the right type for the parameter.
        The parameter value is check to be within a valid domain.

        :param value: String of parameter value
        :return: Parsed parameter value
        """
        if value == "NA":
            return None
        if isinstance(value, str):
            value = eval(value)
        if not self.check_value(value):
            raise ParameterValueError("Value " + str(value) + " for parameter " + self.name +
                                      " is not valid. Parameter " + self.name + " is real, values must in " +
                                      str(self.domain))
        value = round(value, self.digits)
        return value

    def random_value(self, partial_config=None):
        """
        Generates a random value for the parameter. If parameter is not active value returned is None.

        :param partial_config: Partial parameter value assignment (dictionary with parameter name keys)
        :return:    The value of the parameter obtained by a random function.
        """
        if partial_config is None or self.is_active(partial_config):
            if self.is_fixed:
                return self.domain[0]
            value = random.uniform(self.domain[0], self.domain[1])
            return round(value, self.digits)
        else:
            return None

    @staticmethod
    def parse_domain(domain_str, digits):
        """
        Parse a parameter domain from a string

        :param domain_str: Domain string lower_bound, upper_bound
        :param digits: Number of digits of precision for the parameter
        :return: Domain list [lower_bound, upper_bound]
        """
        # parse domain
        try:
            domain = literal_eval(domain_str)
            domain = list(domain)
            domain = [round(x, digits) for x in domain]
        except SyntaxError:
            raise ParameterDefinitionError("incorrect data type ({}) in domain ".format(domain_str))
        except Exception as error:
            raise error

        if len(domain) != 2:
            raise ParameterDefinitionError("incorrect numeric range ({})".format(domain_str))

        if domain[0] > domain[1]:
            raise ParameterDefinitionError(
                "lower bound must be smaller than upper bound in numeric range {}".format(domain_str))

        # if not isinstance(domain[0], float) or not isinstance(domain[1], float):
        #     raise ParameterDefinitionError("for parameter type 'r' values must be real ({}) (with fractional part)".format(domain_str))
        if not isinstance(domain[0], float):
            domain[0] = float(domain[0])
        if not isinstance(domain[1], float):
            domain[1] = domain[1]

        if domain[0] == domain[1]:
            raise ParameterDefinitionError("lower and upper bounds are the same in numeric range {}".format(domain_str))

        return domain

    def get_digits(self):
        """
        Get number of digits of precision
        """
        return self.digits

    def get_transform(self):
        """
        Get flag value that indicates if the parameter is log transformed
        """
        return self.transform

    def get_forbidden_condition(self, value):
        s = self.name + " == " + str(value)
        return s

class CategoricalParameter(ParameterEntry):
    """
    class that implements categorical parameters
    :ivar is_fixed: Boolean that indicates if the parameter has a fixed value
    """

    def __init__(self, name, switch, param_type, domain, condition=None):
        """
        Builds an integer parameter object
        :param name: Parameter name (string)
        :param switch: Parameter switch (string)
        :param param_type: Parameter type from _parameter_types variable
        :param domain: Domain list of a parameter
        :param condition: Activation condition of a parameter
        """
        super().__init__(name, switch, param_type, domain, condition)
        self.is_fixed = len(domain) == 1

    @staticmethod
    def from_dict(obj):
        """
        Creates a Categorical parameter from a dictionary
        :return: CategoricalParameter object
        """
        return CategoricalParameter(obj["name"], obj["switch"], obj["type"], obj["domain"],
                                    compile(parse(obj["condition"], mode='eval'), obj["condition"], 'eval'))

    def as_dict(self):
        """
        Generate a dictionary with the parameter object data
        :returns: Dictionary of parameter class variables
            name
            switch
            type
            domain
            condition
            priority
            depends
            is_fixed
        """
        return {"name": self.name, "switch": self.switch, "type": self.type, "domain": self.domain,
                "condition": self.condition.co_filename, "priority": self.priority, "depends": self.depends,
                "is_fixed": self.is_fixed}

    def get_datastring(self):
        """
        Returns the attributes of the parameters as a string of the form
        "name:		{}
        switch:		{}
        type:		{} {}
        domain:		{}
        conditions:	{}
        priority:	{}
        depends:    {}
        isFixed:	{}
        """
        return (
            "#   name:\t\t{}\n#   switch:\t\t{}\n#   type:\t\t{}\n#   domain:\t\t{}\n"
            "#   conditions:\t{}\n#   priority:\t{}\n#   depends:\t{}\n#   isFixed:\t{}\n".format(
                self.name, self.switch, self.type,
                self.domain, self.condition, self.priority, self.depends,
                self.is_fixed))

    def check_value(self, value):
        """
        Check if a value has the correct type and within the parameter domain

        :param value:   Possible value for the parameter
        :return:     True if the value is in the domain, False in other case
        """
        if isinstance(value, str) and (value in self.domain):
            return True
        return False

    def parse_value(self, value):
        """
        Parse and check a parameter value from a string or the right type for the parameter.
        The parameter value is check to be within a valid domain.

        :param value: String of parameter value
        :return: Parsed parameter value
        """
        if value == "NA":
            return None
        if not self.check_value(value):
            raise ParameterValueError("Value " + str(value) + " for parameter " + self.name +
                                      " is not valid. Parameter " + self.name + " is categorical, values must in " +
                                      str(self.domain))
        return value

    def random_value(self, partial_config=None):
        """
        Generates a random value for the parameter. If parameter is not active value returned is None.

        :param partial_config: Partial parameter value assignment (dictionary with parameter name keys)
        :return:    The value of the parameter obtained by a random function.
        """
        if partial_config is None or self.is_active(partial_config):
            if self.is_fixed:
                return self.domain[0]
            value = random.choice(self.domain)
            return value
        else:
            return None

    @staticmethod
    def parse_domain(domain_str):
        """
        Parse a parameter domain from a string

        :param domain_str: Domain string v1, v2, v3, v4,...
        :param digits: Number of digits of precision for the parameter
        :return: Domain list [v1, v2, v3, v4,...]
        """
        # parse domain
        domain = [x.strip() for x in domain_str.strip("()").split(",")]
        if len(domain) != len(set(domain)):
            raise ParameterDefinitionError("Repeated values in domain of parameter")
        return domain

    def get_forbidden_condition(self, value):
        s = self.name + " == \"" + value + "\""
        return s

class OrderedParameter(ParameterEntry):
    """
    class that implements ordered parameters
    :ivar is_fixed: Boolean that indicates if the parameter has a fixed value
    """
    def __init__(self, name, switch, param_type, domain, condition=None):
        """
        Builds an integer parameter object
        :param name: Parameter name (string)
        :param switch: Parameter switch (string)
        :param param_type: Parameter type from _parameter_types variable
        :param domain: Domain list of a parameter
        :param condition: Activation condition of a parameter
        """
        super().__init__(name, switch, param_type, domain, condition)
        self.is_fixed = len(self.domain) == 1

    @staticmethod
    def from_dict(obj):
        """
        Creates a Ordered parameter from a dictionary
        :return: OrderedParameter object
        """
        return OrderedParameter(obj["name"], obj["switch"], obj["type"], obj["domain"],
                                compile(parse(obj["condition"], mode='eval'), obj["condition"], 'eval'))

    def as_dict(self):
        """
        Generate a dictionary with the parameter object data
        :returns: Dictionary of parameter class variables
            name
            switch
            type
            domain
            condition
            priority
            depends
            is_fixed
        """
        return {"name": self.name, "switch": self.switch, "type": self.type, "domain": self.domain,
                "condition": self.condition.co_filename, "priority": self.priority, "depends": self.depends,
                "is_fixed": self.is_fixed}

    def get_datastring(self):
        """
        Returns the attributes of the parameters as a string of the form
        "name:		{}
        switch:		{}
        type:		{} {}
        domain:		{}
        conditions:	{}
        priority:	{}
        depends:    {}
        isFixed:	{}
        """
        return (
            "#   name:\t\t{}\n#   switch:\t\t{}\n#   type:\t\t{}\n#   domain:\t\t{}\n"
            "#   conditions:\t{}\n#   priority:\t{}\n#   depends:\t{}\n#   isFixed:\t{}\n".format(
                self.name, self.switch, self.type,
                self.domain, self.condition, self.priority, self.depends,
                self.is_fixed))

    def check_value(self, value):
        """
        Check if a value has the correct type and within the parameter domain

        :param value:   Possible value for the parameter
        :return:     True if the value is in the domain, False in other case
        """
        if isinstance(value, str) and (value in self.domain):
            return True
        return False

    def parse_value(self, value):
        """
        Parse and check a parameter value from a string or the right type for the parameter.
        The parameter value is check to be within a valid domain.

        :param value: String of parameter value
        :return: Parsed parameter value
        """
        if value == "NA":
            return None
        if not self.check_value(value):
            raise ParameterValueError("Value " + str(value) + " for parameter " + self.name +
                                      " is not valid. Parameter " + self.name + " is ordered, values must in " +
                                      str(self.domain))
        return value

    def random_value(self, partial_config=None):
        """
        Generates a random value for the parameter. If parameter is not active value returned is None.

        :param partial_config: Partial parameter value assignment (dictionary with parameter name keys)
        :return:    The value of the parameter obtained by a random function.
        """
        if partial_config is None or self.is_active(partial_config):
            if self.is_fixed:
                return self.domain[0]
            value = random.choice(self.domain)
            return value
        else:
            return None

    @staticmethod
    def parse_domain(domain_str):
        """
        Parse a parameter domain from a string

        :param domain_str: Domain string v1, v2, v3, v4,...
        :return: Domain list [v1, v2, v3, v4,...]
        """
        # parse domain
        domain = [x.strip() for x in domain_str.strip("()").split(",")]
        if len(domain) != len(set(domain)):
            raise ParameterDefinitionError("Repeated values in domain of parameter")
        return domain

    def get_forbidden_condition(self, value):
        s = self.name + " == \"" + value + "\""
        return s

class Parameters(Reader):
    """
    Class Parameters manages all parameters defined in the configuration space

    :ivar all_parameters:    Dictionary of Parameter objects, keys are parameter names
    :ivar sorted_parameters: List of the parameter names in sampling order
    :ivar nb_parameters:     Number of parameters
    :ivar nb_fixed:          Number of fix-valued parameters
    :ivar n_variable:        Number of variable-valued parameters
    """
    def __init__(self, parameters_file=None, text=None, digits=4, debug_level=0,
                 exec_dir=None, forbidden_file: str = None, forbidden_text: str = None,
                 recovery_folder=None, silent=False, load_folder=None, log_base_name: str = None):
        """
        Parameter class that contains the definition of the parameters of an scenario

        :param parameters_file: File path of the parameter file
        :param text:            String which contains the parameter definition (alternative to parameters_file)
        :param digits:          Number of digits considered as precision for real-valued parameters
        :param debug_level:     Level of debug (0,1,2,3,4) for printing information
        :param exec_dir: File path of the file used for parameter log
        :param forbidden_file:  File path of the file where forbidden expressions are provided
        :param forbidden_text:  String which contains the forbidden expressions (alternative to forbidden_file)
        :param recovery_folder: Recovery folder
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

        self.all_parameters = {}
        self.sorted_parameters = []
        # self.hierarchy = []
        self.nb_parameters = 0
        self.nb_fixed = 0
        self.nb_variable = 0
        self.debug_level = debug_level

        if recovery_folder is not None:
            parameters_file = recovery_folder + "/parameters.log"

            if os.path.exists(parameters_file):
                with open(parameters_file, newline='') as f:
                    data = json.load(f)

                    for dparam in data[0]:
                        parameter = ParameterEntry.get_from_log(dparam)
                        self.all_parameters[parameter.name] = parameter
                        self.sorted_parameters.append(parameter.name)
                        self.nb_parameters = self.nb_parameters + 1

                        if parameter.is_fixed:
                            self.nb_fixed = self.nb_fixed + 1
                        else:
                            self.nb_variable = self.nb_variable + 1

                    if debug_level >= 2:
                        print("#  total parameters found: ", self.nb_parameters)
                    if debug_level >= 3:
                        print("#  ", self.sorted_parameters)
                    self.forbidden_expr = ForbiddenExpressions(data[1])

            else:
                raise CraceError("Cannot find parameters log file: " + parameters_file)

        else:
            if parameters_file is None and text is None:
                parameters_file = os.path.join(os.path.abspath('./'), 'parameters.txt')
                if not os.path.exists(parameters_file):
                    print(f"\nERROR: There was an error while loading parameters")
                    sys.tracebacklimit = 0
                    raise CraceError("The default parameters.txt is not exist in current directory.\n"
                                     "Either parameters_file or text must be provided.")

            if text is not None:
                lines = text.split("\n")
            else:
                self.check_readable(parameters_file)
                lines = self.get_readable_lines(parameters_file)

            # parse parameters and create parameter objects
            for i in range(len(lines)):
                try:
                    parameter = ParameterEntry.get_parameter(lines[i], i, digits=digits, debug_level=debug_level)
                except ParameterDefinitionError as e:
                    print("\nERROR: There was an error while loading parameters")
                    print(e)
                    sys.exit(1)

                if parameter.name in self.all_parameters.keys():
                    raise CraceError("Repeated parameter name: ", parameter.name)

                self.all_parameters[parameter.name] = parameter
                self.sorted_parameters.append(parameter.name)
                self.nb_parameters = self.nb_parameters + 1
                if parameter.is_fixed:
                    self.nb_fixed = self.nb_fixed + 1
                else:
                    self.nb_variable = self.nb_variable + 1

            if forbidden_file is not None:
                # FIXME: implement forbidden Text option
                if not silent and debug_level >= 1:
                    print("# Reading forbidden expressions file: " + forbidden_file)
                try:
                    self.forbidden_expr = ForbiddenExpressions(self.read_forbidden(forbidden_file))
                except ParameterDefinitionError as e:
                    print("\nERROR: There was an error while loading forbidden expression(s):")
                    print(e)
                    sys.exit(1)

                if not silent and debug_level >= 2:
                    print("#   found ", len(self.forbidden_expr.forbidden_expressions), " expression(s)")
                if not silent and debug_level >= 3:
                    print("#   ", self.forbidden_expr.forbidden_expressions)
            else:
                self.forbidden_expr = ForbiddenExpressions([])

        if len(self.all_parameters) == 0:
            raise CraceError("No parameter definition found: check that the parameter file is not empty")

        # check conditions
        self._check_conditions()

        # process tree dependencies
        self._tree_level()

        # sort by hierarchy
        self.sorted_parameters.sort(key=lambda p: self.all_parameters[p].priority)

        #log files
        if log_folder and not recovery_folder:
            self.log_parameters = log_folder + "/parameters.log"
            if recovery_folder is None:
                file = open(self.log_parameters, "w")
                file.close()
            self.print_to_log(self.log_parameters)

        self.exec_dir = exec_dir
        self.parameters_file = parameters_file

    def _check_conditions(self):
        """
        Test all parameter conditions in each parameter object
        to be valid. The function does not return a value, just raises an
        exception when an error is found.
        """
        for name, param in self.all_parameters.items():
            param.check_set_condition(self.all_parameters.keys())

    def _tree_level(self):
        """
        Goes through the parameters and set its hierarchy according to the
        conditions that define the object. The value of hierarchy is higher
        if the numerical value is closer to 1. So a parameter with hierarchy
        1 has higher priority that one with hierarchy 3.
        """
        tree = {}
        for name, param in self.all_parameters.items():
            self._tree_level_aux(param, param)
            tree[name] = param.priority
        return tree

    def _tree_level_aux(self, param: ParameterEntry, root: ParameterEntry):
        """
        Recursively calculates and assigns the priority of a parameter and
        those on which it depends according to its conditions. If its
        condition is always true (i.e. it has no conditions to exist) returns 1
        (highest hierarchy)

        :param parameter:   Parameter from which the hierarchy will be
                            calculated
        :param root:        First parameter given to the method. if its
                            called from outside the scope then root should be
                            the same as parameter

        :return:            The priority calculated for the current parameter
        """
        current_level = 0
        if param.condition is True:
            param.priority = 1

        if param.priority != 0:
            return param.priority

        dependencies = param.condition.co_names

        for var in dependencies:
            param_aux = self.all_parameters.get(var)
            if param_aux is None:
                raise CraceError(
                    "A parameter definition is missing! ", "Check definition of parameters.\n",
                    "Parameter '{}' depends on '{}' which is not "
                    "defined.".format(param.name, var))

            elif var == root.name:
                raise CraceError(
                    "A cycle detected in subordinate parameters! Check "
                    "definition of conditions.\n"
                    "One parameter of this cycle is '{}'".format(var)
                )

            level = self._tree_level_aux(param_aux, root)

            if level > current_level:
                current_level = level
        param.priority = current_level + 1
        return param.priority

    def show_parameters(self):
        """
        Show the information of every parameter loaded
        """
        print("# Current parameters: ")
        for p in self.sorted_parameters:
            print(self.all_parameters[p].get_datastring())

    def get_cmdline(self, param_values):
        """
        Generates the command line string of a set of parameter values

        :param param_values: Dictionary of the parameter values (parameter names as keys)
        :return: Command line string
        """
        cmd_line = ""
        for name, param in self.all_parameters.items():
            if param_values[name] is not None:
                cmd_line = cmd_line + " " + param.get_cmdline(param_values[name])
        return cmd_line

    def parse_values(self, param_values, init=None):
        """
        Function that parses and checks the parameter values (of a configuration) in dictionary param_values
        to be then assigned to a configuration.

        :param param_values: Dictionary of values with parameter names as keys and values in any format
        :return: dictionary param_values with parsed values and command line cmd_line
        """
        not_provided = np.array([x not in param_values.keys() for x in self.sorted_parameters])
        sorted_parameters = np.array(self.sorted_parameters)
        if any(not_provided):
            raise ParameterValueError("Parameters " + str(sorted_parameters[not_provided]) + " are not provided in configuration")

        unknown = [x not in self.sorted_parameters for x in param_values.keys()]
        if any(unknown):
            raise ParameterValueError("Unknown parameter names " + str(sorted_parameters[unknown]) + " provided in configuration")

        illegal = set()
        for name in self.sorted_parameters:
            value = param_values[name]
            try:
                param_values[name] = self.all_parameters[name].parse_value(value)
            except Exception as e:
                raise ParameterValueError(f"Value {value} of parameter {name} (domain: {self.all_parameters[name].domain}) is incorrect")

            if not self.all_parameters[name].is_active(param_values) and param_values[name] is not None:
                if self.debug_level >= 1:
                    print("# Non active parameter " + name + " has value " + str(param_values[name]) + ", changing to None" )
                if init: illegal.add(name)
                param_values[name] = None

        if init and illegal: print(f"# Non active parameter(s) {', '.join(map(str, illegal))} in initial configuration {init} are changing to None")
        cmd_line = self.get_cmdline(param_values)

        return param_values, cmd_line

    def sample_uniform(self):
        """
        Generates sampled uniform parameter values to be assigned in the configuration

        :return param_values: Dictionary of sampled parameter values (parameter names as keys) and
                              command line string
        """
        param_values = {}
        cmd_line = ""
        # for name, param in self.all_parameters.items():
        for name in self.sorted_parameters:
            param = self.all_parameters[name]
            param_values[name] = param.random_value(param_values)
            if param_values[name] is not None:
                cmd_line = cmd_line + " " + param.get_cmdline(param_values[name])
        return param_values, cmd_line

    def get_names(self):
        """
        Gets the parameter names, these are returned in sampling order

        :return: List of parameter name strings
        """
        return self.sorted_parameters

    def get_parameter(self, name: str):
        """
        Gets a parameter object by its name

        :param name: Parameter name
        :return: Object Parameter
        """
        if name not in self.sorted_parameters:
            raise CraceError("Parameter " + name + " not found.")
        return self.all_parameters[name]

    def get_parameters(self):
        """
        Gets all parameters objects

        :return: Dictionary Parameter objects (key is the parameter name)
        """
        return self.all_parameters

    def get_parameter_forbidden(self, param_name: str, param_value):
        """
        Gets a parameter type

        :param param_name: Parameter name
        :param param_value: parameter value
        :return: Parameter type
        """
        if param_name not in self.sorted_parameters:
            raise CraceError("Parameter " + param_name + " not found.")
        return self.all_parameters[param_name].get_forbidden_condition(param_value)

    def print_to_log(self, file_path):
        """
        Print the Parameters object to a file so it can be recovered later

        :param file_path: File path of the log
        :param forbidden_file_path: File path of the forbidden expressions log
        """
        # create a data frame of the configurations
        params = [self.all_parameters[x].as_dict() for x in self.sorted_parameters]
        json.dump([params, self.forbidden_expr.forbidden_expressions], open(file_path, 'w'))

    def print(self):
        """
        Print the parameters information
        """
        #TODO: improve this output
        params = [self.all_parameters[x].as_dict() for x in self.sorted_parameters]
        print(params)

    def read_forbidden(self, forbidden_file: str):
        """
        Function that reads and returns forbidden expressions from a file

        :param forbidden_file: File path to forbidden expressions
        :return: list of strings which are the expressions
        """
        # read file lines
        if self.check_readable(forbidden_file):
            expressions = self.get_readable_lines(forbidden_file)
        param_names = self.all_parameters.keys()

        # check all variables are in the expression
        for expression in expressions:
            # create expression
            try:
                ee = compile(parse(expression, mode='eval'), expression, 'eval')
            except Exception as e:
                raise ParameterDefinitionError("Forbidden expression " + expression + " is incorrect")

            # check if all variables are parameter names
            for var in ee.co_names:
                if var not in param_names:
                    raise ParameterDefinitionError(
                        "Forbidden expression " + expression + " includes parameter " + var + " that is not defined")

        return expressions

    def is_forbidden(self, param_values):
        """
        Check if a set of parameter values are forbidden

        :param param_values: Dictionary of values (keys are parameter names)
        :return: True if the param_values are forbidden, False otherwise
        """
        return self.forbidden_expr.is_forbidden(param_values)
