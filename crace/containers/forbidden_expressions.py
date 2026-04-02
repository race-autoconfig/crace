class ForbiddenExpressions:
    """
    Expressions for forbidden configurations
    """
    def __init__(self, expressions):
        self.forbidden_expressions = expressions

    def add_forbidden_expression(self, expression):
        self.forbidden_expressions.append(expression)

    def is_forbidden(self, param_values):
        """
        Check if a set of parameter values are forbidden

        :param param_values: Dictionary of values (keys are parameter names)
        :return: True if the param_values are forbidden, False otherwise
        """
        if len(self.forbidden_expressions) > 0:
            results = [ForbiddenExpressions.is_expression_forbidden(x, param_values) for x in self.forbidden_expressions]
            if any(results):
                return True
        return False

    @staticmethod
    def is_expression_forbidden(expression, param_values: dict):
        for param, param_value in param_values.items():
            if param in expression and param_value is None:
                # FIXME: At the moment, this ignores the expression if one element is None
                # This could have the following drawbacks:
                # - The user might want to forbid a value from being None
                # -> This is not a problem since params are None if they are conditional but the condition is not met
                # - Statements where one param can be None without deciding the whole statement
                # (a > 1 or b > 1, when a is None, the statement still depends on b)
                # -> The best example that I can come up with is the "or". But separate statements are joined by "or".
                #    Therefore the user could split the or statements into two statements
                # A possible(?) solution could be to build the AST and replace nodes in the tree for parameters
                # that have a value of None
                # TODO: Check if there is a library for checking logical statements in python that can deal with None
                # print(f"{expression}: {param} is None")
                return False
        return eval(expression, {"__builtins__": None}, param_values)
