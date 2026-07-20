# crace/__init__.py

_IMPORT_MAPPING = {
    'run': 'crace.scripts',
    'main': 'crace.scripts',
    'doc': 'crace.scripts.utils',
    'examples': 'crace.scripts.utils',
    'templates': 'crace.scripts.utils',
    'which': 'crace.scripts.utils',
    'where': 'crace.scripts.utils',
    'Reader': 'crace.utils',
    'Scenario': 'crace.containers.scenario',
    'Parameters': 'crace.containers.parameters',
    'Instances': 'crace.containers.instances',
    'CraceOptions': 'crace.containers.crace_options',
    'CraceResults': 'crace.containers.crace_results',
    'Configurations': 'crace.containers.configurations',
}

_DESC_MAPPING = {
    '__version__': 'version',
    '__author__': 'authors',
    '__maintainers__': 'maintainers',
    '__long_description__': 'long_description',
    '__doc__': 'description'
}

__all__ = list(_IMPORT_MAPPING.keys()) + list(_DESC_MAPPING.keys())


def __getattr__(name: str):
    import importlib
    import crace.settings.description as _csd

    if name in _DESC_MAPPING:
        return getattr(_csd, _DESC_MAPPING[name])

    if name in _IMPORT_MAPPING:
        module = importlib.import_module(_IMPORT_MAPPING[name])

        if name == 'main':
            return getattr(module, 'crace_main')
        if name == 'doc':
            return getattr(module, 'crace_guide')
        if name in ("examples", "templates", "which", "where"):
            func = getattr(module, 'crace_examples')
            def wrapper():
                return func(name)
            return wrapper
        return getattr(module, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    return __all__
