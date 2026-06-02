from datetime import datetime

def _get_version():
    import os
    import inspect
    from pathlib import Path

    try:
        current_dir = Path(__file__).resolve().parent.parent
    except NameError:
        caller_file = inspect.stack()[1].filename
        current_dir = Path(caller_file).resolve().parent.parent

    dest_ver = current_dir / "_version.py"
    dest_git = current_dir / "_vergit.py"

    try:
        if os.path.exists(dest_ver):
            d = {}
            with open(dest_ver) as f:
                exec(f.read(), d)
            return d.get("version", d.get("__version__", "0.0.0"))
    except Exception:
        pass

    try:
        import re
        if os.path.exists(dest_git):
            d = {}
            with open(dest_git) as f:
                exec(f.read(), d)
            ver = d.get("version", d.get("__version__", "0.0.0"))
            match = re.search(r"tag:\s*([^,\)]+)", ver)
            if match: return match.group(1)
    except Exception:
        pass

    return "0.0.0"

package_name = "crace"
authors = ['Leslie Pérez Cáceres', 'Jonas Kuckling', 'Pablo Contreras', 'Yunshuang Xiao', 'Thomas Stützle']
maintainers = ['Leslie Pérez Cáceres', 'Yunshuang Xiao']
maintainers_email = ['leslie.perez@pucv.cl', 'yunshuang.xiao@ulb.be']
contributers = ['Manuel López-Ibáñez']
contact = 'crace developers'
contact_email = 'race.autoconfig@gmail.com'

description = "crace: Continuous Racing for Automatic Algorithm Configuration"
long_description = """
crace: Continuous Racing for Automatic Algorithm Configuration
==========================================================================

crace is a Python implementation of continuous racing 
scheme which takes inspiration from the racing mechanism implemented in irace.
It allows non-sequential model convergence during the race and enables the 
design of a more flexible and adaptable configuration procedure.
"""

url = "https://race-autoconfig.github.io/crace/"
url_home = url
url_docs = "https://race-autoconfig.github.io/crace/crace-package.pdf"
url_source = "https://github.com/race-autoconfig/crace/"
url_tracker = "https://github.com/race-autoconfig/crace/issues"
urls = {
    'Homepage': url_home,
    'Documentation': url_docs,
    'Source': url_source,
    'Tracker': url_tracker,
}

copyright = f"Copyright (c) {datetime.now().year}"

license = """
This is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

crace builds upon previous code from the irace package:
  irace: An implementation in R of (Elitist) Iterated Racing
  Copyright (c) 2010-2025 Manuel Lopez-Ibanez 
"""

citiation = """
To cite package 'crace' in publications, an appropriate citation is

  Yunshuang Xiao, Leslie Pérez Cáceres, Manuel López-Ibáñez, and Thomas Stützle. 2023. Algorithm Configuration via Continuously Racing: Preliminary Results. In Proceedings of the Companion Conference on Genetic and Evolutionary Computation (GECCO '23 Companion). Association for Computing Machinery, New York, NY, USA, 1744-1752. https://doi.org/10.1145/3583133.3596408

A BibTeX entry for LaTeX users is

@inproceedings{10.1145/3583133.3596408,
  title = {Algorithm Configuration via Continuously Racing: Preliminary Results},
  author = {Xiao, Yunshuang and Pérez Cáceres, Leslie and López-Ibáñez, Manuel and Stützle, Thomas},
  booktitle = {Proceedings of the Companion Conference on Genetic and Evolutionary Computation},
  pages = {1744-1752},
  year = {2023},
}
"""

update_logs = """
Based on v1.0.0

- src:
  > execution: improve pool management and add support for different python versions
  > race:
    • bug fix for log_state: support for 'n_instances'
    • bug fix for slice budget: min -> max

- package:
  > makefile: support pyproject, build, tag
    • pyptoject: generate pyproject.toml from generate_pyproject.py
    • build: build package
    • tag: add tag for the current version based on (provided) base tag
  > improve generate_pyproject.py
    • add urls
    • improve classfiers
  > delete setup.py

  > base information:
    • modify package description

"""


def __getattr__(name: str):
    if name == "version":
        ver = _get_version()
        globals()["version"] = ver
        return ver
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")