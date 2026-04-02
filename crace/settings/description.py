from datetime import datetime

def _get_version():
    try:
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "_version.py")
        path = os.path.abspath(path)
        d = {}
        with open(path) as f:
            exec(f.read(), d)
        return d.get("version", d.get("__version__", "0.0.0"))
    except Exception:
        return "0.0.0"

package_name = "crace"
version = _get_version()
authors = ['Leslie Pérez Cáceres', 'Jonas Kuckling', 'Pablo Contreras', 'Yunshuang Xiao', 'Thomas Stützle']
maintainers = ['Leslie Pérez Cáceres', 'Yunshuang Xiao']
maintainers_email = ['leslie.perez@pucv.cl', 'yunshuang.xiao@ulb.be']
contributers = ['Manuel López-Ibáñez']
contact = 'crace developers'
contact_email = 'race.autoconfig@gmail.com'

description = "crace: A Python implementation of Continuous Racing"
long_description = """
crace: A Python implementation of Continuous Racing
==========================================================================

crace is a Python implementation of continuous racing 
scheme which takes inspiration from the racing mechanism implemented in irace.
It allows non-sequential model convergence during the race and enables the 
design of a more flexible and adaptable configuration procedure.
"""

url = "https://github.com/auto-configurator/crace"

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
The first public version: v1.0.0
"""
