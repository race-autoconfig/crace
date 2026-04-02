import os
from setuptools import setup, find_packages, find_namespace_packages
from setuptools.command.sdist import sdist as _sdist
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel

CRACE_HOME = os.path.dirname(__file__)
description_path = os.path.join(CRACE_HOME, 'crace/settings/description.py')
csd = {}

with open (description_path) as f:
    exec(f.read(), csd)

package_name        = csd['package_name']
authors             = ", ".join(csd['authors'])
maintainers         = ",".join(csd['maintainers'])
contact_email       = csd['contact_email']
long_description    = csd['long_description']
description         = csd['description']
url                 = csd['url']
copyright           = csd['copyright']
license             = csd['license']
citiation           = csd['citiation']
update_logs         = csd['update_logs']

def version_scheme(version):
    if version.exact:
        return ""

    node = version.node or ""
    if node.startswith(""):
        node = node[1:5]
    
    print(f".{node}")

    return f".{node}"

class CustomSDist(_sdist):
    """Custom sdist command to generate .zip source distribution."""
    def initialize_options(self):
        _sdist.initialize_options(self)
        self.formats = 'zip,tar,gztar'

setup(
    name=package_name,
    description=description,
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author=authors,
    maintainer=maintainers,
    maintainer_email=contact_email,
    url=url,
    license="GPL-3.0-or-later",
    license_file="LICENSE",

    packages=find_namespace_packages(),
    include_package_data=True,
    package_data={
        "crace": [
            "settings/*.json",
            "scripts/*",
            "inst/**",
            "vignettes/**",
        ],
    },
    scripts=['crace/scripts/crace-parallel'],
    entry_points={
        'console_scripts': [
            'crace=crace.scripts:crace_run',
        ],
    },
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.19.5",
        "pandas>=1.1.5",
        "scipy>=1.5.4",
        "statutils>=0.1.0",
        "statsmodels>=0.12.2",
        "tzlocal>=4.2",
    ],
    extras_require={
        'all': [
            'mpi4py>=3.0.3'
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    cmdclass={
        'sdist': CustomSDist,
        'bdist_wheel': _bdist_wheel,
    },
    use_scm_version={
        "local_scheme": version_scheme,
    }
)