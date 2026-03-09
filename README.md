**crace**: An implementation in python of Continuous Racing
=============================================================

[![PyPI](https://img.shields.io/pypi/v/crace?cacheSeconds=3600)][crace-pycode]
[![Python](https://img.shields.io/pypi/pyversions/crace?cacheSeconds=3600)][crace-pycode]
[![Docs](https://img.shields.io/badge/docs-online-blue?cacheSeconds=3600)][crace-homepage]
[![License](https://img.shields.io/github/license/race-autoconfig/crace?cacheSeconds=3600)](docs/references/LICENSE.md)

[ [**Homepage**][crace-homepage] ] [
[**Source Code**][crace-src] ] [
[**User Guide (PDF)**][user-guide] ] [
[**Report A Bug**](https://github.com/race-autoconfig/crace/issues) ] [
[**Discussions**](https://github.com/race-autoconfig/crace/discussions) ]

**Maintainers:** [Leslie Pérez Cáceres](https://orcid.org/0000-0001-5553-6150), [Yunshuang Xiao](https://orcid.org/0009-0003-6752-8627)

**Authors:** [Leslie Pérez Cáceres](https://orcid.org/0000-0001-5553-6150), [Jonas Kuckling](https://orcid.org/0000-0003-2391-2275), 
[Pablo Contreras](https://orcid.org/0009-0001-8801-9018), [Yunshuang Xiao](https://orcid.org/0009-0003-6752-8627), [Thomas Stützle](https://orcid.org/0000-0002-5820-0473).

**Contributors:** [Manuel López-Ibáñez](https://orcid.org/0000-0001-9974-1295).

**Contact**: <race.autoconfig@gmail.com>


**Relevant literature:**

-  Yunshuang Xiao, Leslie Pérez Cáceres, Manuel López-Ibáñez, and Thomas Stützle. 
   [**Algorithm Configuration via Continuously Racing: Preliminary Results**](
    https://doi.org/10.1145/3583133.3596408). *In Proceedings 
   of the Companion Conference on Genetic and Evolutionary Computation (GECCO '23 Companion
   )*. Association for Computing Machinery, New York, NY, USA, 1744–1752. 
   doi: 10.1145/3583133.3596408

- Manuel López-Ibáñez, Jérémie Dubois-Lacoste, Leslie Pérez Cáceres,
  Thomas Stützle, and Mauro Birattari. [**The irace package: Iterated
  Racing for Automatic Algorithm
  Configuration.**](http://dx.doi.org/10.1016/j.orp.2016.09.002)
  *Operations Research Perspectives*, 2016. doi:
  10.1016/j.orp.2016.09.002


Introduction
---------------------------------------

The crace package implements a continuously racing procedure, which is an alternative to 
irace that performs in each iteration a single race.  The continuously racing configurator 
(crace) evaluates, removes and generates new configurations asynchronously, granting a high 
level of flexibility regarding the configuration process when compared to the previous 
iterative scheme. The main use of crace is the automatic configuration of decision and 
optimization algorithms, that is, finding the most appropriate settings of an algorithm 
given a set of instances of a problem. However, it may also be useful for configuring 
other types of algorithms when performance depends on the used parameter settings. It builds
upon the **race** package by Birattari and **irace** package by López-Ibáñez and it is 
implemented in Python. 

You may also find the [**cplot**](https://race-autoconfig.github.io/cplot) package useful 
for analyzing the output of crace.

**Keywords:** automatic algorithm configuration, automatic algorithm design, offline tuning, 
parameter tuning, racing, irace.


### Requisites

Python ($\text{version} \geq 3.6.0$) is required for running crace, but you don't need to 
know the Python language to use it.  Python is freely available and you can download it 
from the [Python project website](https://www.python.org/). See section Quick Start
for a quick installation guide of Python.

For GNU/Linux and macOS, the command-line executable `parallel-crace` requires GNU bash. 
Individual examples may require additional software.


### User guide

A complete [user guide][user-guide] comes with the package. You can access it 
online or, after installing the crace package, invoking from the terminal console the 
following command:
``` bash
crace doc
```

The following is a quick-start guide. The user guide gives more detailed instructions.


Quick Start
--------------

-  Install Python (with your favourite package manager, and see more details below).
-  Install crace. The crace package can be installed automatically with conda, with pip 
    or from the source code. We advise to use the automatic installation unless particular 
    circumstances do not allow it. 
-  Once crace is installed, executable command, `crace` will be added to the active 
   environment. You can call `crace` to execute a single crace run, `crace parallel` to
   execute multiple crace runs and `crace doc` to check the user guide in Bash shell (Linux and 
    MacOS) as well as in Powershell (Windows).
-  You can open the user guide with the following command. This command
    works on Bash shell (Linux and MacOS) and Powershell (Windows)
    with the active environment added to PATH (see detailed instructions below).

``` bash
crace doc
```


### Installing Python
This section gives a quick Python installation guide that will work in most cases. The 
official instructions are available at https://docs.python.org/3/using/index.html.

#### GNU/Linux
You should install Python from your package manager. On a Debian/Ubuntu system it will be 
something like:
```bash
sudo apt-get install python3 python3-dev
```

Once Python is installed, you can use conda, pip or the source code to
install the crace package (see [Installing the crace package][installing-the-crace-package]).

#### macOS
For macOS 10.9 (Jaguar) up until 12.3 (Catalina) the operating system includes Python 2, 
which is no longer supported and is not a good choice for development. You should go to do 
the [official downloads page](https://www.python.org/downloads/) and download the installer.

For newer versions of macOS, Python is no longer included by default and you will have to 
download and install it. You can refer to the [Python documentation]
(https://docs.python.org/3/using/mac.html) for more details on the installation process 
and getting started.

Alternatively, you can just brew the Python formula from the science tap (unfortunately 
it does not come already bottled so you need to have Xcode[^Xcode] installed to compile it):

```bash
# install homebrew if you don’t have it
/bin/bash -c \
  "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
```python
# install python 3
brew tap homebrew/science
brew install python
```

[^Xcode]: Xcode download webpage https://developer.apple.com/xcode/download/.

Once Python is installed, you can use conda, pip or the source code to install the 
crace package (see section Installing the crace package).


#### Windows
You can install Python from the [official website](https://www.python.org/downloads/windows/).
We recommend that you install Python on a filesystem path without spaces, 
special characters or long names, such as `C:\Python`. 
> Note that Python 3.5 - 3.8 cannot be used on Windows XP or earlier and  Python 3.9 - 
> 3.13 cannot be used on Windows 7 or earlier.

Once Python is installed, you can use conda, pip or the source code to install the 
crace package (see section Installing the crace package).


### Installing the crace package

#### Install automatically with conda

If you use **Conda**, execute the following line at the shell to install the package:

```bash
# Best practice, use an environment rather than install in the base env
conda create -n my-env
conda activate my-env
# If you want to install from conda-forge
conda config --env --add channels conda-forge
# The actual install command
conda install crace
```

Alternatively, within the Conda graphical interface (Anaconda Navigator), you may search and 
install crace.

#### Install automatically with pip

If you use **pip**, execute the following line at the shell to install the package:
```bash
pip3 install crace
```

Also when using pip, it's good practice to use a virtual environment and here is the [guide](
https://dev.to/bowmanjd/python-tools-for-managing-virtual-environments-3bko#howto) for using 
the virtual environment.


#####  Create a new virtual environment.
    
venv (for Python 3) allows you to manage separate package installations for different 
projects. It creates a “virtual” isolated Python installation. When you switch projects, 
you can create a new virtual environment which is isolated from other virtual environments. 
You benefit from the virtual environment since packages can be installed confidently and 
will not interfere with another project’s environment.

To create a virtual environment, go to your project’s directory and run the following command. 
This will create a new virtual environment in a local folder named .venv:

```bash
# Unix/macOS
python3 -m venv .venv
```

```powershell
# Windows
py -m venv .venv
```

The second argument is the location to create the virtual environment. Generally, you can 
just create this in your project and call it $.venv$.

venv will create a virtual Python installation in the .venv folder.

##### Activate a virtual environment.

Before you can start installing or using packages in your virtual environment you’ll need 
to activate it. Activating a virtual environment will put the virtual environment-specific 
python and pip executables into your shell’s PATH.

```bash
# Unix/macOS
source .venv/bin/activate
```

```powershell
# Windows
.venv\Scripts\Activate.PS1
```

To confirm the virtual environment is activated, check the location of your Python interpreter:

```bash
# Unix/macOS
which python
```
```powershell
# Windows
where python
```

While the virtual environment is active, the above command will output a filepath that includes 
the .venv directory, by ending with the following:

```bash
# Unix/macOS
.venv/bin/python
```
```powershell
# Windows
.venv\Scripts\python
```

While a virtual environment is activated, pip will install packages into that specific environment. 
This enables you to import and use packages in your Python application.

##### Deactivate a virtual environment.

If you want to switch projects or leave your virtual environment, deactivate the environment:

```bash
deactivate
```

##### Reactivate a virtual environment.

If you want to reactivate an existing virtual environment, follow the same instructions about 
activating a virtual environment. There’s no need to create a new virtual environment.


#### Manual download and installation from source

If the previous installation instructions fail because of insufficient permissions and you do 
not have sufficient admin rights to install crace system-wide, then you need to force a local 
installation. From the crace package [PyPI website][crace-pycode], search and download 
one of the three versions available depending on your operating system:

- `crace_xxx.tar.gz` (Unix/Linux/BSD/macOS)
- `crace-xxx-py3-none-any.whl` (python3, any platform)

Alternatively, you can use wget or curl from the command line to download the file: 
```bash
# PyPI
wget https://files.pythonhosted.org/packages/source/c/crace/crace-xxx.tar.gz
# Github
wget https://github.com/race-autoconfig/crace/
```

To install the package on GNU/Linux, macOS or Windows using the formats .tar.gz, .tar or .zip, 
you should first extract the source files and then navigate to the directory containing the 
extracted files and run the following command to install the package: 

```bash
pip3 install .
```

To install the package on any platform using the Wheel format, you should run the following 
command to install the package:

```bash
pip3 install /path/to/crace-xxx-py3-none-any.whl
```

#### Testing the installation and invoking crace

Once crace has been installed, it can be listed by using command line at the shell or load 
the package and test that the installation was successful by opening a Python console and 
executing:

```bash
# List installed packages at the shell
pip3 list
# Show the information of crace at the shell
pip3 show crace
```

```python
# Load and test at the Python console
import crace
print(crace.__version__)
```

#### Checking the installation path of crace
Once `crace` is installed, the executable command will be added to the environment. 
To check the installation path of crace using command line at the shell:
```bash
which crace
```

The output of this line must be `$PYTHON_HOME/bin/crace`, which means
the path before `/bin/crace` is `$PYTHON_HOME`. Here, `$PYTHON_HOME` should be:
```bash
CRACE_HOME=$PYTHON_HOME/lib/python3.X/site-packages/crace
```

Also, you can check the installation path of crace opening a Python console 
and executing:
```python
import importlib.util
print(importlib.util.find_spec('crace').submodule_search_locations)
```

This command must print out the filesystem path where **crace** is installed. 
In the remainder of this guide, the variable `$CRACE_HOME` is used to
denote this path. When executing any provided command that includes the 
`$CRACE_HOME` variable, do not forget to replace it with the true
installation path of **crace**.

On GNU/Linux or macOS, you can let the operating system know where to find 
**crace** by defining the `$PYTHON_HOME` variable and adding it to the system
`PATH`. Append the following commands to `~/.bash_profile`, `~/.bashrc` or 
`~/.profile`:
```bash
# Replace <CRACE_HOME> with the crace installation path
export CRACE_HOME=<CRACE_HOME>
# Tell operating system where to find crace
export PATH=${PYTHON_HOME}/bin/:$PATH
```

Then, open a new terminal and launch **crace** as follows:

```bash
crace --help
```

Alternatively, you may directly invoke **crace** within the Python
console by executing:
```python
import crace
crace.run("--help")
```


Usage
-----

1.  Create a directory for storing the tuning scenario setup (Bash
    shell):

    ``` bash
    mkdir ./tuning
    cd ./tuning
    ```

2.  Initialize your tuning directory with template config files (Bash
    shell):

    ``` bash
    crace --init
    ```

3.  Modify the generated files following the instructions found within
    each file. In particular,

    - The scripts `target-runner` should be executable. The output of 
      `target-runner` is minimized by default. If you wish to maximize 
      it, just multiply the value by `-1` within the script.

    - In `scenario.txt`, uncomment and assign only the parameters for
      which you need a value different from the default one. For
      example, you may need to set `trainInstancesDir="./Instances/"`.


4.  Put the instances in `./tuning/Instances/`. In addition, you can
    create a file that specifies which instances from that directory
    should be run and which instance-specific parameters to use. See
    `scenario.txt` and `instances-list.txt` for examples. The command
    crace will not attempt to create the execution directory
    (`execDir`), so it must exist before calling crace. The default
    `execDir` is the current directory.

5.  Calling the command in the Bash shell to perform one run of Continuous Race.:

    ``` bash
    cd ./tuning/ && crace
    ```

See the output of `crace --help` for additional available crace parameters. 
Command-line parameters override the scenario setup specified in the `scenario.txt` file.

### Many tuning runs in parallel

On GNU/Linux or macOS, several repetitions of crace in parallel is allowed. 
Call the program `crace parallel` from the Bash shell:

``` bash
cd ./tuning/ && crace parallel N
```

where N is the number of repetitions. By default, the execution
directory of each run of crace will be set to `./exp-dd`, where `dd`
is a number padded with zeroes.

**Be careful**, `crace parallel` will create these directories from
scratch, deleting them first if they already exist.

Check the help of `crace parallel` by running it without parameters.

### Parallelize one tuning

A single run of crace can be done much faster by executing the calls to
`targetRunner` (the runs of the algorithm being tuned) in parallel:

```bash
crace --parallel N
```

where N is the number of slaves to execute the target algorighm in parallel.
See the [user guide][user-guide] for more details.

License
-------

The crace package is Copyright © 2026 and distributed under the [GNU 
General Public License version 3.0](http://www.gnu.org/licenses/gpl-3.0.en.html).

This program is free software (software libre): you can redistribute it and/or 
modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation, either version 3 of the License, or (at your option) 
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. 

**IMPORTANT NOTE:** Please be aware that the fact that this program is released as Free Software 
does not excuse you from scientific propriety, which obligates you to give 
appropriate credit! If you write a scientific paper describing research that 
made substantive use of this program, it is your obligation as a scientist to 
(a) mention the fashion in which this software was used in the Methods section; 
(b) mention the algorithm in the References section. 
The appropriate citation is:

- Yunshuang Xiao, Leslie Pérez Cáceres, Manuel López-Ibáñez, and Thomas Stützle. 2023. 
  Algorithm Configuration via Continuously Racing: Preliminary Results. *In Proceedings 
  of the Companion Conference on Genetic and Evolutionary Computation (GECCO '23 Companion
  )*. Association for Computing Machinery, New York, NY, USA, 1744–1752. 
  doi: [10.1145/3583133.3596408](https://doi.org/10.1145/3583133.3596408)


The **crace** package incorporates code under the GPL from the [irace
package](https://CRAN.R-project.org/package=irace) is Copyright (C) 2010-2025
Manuel Lopez-Ibanez, Jeremie Dubois-Lacoste, Leslie Perez Caceres.


Frequently Asked Questions
--------------------------

The [user guide][user-guide] contains
a list of frequently asked questions.


[user-guide]: https://race-autoconfig.github.io/crace/crace-package.pdf
[crace-homepage]: https://race-autoconfig.github.io/crace/
[crace-src]: https://github.com/race-autoconfig/crace/
[crace-pycode]: https://pypi.org/project/crace/
