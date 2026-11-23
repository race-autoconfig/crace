from crace.utils.const import *
from crace.utils.format import format_string
import crace.settings.description as csd


def print_crace_header(arguments=None):
    import os.path as osp
    print(f"#{f'':-^{WIDTH-1}}")
    print(f'# {csd.description}')
    print(f'# Version: {csd.version}')
    print(f'# {csd.copyright}')
    print('#')
    print('# Authors: ')
    for x in csd.authors: print(f'#   {x}')
    print('#')
    print('# Contributors: ')
    for x in csd.contributers: print(f'#   {x}')
    print('#')
    print(f'# Contact:\n#   {csd.contact} ({csd.contact_email})')
    print('#')
    print(f'# Check more details at {csd.url} ')
    line = format_string(csd.license, hanging=0, space=True)
    print("\n".join(f"# {line}" for line in line.splitlines()))
    print('#')
    current_file = osp.normpath(__file__)
    CRACE_HOME = osp.dirname(osp.dirname(osp.dirname(current_file)))
    print(f'# installed at: {CRACE_HOME}')
    if arguments: print(f'# called with: {" ".join(arguments)}')
    print(f"#{f'':-^{WIDTH-1}}")


def print_cite_info():
    cit = format_string(csd.citiation, width=WIDTH, hanging=0, space=True)
    print("\n".join(f"# {line}" for line in cit.splitlines()))
    print(f"#{f'':-^{WIDTH-1}}")


def print_version_long():
    print_crace_header()
    print_cite_info()


def print_version_short():
    print(f"{csd.package_name} {csd.version}")


def copy_templates():
    """
    Initialize the tuning directory with template config files. 
    Two scenarios and some script files are copied to current folder.
    """
    import os
    import shutil
    import importlib.util

    dest = os.getcwd()

    spec = importlib.util.find_spec('crace')
    if spec is None or spec.origin is None:
        raise FileNotFoundError(f"! Error: Package {'crace'} is not installed.")
    package_path = os.path.dirname(spec.origin)

    source = os.path.join(package_path, 'inst/templates')

    if not os.path.exists(source):
        raise FileNotFoundError(f"! Error: The file {source} does not exist in the package {'crace'}.")

    destination = os.path.join(dest, 'templates')

    # check if destination already exists
    if os.path.exists(destination):
        ans = input(
            f"! WARNING: destination directory already exists:\n"
            f"!   {destination}\n"
            f"! Overwrite existing files? [Y/n]: "
        ).strip().lower()

        # default
        if ans not in ("", "y", "yes"):
            print("! Copy aborted.")
            return

    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"Copied templates from package {bold}crace{reset} to {destination}")