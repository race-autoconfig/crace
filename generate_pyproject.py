#!/usr/bin/env python3
import textwrap

# === load description ===
csd = {}
with open("crace/settings/description.py") as f:
    exec(f.read(), csd)

# === extract ===
package_name        = csd['package_name']
authors             = csd['authors']
maintainers         = csd['maintainers']
maintainers_email   = csd['maintainers_email']
contributers        = csd['contributers']
contact             = csd['contact']
contact_email       = csd['contact_email']
long_description    = csd['long_description']
description         = csd['description']
url                 = csd['url']
urls                = csd['urls']
copyright           = csd['copyright']
license             = csd['license']
citiation           = csd['citiation']
update_logs         = csd['update_logs']

# === dependencies ===
dependencies = [
        "numpy>=1.19.5",
        "pandas>=1.1.5",
        "scipy>=1.5.4",
        "statutils>=0.1.0",
        "statsmodels>=0.12.2",
        "tzlocal>=4.2",
    ]
opt_dependencies = {
        'all': [
            'mpi4py>=3.0.3'
        ]
    }

# === formats ===
def format_people(lst1, lst2=None, lab1=False, lab2=False):
    """
    lst1: list of names
    lst2: list of emails (optional)
    lab1: whether to include public account as maintainer (optional)
    lab2: whether to include contributors as author (optional)
    """
    if not lst1:
        return "[]"
    out = []

    if lst2:
        # add name and email for maintainers
        for p, e in zip(lst1, lst2):
            out.append(f'{{ name = "{p}", email = "{e}" }}')
    else:
        # add name only
        for p in lst1:
            out.append(f'{{ name = "{p}" }}')

    # add public contact for maintainers
    if lab1:
        out.insert(0, f'{{ name = "{contact}", email = "{contact_email}" }}')

    if lab2:
        # add contributers' name only
        for p in contributers:
            out.append(f'{{ name = "{p}" }}')

    return "[\n  " + ",\n  ".join(out) + "\n]"


def format_list(lst):
    if not lst:
        return "[]"
    return "[\n  " + ",\n  ".join(f'"{x}"' for x in lst) + "\n]"


def format_optional(opt):
    if not opt:
        return ""
    s = "\n[project.optional-dependencies]\n"
    for k, v in opt.items():
        s += f'{k} = {format_list(v)}\n'
    return s

def format_urls(urls):
    if not urls:
        return "[]"
    out = []
    for k, v in urls.items():
        out.append(f'{k} = "{v}"')
    return "\n".join(out)


# === block ===
authors_block = format_people(lst1=authors, lab2=True)
maintainers_block = format_people(lst1=maintainers, lst2=maintainers_email, lab1=True)
deps_block = format_list(dependencies)
opt_block = format_optional(opt_dependencies)

# === dependencies ===
deps_block = "[\n  " + ",\n  ".join(f'"{d}"' for d in dependencies) + "\n]"

# === optional ===
opt_block = ""
if opt_dependencies:
    opt_block = "[project.optional-dependencies]\n"
    for k, v in opt_dependencies.items():
        opt_block += f"""{k} = [\"{", ".join(f"{x}" for x in v)}\"]"""

# === urls ===
urls_block = "[project.urls]\n" + format_urls(urls)

# === generate toml ===
content = f"""
[build-system]
requires = ["setuptools>=70.1", "setuptools_scm", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{package_name}"
dynamic = ["version"]

description = "{description}"
authors = {authors_block}
maintainers = {maintainers_block}
readme = {{ file = "README.md", content-type = "text/markdown" }}

license = "GPL-3.0-or-later"
license-files = ["LICENSE"]
requires-python = ">=3.6"
classifiers = [
  "Intended Audience :: Developers",
  "Intended Audience :: Education", 
  "Intended Audience :: Science/Research",
  "Topic :: Scientific/Engineering :: Artificial Intelligence",
  "Topic :: Software Development :: Libraries :: Python Modules",

  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.6",
  "Programming Language :: Python :: 3.7",
  "Programming Language :: Python :: 3.8",
  "Programming Language :: Python :: 3.9",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
  "Programming Language :: Python :: 3.12",
  "Programming Language :: Python :: 3.13",
  "Programming Language :: Python :: 3.14",

  "Operating System :: OS Independent",
]

dependencies = {deps_block}

{opt_block}

{urls_block}

[project.scripts]
crace = "crace.scripts:crace_run"

[tool.setuptools]
packages = {{ find = {{}} }}
include-package-data = true

[tool.setuptools.package-data]
crace = [
  "settings/*",
  "scripts/*",
  "inst/*",
  "vignettes/*"
]

[tool.setuptools_scm]
version_scheme = "post-release"
write_to = "crace/_version.py"
fallback_version = "{csd['_VERSION']}"

"""

def main():
    # === write ===
    with open("pyproject.toml", "w") as f:
        f.write(textwrap.dedent(content).strip() + "\n")
    print("pyproject.toml generated")

if __name__ == "__main__":
    main()