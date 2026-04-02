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
contact             = csd['contact']
contact_email       = csd['contact_email']
long_description    = csd['long_description']
description         = csd['description']
url                 = csd['url']
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
def format_people(lst1, lst2=None, email=False):
    if not lst1:
        return "[]"
    out = []
    for p, e in zip(lst1, lst2 or []):
        out.append(f'{{ name = "{p}", email = "{e}" }}')
    if email:
        out.append(f'{{ name = "{contact}", email = "{contact_email}" }}')
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


# === block ===
authors_block = format_people(authors)
maintainers_block = format_people(maintainers, maintainers_email, True)
deps_block = format_list(dependencies)
opt_block = format_optional(opt_dependencies)

# === dependencies ===
deps_block = "[\n  " + ",\n  ".join(f'"{d}"' for d in dependencies) + "\n]"

# === optional ===
opt_block = ""
if opt_dependencies:
    opt_block = "\n[project.optional-dependencies]\n"
    for k, v in opt_dependencies.items():
        opt_block += f"""{k} = [\"{", ".join(f"{x}" for x in v)}\"]\n"""

# === generate toml ===
content = f"""
[build-system]
requires = ["setuptools>=70.1", "setuptools_scm"]
build-backend = "setuptools.build_meta"

[project]
name = "{package_name}"
dynamic = ["version"]

description = "{description}"
readme = {{ file = "README.md", content-type = "text/markdown" }}
requires-python = ">=3.6"

authors = {authors_block}
maintainers = {maintainers_block}

license = {{ text = "GPL-3.0-or-later" }}

dependencies = {deps_block}
{opt_block}

[project.scripts]
crace = "crace.scripts:crace_run"

[tool.setuptools]
packages = {{ find = {{}} }}
include-package-data = true

[tool.setuptools.package-data]
crace = [
  "settings/*.json",
  "scripts/*",
  "inst/**",
  "vignettes/**"
]

[tool.setuptools_scm]
version_scheme = "post-release"
write_to = "crace/_version.py"
"""

def main():
    # === write ===
    with open("pyproject.toml", "w") as f:
        f.write(textwrap.dedent(content).strip() + "\n")
    print("✔ pyproject.toml generated")

if __name__ == "__main__":
    main()