"""One time setup of internal project paths."""

import os.path

# This script initializes a new project with the correct directory structure.
# Run `python new_project_setup.py`

DIRECTORIES = [
    "saved",
    "saved/models",
    "saved/output",
    "saved/figures",
    "saved/figures/data_diagnostics",
    "saved/figures/release",
    "data",
    "data/shapefiles",
]


def setup_directory(target):
    if not os.path.isdir(target):
        os.mkdir(target)
        print(f"directory {target} has been created.")
        if "data" in target:
            print("  USER ACTION: You will need to fill this directory with data yourself.")
    else:
        print(f"directory {target} already exists.")


def setup_directories():
    for target in DIRECTORIES:
        setup_directory(target)


if __name__ == "__main__":
    setup_directories()
