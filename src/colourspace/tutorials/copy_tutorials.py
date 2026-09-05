# -*- coding: utf-8 -*-
"""
Copy the notebook tutorials in the current directory.
"""

try:
    from importlib.resources import files, as_file
except ImportError:
    # try to import from the backport package
    from importlib_resources import files, as_file
import os
import shutil
from pathlib import Path

import colourspace

def copy_tutorials(sub_dir="colourspace_tutorials"):
    current_dir = os.getcwd()
    if not os.path.exists("%s/%s"%(current_dir,sub_dir)):
        os.mkdir("%s/%s"%(current_dir,sub_dir))
    destination_path = Path("%s/%s"%(current_dir,sub_dir))

    for resource in files("colourspace.tutorials").iterdir():
        if resource.is_file():
            with as_file(resource) as source_path:
                if source_path.suffix == ".ipynb":
                    shutil.copy2(source_path, destination_path / source_path.name)
                    print("Copied %s to %s"%(source_path.name, sub_dir))

def main():
    if len(os.sys.argv) > 1: 
        sub_dir = str(os.sys.argv[1])
    else:
        sub_dir = "colourspace_tutorials"
    copy_tutorials(sub_dir)

if __name__ == "__main__":
    main()
