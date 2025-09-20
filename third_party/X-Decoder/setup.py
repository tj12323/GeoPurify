import os
from setuptools import setup, find_packages

all_raw_pkgs = find_packages(where="./xdecoder")
packages = [f"xdecoder.{raw}" for raw in all_raw_pkgs]
package_dir = {}
for raw in all_raw_pkgs:
    pkg_name = f"xdecoder.{raw}"
    package_dir[pkg_name] = os.path.join("xdecoder", raw.replace(".", os.sep))

setup(
    name="xdecoder",
    version="0.1.0",
    description="X-Decoder: treat everything under X-Decoder/ as xdecoder.*",

    # ─────────────────────────────────────────────────────────────────────────
    #  Tell setuptools: these are all the “xdecoder.*” packages to install,
    #  and here is exactly where each one lives on disk.
    # ─────────────────────────────────────────────────────────────────────────
    packages=packages,
    package_dir=package_dir,
    include_package_data=True,
    zip_safe=False,

    # ─────────────────────────────────────────────────────────────────────────
    #  (If X-Decoder has any third‐party dependencies, list them here.)
    # ─────────────────────────────────────────────────────────────────────────
    install_requires=[
        # "torch>=1.12.0",
        # "torchvision>=0.13.0",
        # "pyyaml",
        # "termcolor",
        # …etc. (whatever X-Decoder actually needs)
    ],

    # ─────────────────────────────────────────────────────────────────────────
    #  (Optional) If X-Decoder exposes any console_scripts, list them here.
    # ─────────────────────────────────────────────────────────────────────────
    entry_points={
        "console_scripts": [
            # e.g. "xdecoder_cli = xdecoder.entry:main",
        ],
    },
)
