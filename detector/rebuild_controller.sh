#!/bin/sh
rm -rf ./dist/control_plane-1.0-py3-none-any.whl
pip uninstall -y control_plane
python setup.py bdist_wheel
