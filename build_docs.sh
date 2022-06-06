#!/bin/bash

keyboard_interrupt() {
    exit 1
}

trap keyboard_interrupt SIGINT

RELEASE="$1"

if [[ "$RELEASE" == '' ]]; then
    echo 'Missing required positional argument: `release number`!'
    echo "  For example: '0.1.0', '0.2.1', etc."
    exit 1
fi

PACKAGE_FOLDER_NAME='birdfsd_yolov5'
AUTHOR='Mohammad Alyetama'

python -m pip install -e .

rm -rf docs && mkdir docs

cd docs || exit

sphinx-quickstart --no-sep --project "$PACKAGE_FOLDER_NAME" \
    --author "$AUTHOR" --release "$RELEASE" --language 'en'
cd ..

curl "https://openmoji.org/data/color/svg/1F426.svg" \
    --output docs/_static/logo.svg

sphinx-apidoc -f -o docs "$PACKAGE_FOLDER_NAME"

cd docs || exit

rm conf.py && cp ../conf.example.py conf.py

awk -i inplace 'NR==13{print "   modules.rst"}1' index.rst

make html
cd ..

echo -e "\nopen docs/_build/html/index.html\n"
