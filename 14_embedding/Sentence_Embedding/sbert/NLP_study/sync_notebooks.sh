#!/bin/bash

NOTEBOOK_DIR="./03_nlp_architectures"

for nb in $NOTEBOOK_DIR/*.ipynb
do
  echo "Converting $nb to Python script..."
  jupyter nbconvert --to python "$nb"
done
