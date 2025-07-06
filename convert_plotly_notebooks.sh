#!/bin/bash

NOTEBOOK_DIR="05_Data_Visualization/Plotly"

for nb in $NOTEBOOK_DIR/*.ipynb
do
  echo "Converting $nb to Python script..."
  jupyter nbconvert --to python "$nb"
done
