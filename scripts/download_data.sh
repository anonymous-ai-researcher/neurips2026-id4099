#!/bin/bash
# Download datasets from CZI CELLxGENE
set -e
mkdir -p data/raw

echo "Downloading Tabula Sapiens..."
# wget -O data/raw/tabula_sapiens.h5ad "https://cellxgene.cziscience.com/..."
echo "Please download Tabula Sapiens from CZI CELLxGENE and place in data/raw/"

echo "Downloading HLCA..."
echo "Please download HLCA from CZI CELLxGENE and place in data/raw/"

echo "Downloading AIDA v2..."
echo "Please download AIDA v2 from CZI CELLxGENE and place in data/raw/"

echo "Downloading Cell Ontology..."
mkdir -p data/ontology
wget -q -O data/ontology/cl.owl "https://github.com/obophenotype/cell-ontology/releases/download/v2024-04-05/cl.owl"
echo "CL downloaded."

echo "Downloading Uberon..."
wget -q -O data/ontology/uberon.owl "https://github.com/obophenotype/uberon/releases/download/v2024-03-22/uberon.owl"
echo "Uberon downloaded."
