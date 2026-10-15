#!/usr/bin/env bash
# Create the conda environment and fetch the NLTK stopword list.
set -e
conda env create -f environment.yml || conda env update -f environment.yml
conda run -n ir-part1 python -c "import nltk; nltk.download('stopwords')"
echo "Done. Activate with: conda activate ir-part1"
