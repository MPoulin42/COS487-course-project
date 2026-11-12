#!/usr/bin/env bash
# Create the conda environment (Part 1 + Part 2 dependencies) and fetch the NLTK stopword list.
set -e
conda env create -f environment.yml || conda env update -f environment.yml
conda run -n ir-project python -c "import nltk; nltk.download('stopwords')"
echo "Done. Activate with: conda activate ir-project"
echo "On the CS servers, keep Hugging Face downloads on netstore: export HF_HOME=/path/to/your/netstore/hf_cache"
