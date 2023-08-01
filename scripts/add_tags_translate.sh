#!/bin/bash

# Assign command line arguments to variables
src_lang=$1
tgt_lang=$2

# Read the input file line by line
while IFS= read -r line; do
    # Add tokens and write to output file
    echo "${src_lang} ${tgt_lang} ${line}"
done
