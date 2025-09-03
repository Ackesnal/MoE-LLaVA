#!/bin/bash

conda env create -f environment.yml
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
cd .. && pip install .