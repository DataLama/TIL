#!/bin/sh
wget --no-check-certificate https://klue-benchmark.com.s3.amazonaws.com/app/Competitions/000066/data/ynat-v1.tar.gz -P data
tar -zxvf data/ynat-v1.tar.gz -C data
rm data/ynat-v1.tar.gz
#python build_dataset_YNAT.py

## KLUE-NER
# wget --no-check-certificate https://klue-benchmark.com.s3.amazonaws.com/app/Competitions/000069/data/klue-ner-v1.tar.gz -P data
# tar -zxvf data/klue-ner-v1.tar.gz -C data
# rm data/ynat-v1.tar.gz