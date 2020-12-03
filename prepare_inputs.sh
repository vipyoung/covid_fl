#!/bin/bash

# convert plink formats to text files
#plink --bfile /path_to_M1/finalMask/MergeM1 --recode --tab --out /path_to_M1/finalMask/MergeM1

# to be sure that we all use the same list of features in the same order
wget ftp://ftp.ensembl.org/pub/release-102/tsv/homo_sapiens/Homo_sapiens.GRCh38.102.refseq.tsv.gz
zcat Homo_sapiens.GRCh38.102.refseq.tsv.gz | cut -f1 | grep ENS | sort | uniq > genes_list.txt

exit
