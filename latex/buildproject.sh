#!/bin/bash

if [ "$#" -eq 0 ]; then
    name=main
else
    if [ "$#" -gt 1 ]; then
        echo "Too many arguments provided, Quitting."
        exit 1
    else
        name=$1
    fi
fi
echo "Compiling file $name"

pdflatex $name -synctex=1 -interaction=nonstopmode -file-line-error > /dev/null
bibtex $name > /dev/null
pdflatex $name -synctex=1 -interaction=nonstopmode -file-line-error > /dev/null
pdflatex $name -synctex=1 -interaction=nonstopmode -file-line-error > /dev/null

rm $name.aux $name.log $name.synctex.gz $name.toc $name.bbl $name.blg 
mv $name.pdf tmp/
