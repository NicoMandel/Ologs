# Ologs
Source Files and Paper for Ologs Paper

## Structure
1. `src` folder contains the python files
2. Latex Files are in the directory `latex`
    * images in the subdirectory `imgs`
    * `cls` file for IEEE Aerospace conference
    * Template for IEEE Aerospace conference paper
3. `excels` folder:
    * `classif.xslx` is the literature summary
    * `area-objects` is the cosine similarity of all the terms falling under "area" or "region" in WordNet compared with all objects given in Google's Open Image Detection dataset, highlighted by top 10 entries per column
    

## Additional Requirements:
Download the `wiki-news-300d-1M.vec.zip`-model from the Fasttext website under [this link](https://fasttext.cc/docs/en/english-vectors.html)
Approx. 600 MB! Not included in git.
* python file `wnclient_usage.py` looks in the `tmp` directory at the same level as itself for this file
* The distances are already calculated and included in the `excels` folder

## Gitignore
Compiled files, such as:
* `.pdf`
* `.pyc`
* `.vscode/` config folder
* `.zip` & `.vec` files - to ensure fasttext is not tracked
* General `Tex.gitignore` file from the [github repository](https://github.com/github/gitignore/blob/master/TeX.gitignore)

## Branches & tags
* `master` - current working master
* `LitNico` - for the first draft of the literature review in additional file `lit.tex`
* tag `v0.1` - initial commit 

# Reviews

* [x] `.pdf` file - 
    * [x] Highlight 1: `through` to `with`
    * [x] Highlight 2: `anotation` with double n
* [ ] Reviewer 1:
    * [ ] Overall a good contribution and writeup. Minor editorial comments in attachment.
    * [ ] In general, can the simulation results section be included after the discussion of the literature survey or is this an important order? Traditionally, the survey results are presented prior to the simulation results. 
* [ ] Reviewer 2:
    * [ ] there are some missing spaces between paragraphs throughout the paper - please go through and fix.
    * [ ] In Figure 5, what do the different colors represent? Please add a legend if the colors are significant
* [ ] Self:
    * [x] `as clearly by` to `as clearly indicated`, just before 5. Future Work

