# Ologs
Source Files and Paper for Ologs Paper

## Structure
1. `src` folder contains the python files
2. Latex Files are in the directory `latex`
    * images in the subdirectory `imgs`
    * `cls` file for IEEE Aerospace conference
    * Template for IEEE Aerospace conference paper

## Additional Requirements:
Download the `wiki-news-300d-1M.vec.zip`-model from the Fasttext website under [this link](https://fasttext.cc/docs/en/english-vectors.html)
Approx. 600 MB! Not included in git.

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
