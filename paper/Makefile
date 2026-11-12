.PHONY: all figures clean
all: figures main.pdf
figures:
	python3 make_figures.py
main.pdf: main.tex references.bib sections/*.tex generated/*
	latexmk -pdf -interaction=nonstopmode main.tex
clean:
	latexmk -C; rm -f *.bbl *.run.xml
