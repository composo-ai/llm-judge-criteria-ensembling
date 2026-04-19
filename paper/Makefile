MAIN = main
WORKSHOP = workshop

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

workshop: $(WORKSHOP).pdf

$(WORKSHOP).pdf: $(WORKSHOP).tex references.bib
	pdflatex $(WORKSHOP)
	bibtex $(WORKSHOP)
	pdflatex $(WORKSHOP)
	pdflatex $(WORKSHOP)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(MAIN).pdf $(WORKSHOP).pdf

.PHONY: all workshop clean distclean
