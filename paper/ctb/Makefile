WORKSHOP = workshop

all: $(WORKSHOP).pdf

$(WORKSHOP).pdf: $(WORKSHOP).tex references.bib
	pdflatex $(WORKSHOP)
	bibtex $(WORKSHOP)
	pdflatex $(WORKSHOP)
	pdflatex $(WORKSHOP)

workshop: $(WORKSHOP).pdf

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(WORKSHOP).pdf

.PHONY: all workshop clean distclean
