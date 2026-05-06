# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
LANGUAGES	  = es ca

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile multiversion gettext translations html

# Build for all versions of the documentation and all languages
multiversion:
	rm -rf "$(BUILDDIR)/en"
	sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)/en"
	for lang in $(LANGUAGES); do \
		rm -rf "$(BUILDDIR)/$$lang"; \
		SPHINX_LOCALE_DIR="$(CURDIR)/locale" sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)/$$lang" -D language=$$lang; \
	done
	find "$(BUILDDIR)/" -name "pygments.css" -exec sed -i 's/@media (prefers-color-scheme: dark)/.dark/g' {} \;; \

# Run this to generate the files for translation
gettext:
	rm -rf "$(BUILDDIR)/"
	$(SPHINXBUILD) -M gettext "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) -D exclude_patterns="code/**,shared/**" $(O)
	LANGUAGE_FLAGS=""; \
	for lang in $(LANGUAGES); do \
		LANGUAGE_FLAGS="$$LANGUAGE_FLAGS -l $$lang"; \
	done; \
	sphinx-intl update -p "$(BUILDDIR)/gettext" $$LANGUAGE_FLAGS

# Build the translations just for main
translations:
	rm -rf "$(BUILDDIR)/en/main"
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/en/main" $(SPHINXOPTS) $(O)
	for lang in $(LANGUAGES); do \
		rm -rf "$(BUILDDIR)/$$lang/main"; \
		$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/$$lang/main" $(SPHINXOPTS) -D language=$$lang $(O); \
	done;
	find "$(BUILDDIR)/" -name "pygments.css" -exec sed -i 's/@media (prefers-color-scheme: dark)/.dark/g' {} \;; \

# Just build the English version, clearing the cache first
html:
	rm -rf "$(BUILDDIR)/en/main"
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/en/main" $(SPHINXOPTS) $(O)
	find "$(BUILDDIR)/" -name "pygments.css" -exec sed -i 's/@media (prefers-color-scheme: dark)/.dark/g' {} \;; \

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)/en/main" $(SPHINXOPTS) $(O)
