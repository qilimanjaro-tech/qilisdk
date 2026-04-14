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

.PHONY: help Makefile

multiversion:
	@sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)/html"

.PHONY: multiversion

# Run this to generate the files for translation
gettext:
	@$(SPHINXBUILD) -M gettext "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	LANGUAGE_FLAGS=""
	for lang in $(LANGUAGES); do \
		LANGUAGE_FLAGS="$$LANGUAGE_FLAGS -l $$lang"; \
	done;
	sphinx-intl update -p "$(BUILDDIR)/gettext" $(LANGUAGE_FLAGS)

.PHONY: gettext

# Build the translations
translations:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	for lang in $(LANGUAGES); do \
		$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)/$$lang" $(SPHINXOPTS) -D language=$$lang $(O); \
	done;

.PHONY: translations

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
