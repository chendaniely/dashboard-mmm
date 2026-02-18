.PHONY: init
init:
	Rscript -e "if (!requireNamespace('remotes', quietly = TRUE)) install.packages('remotes'); remotes::install_deps()"
	Rscript model.R

.PHONY: deploy
deploy:
	Rscript -e "rsconnect::writeManifest()"
