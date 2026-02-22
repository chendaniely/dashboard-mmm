.PHONY: setup
setup:
	Rscript -e "if (!requireNamespace('remotes', quietly = TRUE)) install.packages('remotes'); remotes::install_deps()"
	Rscript -e "renv::snapshot()"

.PHONY: model
model:
	Rscript model.R

.PHONY: deploy
deploy:
	Rscript -e "rsconnect::writeManifest()"
