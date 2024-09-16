.PHONY: up # pull and push changes
up:
	git pull
	git add .
	if [ -z "$(msg)" ]; then git commit -m "up"; else git commit -m "$(msg)"; fi
	git push

.PHONY: help # generate help message
help:
	echo "Usage: make [target]\n"
	grep '^.PHONY: .* #' Makefile | sed 's/\.PHONY: \(.*\) # \(.*\)/\1	\2/' | expand -t20
