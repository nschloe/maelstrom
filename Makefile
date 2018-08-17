VERSION=$(shell python3 -c "import maelstrom; print(maelstrom.__version__)")

default:
	@echo "\"make tag\"?"

tag:
	# Make sure we're on the master branch
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then exit 1; fi
	@echo "Tagging v$(VERSION)..."
	git tag v$(VERSION)
	git push --tags

clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf
	@rm -rf *.egg-info/ build/ dist/

lint:
	black --check maelstrom/ examples/*.py test/*.py
	flake8 maelstrom/ examples/*.py test/*.py

black:
	black maelstrom/ examples/*.py test/*.py
