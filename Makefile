VENV := $(shell poetry env info --path 2>/dev/null || echo ".venv")
TIMESTAMP := $(VENV)/.poetry_installed
PYTHON_VERSION := 3.12

.PHONY: check-deps bump-patch bump-minor bump-major bump-release bump-prepatch bump-preminor bump-premajor bump-prerelease re-tag

# Ensure dependencies are available
check-deps:
	@command -v git >/dev/null 2>&1 || { echo "Error: 'git' is not installed or not in PATH." >&2; exit 1; }
	@command -v poetry >/dev/null 2>&1 || { echo "Error: 'poetry' is not installed or not in PATH." >&2; exit 1; }

$(VENV): check-deps
	poetry env use $(PYTHON_VERSION)

$(TIMESTAMP): pyproject.toml poetry.lock | $(VENV)
	poetry install
	touch $(TIMESTAMP)

bootstrap: $(TIMESTAMP)

clean:
	rm -rfv .venv dist

format: bootstrap
	poetry run black .

check: bootstrap
	poetry run black --check .
	poetry run ruff check .

test: check
	poetry run pytest

build: format test
	poetry build

re-tag: check-deps
	git push --delete origin refs/tags/$(TAG) &&\
	git tag --delete $(TAG) &&\
	git tag $(TAG) &&\
	git push --tags

# Helper function to bump version, commit, and tag
bump-version: check-deps
	@poetry version $(PART)
	@VERSION=$$(poetry version -s); \
	git add pyproject.toml; \
	git commit -m "Bump version to $$VERSION";

# Targets for different version increments
bump-patch:
	@$(MAKE) PART=patch bump-version

bump-minor:
	@$(MAKE) PART=minor bump-version

bump-major:
	@$(MAKE) PART=major bump-version

bump-release:
	@$(MAKE) PART=release bump-version

bump-prepatch:
	@$(MAKE) PART=prepatch bump-version

bump-preminor:
	@$(MAKE) PART=preminor bump-version

bump-premajor:
	@$(MAKE) PART=premajor bump-version

bump-prerelease:
	@$(MAKE) PART=prerelease bump-version

new-tag:
	@VERSION=$$(poetry version -s); \
	git tag -a "v$$VERSION" -m "Version $$VERSION" &&\
	git push --tags &&\
	echo "Pushed tag v$$VERSION" ||\
	echo "Failed to push tag v$$VERSION"
