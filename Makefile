.PHONY: check-deps bump-patch bump-minor bump-major bump-release bump-prepatch bump-preminor bump-premajor bump-prerelease re-tag

# Ensure dependencies are available
check-deps:
	@command -v git >/dev/null 2>&1 || { echo "Error: 'git' is not installed or not in PATH." >&2; exit 1; }
	@command -v poetry >/dev/null 2>&1 || { echo "Error: 'poetry' is not installed or not in PATH." >&2; exit 1; }

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
	git commit -m "Bump version to $$VERSION"; \
	git tag -a "v$$VERSION" -m "Version $$VERSION"; \
	echo "Bumped version to $$VERSION and tagged as v$$VERSION"

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
