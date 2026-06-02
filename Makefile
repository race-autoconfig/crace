ifdef t
    BASE_TAG_ARG := $(t)
else ifdef base_tag
    BASE_TAG_ARG := $(base_tag)
endif


.PHONY: all build clean tag pyproject

all: pyproject tag

pyproject:
	python3 generate_pyproject.py

build:
	python3 -m build

tag:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "ERROR: Uncommitted changes detected. Please commit or stash your changes before tagging."; \
		exit 1; \
	fi
	@if [ -z "$(BASE_TAG_ARG)" ]; then \
		RAW_TAG=$$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0"); \
		BASE_TAG=$$(echo $$RAW_TAG | sed -E 's/^(v[0-9]+\.[0-9]+\.[0-9]+).*/\1/'); \
	else \
		BASE_TAG="$(BASE_TAG_ARG)"; \
		if ! git rev-parse -q --verify "refs/tags/$$BASE_TAG" > /dev/null; then \
			echo "ERROR: The specified base tag '$$BASE_TAG' does not exist in the Git repository. Please provide a valid base tag."; \
			exit 1; \
		fi; \
	fi; \
	RAW_COUNT=$$(git rev-list --count "$$BASE_TAG"..HEAD 2>/dev/null || echo "0"); \
	if [ "$$RAW_COUNT" -eq 0 ]; then \
		echo "ERROR: No commits found since the base tag '$$BASE_TAG'. Please make some commits before tagging."; \
		exit 1; \
	fi; \
	POST_COUNT=$$((RAW_COUNT - 1)); \
	NEW_TAG="$$BASE_TAG.post$$POST_COUNT"; \
	echo "Base tag: $$BASE_TAG -> New tag: $$NEW_TAG (Commits since base: $$RAW_COUNT, post index: $$POST_COUNT)"; \
	if git rev-parse -q --verify "refs/tags/$$NEW_TAG" > /dev/null; then \
		echo "WARNING: Tag $$NEW_TAG already exists. No new tag will be created."; \
		exit 0; \
	fi; \
	echo "Creating new Git tag: $$NEW_TAG"; \
	git tag -a "$$NEW_TAG" -m "Auto-release $$NEW_TAG"


clean:
	rm -rf build dist *.egg-info crace/_version.py
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	@echo "Cleaned up build directories."
