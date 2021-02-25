.DEFAULT: help
help:
	@echo "make test"
	@echo "	   run  pre-commit, test and coverage"
	@echo "make test-only"
	@echo "	   run tests only"
	@echo "make check"
	@echo "	   run  pre-commit"
	@echo "make clean"
	@echo "	   clean up pyc and caches"
	@echo "make clean-git"
	@echo "    clean git branches merged with master"
	@echo "make clean-remote"
	@echo "    clean up outdated remote references"
	@echo "make new-branch"
	@echo "    create a new local branch"

.PHONY: clean
clean:
	@rm -rf */.pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .coverage
	@find . -name '.coverage.*' -exec rm -f {} +
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

.PHONY: test
test: check test-only coverage

.PHONY: check
check:
	pre-commit run --all-files

.PHONY: test-only
test-only:
	pytest tests --doctest-modules --junitxml=junit/test-results.xml --cov=. --cov-report=xml --cov-report=html


.PHONY: clean-git
clean-git:
	git branch --merged | egrep -v "(^\*|master)" | xargs git branch -d

.PHONY: clean-remote
clean-remote:
	git remote prune origin

.PHONY: new-branch
new-branch:
	@echo "What is the branch name?: "; \
	read BRANCH_NAME; \
	git checkout -b $${BRANCH_NAME}; \
	echo "New branch succesfully created ": $${BRANCH_NAME}
