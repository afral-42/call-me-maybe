NAME         = call_me_maybe
PYTHON       = python3
UV           = uv
SRC_DIR      = src

GREEN        = \033[0;32m
BLUE         = \033[0;34m
YELLOW       = \033[0;33m
RED          = \033[0;31m
RESET        = \033[0m


.PHONY: all
all: install

.PHONY: install
install:
	@echo "$(BLUE)[ (•◡•) ] Installing project dependencies...$(RESET)"
	@$(UV) sync
	@echo "$(GREEN)[ ٩(^‿^)۶ ] Everything is ready!$(RESET)"

.PHONY: run
run: install
	@echo "$(BLUE)[ ᕕ( ᐛ )ᕗ ] Launching the generation engine...$(RESET)"
	@$(UV) run python -m $(SRC_DIR)

.PHONY: debug
debug:
	@echo "$(YELLOW)[ ಠ_ಠ ] Entering debug mode (pdb)...$(RESET)"
	@$(UV) run python -m pdb -m $(SRC_DIR)

.PHONY: clean
clean:
	@echo "$(YELLOW)[ (╯°□°）╯︵ ┻━┻ ] Tossing caches out the window...$(RESET)"
	@rm -rf .mypy_cache .pytest_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "$(GREEN)[ ┬──┬◡ﾉ(° - °ﾉ) ] Back to a clean state.$(RESET)"

.PHONY: lint
lint:
	@echo "$(BLUE)[ (－‸－) ] Checking style (flake8)...$(RESET)"
	@$(UV) run flake8 $(SRC_DIR)
	@echo "$(BLUE)[ ( ͡° ͜ʖ ͡°) ] Checking types (mypy)...$(RESET)"
	@$(UV) run mypy $(SRC_DIR) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
	@echo "$(GREEN)[ ＼(≧▽≦)／ ] Code is flawless!$(RESET)"

.PHONY: lint-strict
lint-strict:
	@echo "$(RED)[ ᕙ(⇀‸↼‶)ᕗ ] Lint in strict mode...$(RESET)"
	@$(UV) run flake8 $(SRC_DIR)
	@$(UV) run mypy --strict $(SRC_DIR)
	@echo "$(GREEN)[ ＼(≧▽≦)／ ] Code is flawless!$(RESET)"


.PHONY: help
help:
	@echo "$(BLUE)Available rules:$(RESET)"
	@echo "  $(GREEN)install$(RESET)     : [ (•◡•) ] Download and sync dependencies"
	@echo "  $(GREEN)run$(RESET)         : [ ᕕ( ᐛ )ᕗ ] Execute the main script via module"
	@echo "  $(GREEN)lint$(RESET)        : [ ( ͡° ͜ʖ ͡°) ] Run mandatory flake8 and mypy flags"
	@echo "  $(GREEN)lint-strict$(RESET) : [ ᕙ(⇀‸↼‶)ᕗ ] Run mypy in --strict mode"
	@echo "  $(GREEN)clean$(RESET)       : [ (╯°□°）╯︵ ┻━┻ ] Remove venv and pycaches"