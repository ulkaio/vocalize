SHELL := /bin/bash
APP_NAME := Vocalize
PROJECT := $(APP_NAME).xcodeproj
SCHEME := $(APP_NAME)
CONFIG ?= Release
BUILD_DIR := build
INSTALL_DIR := /Applications

.PHONY: project build run install clean

project:  ## Generate Xcode project (requires xcodegen)
	@if ! command -v xcodegen >/dev/null 2>&1; then \
		echo "xcodegen not found. Install with: brew install xcodegen"; \
		exit 1; \
	fi
	xcodegen generate

build: project  ## Build the app
	xcodebuild -project $(PROJECT) -scheme $(SCHEME) -configuration $(CONFIG) -derivedDataPath $(BUILD_DIR) -destination "platform=macOS"

run:  ## Build and run the app
	@$(MAKE) build
	@open $(BUILD_DIR)/Build/Products/$(CONFIG)/$(APP_NAME).app

install:  ## Install app to /Applications (for stable permissions)
	@$(MAKE) build
	@echo "Installing to $(INSTALL_DIR)/$(APP_NAME).app"
	@rm -rf $(INSTALL_DIR)/$(APP_NAME).app
	@cp -R $(BUILD_DIR)/Build/Products/$(CONFIG)/$(APP_NAME).app $(INSTALL_DIR)/

clean:  ## Remove build artifacts
	rm -rf $(BUILD_DIR) $(PROJECT)
