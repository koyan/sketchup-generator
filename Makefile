IMAGE_NAME := sketchup-generator

.PHONY: build bash

build:
	docker build -t $(IMAGE_NAME) .

bash:
	docker run --rm -it -v $(PWD):/app $(IMAGE_NAME) bash
