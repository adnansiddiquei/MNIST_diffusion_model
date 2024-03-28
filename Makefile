IMAGE_NAME=as3438m2cw
CONTAINER_NAME=as3438m1cw-container

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run docker container with src/outputs and report/figures mounted as volumes
run: build
	docker run -it --name $(CONTAINER_NAME) \
	-v "$(PWD)/src/outputs":/usr/src/app/src/outputs \
	-v "$(PWD)/report/figures":/usr/src/app/report/figures \
	-v "$(PWD)/data":/usr/src/app/data \
	-v "$(PWD)/docs":/usr/src/app/docs \
	$(IMAGE_NAME) /bin/bash

# Stop and remove the container and image
clean-docker:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)
	docker rmi $(IMAGE_NAME)

conda-env :
	conda env create -f environment.yml
