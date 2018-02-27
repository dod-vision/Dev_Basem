FROM tensorflow/tensorflow:latest-py3
MAINTAINER Leonardo Valeriano Neri <leonardovalerianoneri@gmail.com>

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update -y
RUN pip3 install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80
