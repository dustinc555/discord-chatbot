FROM rocm/tensorflow

COPY . /project_code
WORKDIR /project_code

RUN apt-get update
RUN apt-get install -y python3-tk
RUN apt-get install -y ipython

RUN pip install -r requirments.txt
