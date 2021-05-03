FROM pure/python:3.7-cuda10.2-base

RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install -y build-essential swig
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y libsm6
RUN apt-get install -y libxrender1
RUN apt-get install -y libxext-dev

RUN mkdir /app

ADD . /app/

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python","main.py"]

