# For travis
FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -s -y ppa:fenics-packages/fenics
RUN apt-get update
RUN apt-get install -y python-dolfin python-numpy python-pip
RUN pip install matplotlib pytest pytest-cov

ADD . /root/
RUN cd /root/ && PYTHONPATH=$PWD:$PYTHONPATH MPLBACKEND=Agg pytest --cov maelstrom
RUN cd /root/ && bash <(curl -s https://codecov.io/bash)
