#!/usr/bin/bash

set -e 

if [ "$(python --version)" != "Python 3.8.4" ]; then
    echo "Please use python3.8.4"
    exit 0
fi

echo "Install packages"
pip3 install -r requirements.txt --break-system-packages

mkdir build
python3 -m grpc_tools.protoc -I protos --python_out=build --grpc_python_out=build protos/prime.proto

# openssl req -newkey rsa:2048 -nodes -keyout certs/privkey.pem -config certs/ca.cnf -out certs/csr.pem
# openssl x509 -req -in certs/csr.pem -signkey certs/privkey.pem -out certs/cert.pem

wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1Ecvge9yeFnwDe0Z3_gnxbcEAdyU18e4D" -O mnist.tar
tar xvf mnist.tar -C $PWD/datasets/mnist
rm mnist.tar

wget --no-check-certificate "https://docs.google.com/uc?export=download&id=1HLlLZQaJstYwFMyGGjJKYyicoobxj55Q" -O cifar10.tar.gz
tar xvf cifar10.tar.gz -C $PWD/datasets/cifar10
rm cifar10.tar.gz

ln -s ci-tests ci_tests
ln -s eval-tests eval_tests

ln -s prime_wrapper prime_torch
ln -s prime_wrapper prime_pytorch_lightning
ln -s prime_wrapper prime_torchvision
ln -s prime_wrapper prime_PIL
ln -s prime_wrapper prime_numpy
ln -s prime_wrapper prime_types
ln -s prime_wrapper prime_time

export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/build

echo "Run training on mnist"
python3 -m pytest -s ci-tests/test_mnist.py

echo "Run training on cifar10"
python3 -m pytest -s ci-tests/test_cifar10.py
