#!/usr/bin/bash

set -e 

if [ "$(python --version)" != "Python 3.11.2" ]; then
    echo "Please use python3.11.2"
    exit 0
fi

echo "Install packages"
pip3 install -r requirements.txt --break-system-packages

mkdir build
python3 -m grpc_tools.protoc -I protos --python_out=build --grpc_python_out=build protos/prime.proto

openssl req -newkey rsa:2048 -nodes -keyout certs/privkey.pem -config certs/ca.cnf -out certs/csr.pem
openssl x509 -req -in certs/csr.pem -signkey certs/privkey.pem -out certs/cert.pem

export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/build

echo "[0] Run tests"
python3 -m pytest tests

wget http://147.46.174.102:37373/mnist.tar
tar xvf mnist.tar
cp -r mnist/* ci-tests/mnist/
rm mnist.tar

wget http://147.46.174.102:37373/cifar10.tar
tar xvf cifar10.tar
cp -r cifar10 ci-tests/cifar10/cifar-10-batches-py
rm cifar10.tar

# mkdir eval-tests/datasets
for dn in cifar10 utkface chestxray
do
    wget http://147.46.174.102:37373/$dn.tar
    tar xvf $dn.tar
    mv $dn eval-tests/datasets/
    rm $dn.tar
done
ln -s eval-tests eval_tests

echo "Run ci-tests"
ln -s ci-tests ci_tests

echo "[1.1] test_mnist"
python3 -m pytest -s ci-tests/test_mnist.py

echo "[1.2] test_cifar10 (googlenet)"
python3 -m pytest -s ci-tests/test_cifar10.py --model googlenet

# echo "[1.3] test_cifar10 (resnet)"
# python3 -m pytest -s ci-tests/test_cifar10.py --model resnet
