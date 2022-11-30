#!/usr/bin/bash

set -e 

if [ "$(python --version)" != "Python 3.8.4" ]; then
    echo "Please use python3.8.4"
    exit 0
fi

echo "Install packages"
pip3 install -r requirements.txt

echo "[0] Run tests"
python3 -m pytest tests

for dn in mnist cifar_10
do
    wget http://147.46.174.102:37373/$dn.tar
    tar xvf $dn.tar
    cp $dn/* ci-tests/$dn/
    rm -rf $dn
done

echo "Run ci-tests"
ln -s ci-tests ci_tests

echo "[1.1] test_mnist"
python3 -m pytest -s ci-tests/test_mnist.py

echo "[1.2] test_cifar10 (googlenet)"
python3 -m pytest -s ci-tests/test_cifar10.py --model googlenet

echo "[1.3] test_cifar10 (resnet)"
python3 -m pytest -s ci-tests/test_cifar10.py --model resnet
