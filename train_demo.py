#!coding=utf-8

from src.vgg16 import Vgg16

def main():
    net = Vgg16()
    net.loadWithUntrainedJson(srcDir="./example")
    net.train(tgtDir='./example')
    assert net.trained

if __name__ == '__main__':
    main()
