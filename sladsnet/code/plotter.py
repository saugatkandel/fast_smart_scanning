#! /usr/bin/env python3
import matplotlib.pyplot as plt
import pylab


def plotImage(Image, Num):
    plt.imshow(Image)
    pylab.show()


def plotAfterSLADSSimulation(Im1, Im2, Im3):
    plt.figure(1)
    plt.subplot(131)
    plt.imshow(Im1)
    plt.title('Sampled mask')
    plt.subplot(132)
    plt.imshow(Im2)
    plt.title('Reconstructed Image')
    plt.subplot(133)
    plt.imshow(Im3)
    plt.title('Ground-truth Image')


def plotAfterSLADS(Im1, Im2):
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(Im1)
    plt.title('Sampled mask')
    plt.subplot(122)
    plt.imshow(Im2)
    plt.title('Reconstructed Image')
