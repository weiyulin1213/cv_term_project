clc; clear all; close all;

img = imread('chair02.bmp');

w = 64;
h = 64;
img = double(imresize(img,[w,h]));

L = getLaplacian1(img,zeros([w,h]),0.0001,1);
A = full(L);
csvwrite('laplacian_chair_64_64.csv',A);
zero_count = sum(sum(A==0));