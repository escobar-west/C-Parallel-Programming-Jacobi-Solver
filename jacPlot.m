fileID = fopen('data.txt')
A = fscanf(fileID, '%f');

N = A(1);

T = A(2);

h = A(3);

A = vec2mat(A(4:end), N);

[X, Y] = meshgrid( linspace(-T, T, N), linspace(-T, T, N) );

figure
surf(X, Y, A)
title('JackMPI')

fclose(fileID)
