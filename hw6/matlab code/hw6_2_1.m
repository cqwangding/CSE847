% load grey image
img = imread('whitehouse_grey.bmp');
X = double(img);

% randomly remove 50% of pixels
idx = randperm(128*128,128*128/2);
X_missing = X;
X_missing(idx) = 0;
Omega = true(128,128);
Omega(idx) = false;

% different ranks
r_arr = [1, 5, 10, 15, 20, 25, 30];
len = length(r_arr);

% apply hard-impute
X_complete = {};
error = zeros(len,1);
for i=1:len
    X_complete{i} = hardimpute(X_missing, Omega, r_arr(i));
    error(i) = sum(sum((X_complete{i}-X).^2));
end

% plot errors
figure;
plot(r_arr, error, 'x-');
xlabel('Hard impute with rank r');
ylabel('Recovery errors');

% plot recovery images
figure;
hold on;
ax = subplot(9,2,1);
text(0.5, 0.5, 'Original image');
set ( ax, 'visible', 'off');
ax = subplot(9,2,2);
imshow(uint8(X));
ax = subplot(9,2,3);
text(0.5, 0.5, 'Noise image');
set ( ax, 'visible', 'off');
ax = subplot(9,2,4);
imshow(uint8(X_missing));
for i=1:len
    ax = subplot(9,2,i*2+3);
    text(0.5, 0.5,  ['r=' num2str(r_arr(i))]);
    set ( ax, 'visible', 'off');
    ax = subplot(9,2,i*2+4);
    imshow(uint8(X_complete{i}));
end
hold off;
