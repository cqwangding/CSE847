% load usps.mat file
load 'USPS.mat';

p = [10 50 100 200];
[coeff,score,latent] = pca(A);

len = length(p);
error = zeros(len,1);
figure;
hold on;
for i=1:len
    [~,reconstructed] = pcares(A,p(i));
    % calculate error
    error(i) = sum(sum((reconstructed-A).^2, 2));
    % plot first two images
    ax = subplot(len,3,i*3-2);
    text(0.5, 0.5, ['p=' num2str(p(i))]);
    set ( ax, 'visible', 'off')
    A2 = reshape(reconstructed(1,:), 16, 16);
    subplot(len,3,i*3-1);
    imshow(A2');
    A2 = reshape(reconstructed(2,:), 16, 16);
    subplot(len,3,i*3);
    imshow(A2');
end
hold off;

% plot error
figure;
plot(p, error, 'x-');
xlabel('PCA using p principal components');
ylabel('Total reconstruction error');
