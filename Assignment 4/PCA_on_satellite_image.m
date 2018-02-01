img = zeros(4,512,512);

% imshow(img1);img1 =  imread('band1.gif');
img(1,:,:) = imread('band1.gif');
img(2,:,:) = imread('band2.gif');
img(3,:,:) = imread('band3.gif');
img(4,:,:) = imread('band4.gif');


mean1=double(0);
mean2=double(0);
mean3=double(0);
mean4=double(0);

for i = 1:512
    for j=1:512
        mean1 = mean1+img(1,i,j);
        mean2 = mean2+img(2,i,j);
        mean3 = mean3+img(3,i,j);
        mean4 = mean4+img(4,i,j);
    end
end

mean1 = mean1*1.0/(512.0*512.0);
mean2 = mean2*1.0/(512.0*512.0);
mean3 = mean3*1.0/(512.0*512.0);
mean4 = mean4*1.0/(512.0*512.0);

T1 = [mean1; mean2; mean3 ;mean4];

cov = zeros(4,4);
for i=1:4
    for j=1:4
        for k=1:512
            for l=1:512
                cov(i,j) = cov(i,j) + (img(i,k,l)-T1(i))*(img(j,k,l)-T1(j));
            end
        end
        cov(i,j) = cov(i,j)/(512*512);
    end
end

eigValue = eig(cov); %eigValue is just lambda
[eigVec, eigVal] = eig(cov); % here eigVal is lambda* I 

%eigV_band1 = img(1,:,:) - eigVal  % A - lambda* I
imgTemp = zeros(4,512*512);
temp = imread('band1.gif');
imgTemp(1,:) = temp(:);
temp = imread('band2.gif');
imgTemp(2,:) = temp(:);
temp = imread('band3.gif');
imgTemp(3,:) = temp(:);
temp = imread('band4.gif');
imgTemp(4,:) = temp(:);

PC=eigVec' * imgTemp;

out_1 = reshape(PC(1,:),512,512);
out_2 = reshape(PC(2,:),512,512);
out_3 = reshape(PC(3,:),512,512);
out_4 = reshape(PC(4,:),512,512);

subplot(2,2,1);
imshow(histeq(uint8(out_1)));
subplot(2,2,2);
imshow(histeq(uint8(out_2)));
subplot(2,2,3);
imshow(histeq(uint8(out_3)));
subplot(2,2,4);
imshow(histeq(uint8(out_4)));

