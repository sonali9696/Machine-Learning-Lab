clc;
clear;

img = zeros(4,512,512);

img(1,:,:) = imread('band1.gif');
img(2,:,:) = imread('band2.gif');
img(3,:,:) = imread('band3.gif');
img(4,:,:) = imread('band4.gif');

n = 512*512;
K = 5;
e = 0;

count=0;
center = zeros(K,4);
centerNew = zeros(K,4);
dist = zeros(1,K);
cluster = zeros(512,512);
output = zeros(512,512,3);

for i=1:K
    x1 = round(rand()*512);
    x2 = round(rand()*512);
    center(i,:) = img(:,x1,x2);
end

colors = distinguishable_colors(K);

while true
    for i=1:512
        for j=1:512
            a = img(:,round(i),round(j))'; %row vector

            min = 10000000;
            minInd = -1;
            for k=1:K
                dist(1,k) = norm(center(k,:)-a);
                if(min > dist(1,k))
                    min = dist(1,k);
                    minInd = k;
                end
            end
            
            cluster(i,j) = minInd;
            output(i,j,:)= colors(minInd,:);
        end
    end
    
    win = imshow(output,[]);
    waitfor(win);
    
    %find new mean
    
    mean_clus = zeros(K,4); %4 features*no of clusters
    num_pts_clus = zeros(1,K);
    
    for i=1:512
        for j=1:512
            clus = cluster(i,j);
            num_pts_clus(clus) = num_pts_clus(clus) + 1; 
            mean_clus(clus,:)= mean_clus(clus,:)+img(:,i,j)';
        end
    end
    
    for i=1:K
        mean_clus(i,:) = mean_clus(i,:)/num_pts_clus(1,i); 
    end
    
    flag=0;
    for i=1:K
        d = norm(center(i,:)-mean_clus(i,:));
        if(d>e)
            flag=1;
            break;
        end
    end
    
    if(flag == 0)
        break;
    end
    
    center(:,:) = mean_clus(:,:);
end



