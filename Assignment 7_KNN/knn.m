imgTemp = zeros(4,512*512);
temp = imread('band1.gif');
imgTemp(1,:) = temp(:);
temp = imread('band2.gif');
imgTemp(2,:) = temp(:);
temp = imread('band3.gif');
imgTemp(3,:) = temp(:);
temp = imread('band4.gif');
imgTemp(4,:) = temp(:);

img = zeros(4,512,512);

img(1,:,:) = imread('band1.gif');
img(2,:,:) = imread('band2.gif');
img(3,:,:) = imread('band3.gif');
img(4,:,:) = imread('band4.gif');


% training set
x_river = [20 41 40 67 91 112 128 144 159 175 193 212 228 246 264 278 294 316 333 349 366 386 404 427 445 462 479 495 508 441 369 415 303 4 104 132 203 226 248 276 303 334 359 390 223 322 467 497 10 96];
y_river = [155 162 166 173 175 175 167 166 165 165 166 171 179 188 197 207 213 216 215 207 194 191 186 186 180 170 162 153 146 179 197 185 216 151 181 162 171 178 194 204 219 213 205 189 176 223 168 152 154 171];
x_nonriver = [25 90 113 143 153 157 193 223 269 312 354 383 407 437 467 496 505 506 463 403 365 323 284 239 207 162 136 95 42 16 9 7 223 293 319 325 11 7 7 6 36 34 29 33 33 60 80 115 116 92 54 43 73 112 132 137 147 171 199 208 210 195 171 154 171 180 203 212 229 248 261 259 281 309 344 371 393 416 439 462 487 493 500 504 477 412 348 379 424 491 501 424 375 301 503 369 193 116 249 10];
y_nonriver = [42 36 68 89 40 10 9 16 17 16 16 29 17 12 21 16 50 84 85 116 114 105 104 104 100 99 101 94 94 99 76 41 43 54 89 62 231 327 401 487 213 281 355 403 447 488 267 278 331 382 398 432 444 455 388 348 262 266 242 282 345 376 440 477 503 473 448 479 493 462 436 478 492 495 485 492 484 501 508 504 492 438 386 321 309 314 337 385 384 299 251 262 252 270 196 259 261 268 344 428];

train = zeros(150,4);
for i=1:50
    train(i,:) = img(:,round(x_river(i)),round(y_river(i)));
end

for i=1:100
    train(50+i,:) = img(:,round(x_nonriver(i)),round(y_nonriver(i)));
end



%testing
count = 1;
for K = 1:2:9
    dist = zeros(1,150);
    output = zeros(512,512);
    for j=1:512
        for k=1:512
            %4 features of the test point: distance with the training
            %points
            for l=1:150
                b = train(l,:);
                a = zeros(1,4);
                a(1,:) = img(:,j,k);
                d = norm(b-a);
                dist(l) = d;
            end

            %sort distance values
            [dist, I] = sort(dist);
            neigh = I(1,1:K);
            r_count = 0;
            nr_count = 0;

            for q=1:K
                if neigh(q) <= 50
                    r_count = r_count+1;
                else
                    nr_count = nr_count+1;
                end
            end

            if (r_count >= nr_count)
                output(j,k) = 255;
            else
                output(j,k) = 0;
            end
        end
    end

    subplot(2,3,count);
    imshow(output);
    count = count+1;
    count
end

