img = zeros(4,512,512);

% imshow(img1);img1 =  imread('band1.gif');
img(1,:,:) = imread('band1.gif');
img(2,:,:) = imread('band4.gif');

%points stored in x1,y1 (river) and x2,y2

x1 = [20 41 40 67 91 112 128 144 159 175 193 212 228 246 264 278 294 316 333 349 366 386 404 427 445 462 479 495 508 441 369 415 303 4 104 132 203 226 248 276 303 334 359 390 223 322 467 497 10 96]
y1 = [155 162 166 173 175 175 167 166 165 165 166 171 179 188 197 207 213 216 215 207 194 191 186 186 180 170 162 153 146 179 197 185 216 151 181 162 171 178 194 204 219 213 205 189 176 223 168 152 154 171]
x2 = [25 90 113 143 153 157 193 223 269 312 354 383 407 437 467 496 505 506 463 403 365 323 284 239 207 162 136 95 42 16 9 7 223 293 319 325 11 7 7 6 36 34 29 33 33 60 80 115 116 92 54 43 73 112 132 137 147 171 199 208 210 195 171 154 171 180 203 212 229 248 261 259 281 309 344 371 393 416 439 462 487 493 500 504 477 412 348 379 424 491 501 424 375 301 503 369 193 116 249 10]
y2 = [42 36 68 89 40 10 9 16 17 16 16 29 17 12 21 16 50 84 85 116 114 105 104 104 100 99 101 94 94 99 76 41 43 54 89 62 231 327 401 487 213 281 355 403 447 488 267 278 331 382 398 432 444 455 388 348 262 266 242 282 345 376 440 477 503 473 448 479 493 462 436 478 492 495 485 492 484 501 508 504 492 438 386 321 309 314 337 385 384 299 251 262 252 270 196 259 261 268 344 428]

mean1=double(0);
mean2=double(0);

for i = 1:50
    mean1 = mean1+img(1,round(x1(i)),round(y1(i)));
    mean2 = mean2+img(2,round(x1(i)),round(y1(i)));
end

mean1 = mean1*1.0/50.0;
mean2 = mean2*1.0/50.0;

T1 = [mean1; mean2];

mean1=0;
mean2=0;

for i = 1:100
    mean1 = mean1+img(1,round(x2(i)),round(y2(i)));
    mean2 = mean2+img(2,round(x2(i)),round(y2(i)));
end

mean1 = mean1*1.0/100;
mean2 = mean2*1.0/100;
T2 = [mean1; mean2];

cov_r = zeros(2,2);
for i=1:2
    for j=1:2
        for k=1:50
            cov_r(i,j) = cov_r(i,j) + (img(i,round(x1(k)),round(y1(k)))-T1(i))*(img(j,round(x1(k)),round(y1(k)))-T1(j));
        end
        cov_r(i,j) = cov_r(i,j)/50;
    end
end

cov_nr = zeros(2,2);
for i=1:2
    for j=1:2
        for k=1:100
            cov_nr(i,j) = cov_nr(i,j) + (img(i,round(x2(k)),round(y2(k)))-T2(i))*(img(j,round(x2(k)),round(y2(k)))-T2(j));
        end
        cov_nr(i,j) = cov_nr(i,j)/100;
    end
end

P1 = 0.3;
P2 = 0.7;
for i=1:512
    for j=1:512
        test_data = [img(1,i,j);img(2,i,j)];
        river_class = ((test_data-T1)' / cov_r) * (test_data-T1);
        nonriver_class = ((test_data-T2)' / (cov_nr)) * (test_data-T2);
        p1 = 1/sqrt(det(cov_r)) * exp(-0.5*river_class);
        p2 = 1/sqrt(det(cov_nr)) * exp(-0.5*nonriver_class);
        if(P1*p1 >= P2*p2)
            out_img(i,j) = 255;
        else
            out_img(i,j) = 0;
        end
    end
end

imshow(out_img);
