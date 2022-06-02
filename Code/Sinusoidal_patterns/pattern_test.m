Number_Of_Cycles = 40;
A = 0.5;
B = 0.5;
f = 1; 
Sampling_Frequency = 100;
Start_Time = 0;
End_Time = Number_Of_Cycles*1/f;
Sampling_Period = 1/Sampling_Frequency;
t = (Start_Time: Sampling_Period: End_Time);
y1 = B+A*cos(2*pi*f*t);

for i = 1:numel(y1)
    if y1(i) < 0.5
        y1(i) = 0;
    else
        y1(i) = 1;
    end
end

% plot(t, y1, '-')
% xlabel("t"); ylabel("Amplitude");
% axis([Start_Time End_Time -1.2 1.2]);

figure()
% Vertical pattern
Image_Height = 2160;
im1 = repmat(y1,Image_Height,1);
size(im1)
im1 = imresize(im1, [2160 4096]);
size(im1)
a = imshow(im1);
%impixelinfo(a)
% imwrite(im1, 'Vertical_40.png')

% Horizontal Pattern
% Image_width = 4096;
% im2 = repmat(y1,Image_width,1);
% im2 = imresize(im1, [4096 2160]);
% im2 = rot90(im2);
% imwrite(im2, 'Horizontal.png')
% a = imshow(im2);
% impixelinfo(a)