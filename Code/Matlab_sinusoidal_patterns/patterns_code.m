% I = B + A*sin(2*pi*f(x)+theta)
clc, clear, close all

Number_Of_Cycles = 25;
A = 0.5;
B = 0.5;
f = 1; 
Sampling_Frequency = 100;
Start_Time = 0;
End_Time = Number_Of_Cycles*1/f;
Sampling_Period = 1/Sampling_Frequency;
t = (Start_Time: Sampling_Period: End_Time);
y1 = B+A*cos(2*pi*f*t);

subplot(3,2,1)
plot(t, y1, '-')
xlabel("t"); ylabel("Amplitude");
axis([Start_Time End_Time -1.2 1.2]);

% Vertical pattern
subplot(3,2,2)
Image_Height = 480;
im1 = repmat(y1,Image_Height,1);
im1 = imresize(im1, [480 640]);
a = imshow(im1);
impixelinfo(a)
title("Vertical")

% Horizontal pattern
Number_Of_Cycles = 25;
A = 0.5;
B = 0.5;
f = 1; 
Sampling_Frequency = 100;
Start_Time = 0;
End_Time = Number_Of_Cycles*1/f;
Sampling_Period = 1/Sampling_Frequency;
t = (Start_Time: Sampling_Period: End_Time);
y2 = B+A*cos(2*pi*f*t);

subplot(3,2,3)
plot(t, y2, '-')
xlabel("t"); ylabel("Amplitude");
axis([Start_Time End_Time -1.2 1.2]);

subplot(3,2,4)
Image_width = 640;
im2 = repmat(y2,Image_width,1);
im2 = imresize(im1, [640 480]);
im2 = rot90(im2);
a = imshow(im2);
impixelinfo(a)
title("Horizontal");

% Sum of the two patterns
subplot(3,2,5)
y3 = y1+y2;
im3 = im1+im2;
plot(t, y3, '-')
xlabel("t"); ylabel("Amplitude");
axis([Start_Time End_Time -3 3]);

subplot(3,2,6)
a = imshow(im3);
impixelinfo(a)


% Number_Of_Cycles = 50;
% A = 1;
% B = 0.5;
% f = 1; 
% Sampling_Frequency = 100;
% Start_Time = 0;
% End_Time = Number_Of_Cycles*1/f;
% Sampling_Period = 1/Sampling_Frequency;
% t = (Start_Time: Sampling_Period: End_Time);
% y2 = B+A*cos(2*pi*f*t);
% 
% subplot(3,2,3)
% plot(t, y2, '-')
% xlabel("t"); ylabel("Amplitude");
% axis([Start_Time End_Time -1.2 1.2]);
% 
% % Vertical pattern
% subplot(3,2,4)
% Image_Height = 480;
% im1 = repmat(y2,Image_Height,1);
% im1 = imresize(im1, [480 640]);
% a = imshow(im1);
% impixelinfo(a)
% title("Vertical")





