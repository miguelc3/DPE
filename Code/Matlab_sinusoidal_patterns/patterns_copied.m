% Font: https://stackoverflow.com/questions/64962073/how-to-create-a-sinusoidal-pattern-in-matlab

%% Setting Number of Sample Points to Fit Sinusoidal Cycles
% Aliasing Preface: 
% https://en.wikipedia.org/wiki/Aliasing - what is aliasing 
clc, clear, close all

%Plot 1
Sampling_Frequency = 100; %Sampling frequency
Sampling_Period = 1/Sampling_Frequency;

Start_Time = 0;
End_Time = 5;
t = (Start_Time: Sampling_Period: End_Time);

f = 2; %Frequency of sinusoid
y = sin(2*pi*f*t);
subplot(2,1,1); plot(t,y,'-o');
title("Sinusoid Sampled at: " + num2str(Sampling_Frequency));
xlabel("t"); ylabel("Amplitude");
axis([Start_Time End_Time -1.2 1.2]);

%Plot 2
Sampling_Frequency = 9; %Sampling frequency
Sampling_Period = 1/Sampling_Frequency;

Start_Time = 0;
End_Time = 5;
t = (Start_Time: Sampling_Period: End_Time);


f = 2; %Frequency of sinusoid
y = sin(2*pi*f*t);
subplot(2,1,2); plot(t,y,'-o');
title("Sinusoid Sampled at: " + num2str(Sampling_Frequency));
xlabel("t"); ylabel("Amplitude");
axis([Start_Time End_Time -1.2 1.2]);

%% Test with other frequencies
clc, clear, close all

f1 = 110; 
f2 = 140;

Number_Of_Samples = 5000*pi*480;
x = linspace(0,pi*480, Number_Of_Samples);

Z1 = 0;
Z2 = 2*pi*1/3;
Z3 = 2*pi*2/3;

im1 = 127 + 127*cos(2*pi*f1*x + Z1);
im2 = 127 + 127*cos(2*pi*f1*x + Z2);
im3 = 127 + 127*cos(2*pi*f1*x + Z3);
im4 = 127 + 127*cos(2*pi*f2*x + Z1);
im5 = 127 + 127*cos(2*pi*f2*x + Z2);
im6 = 127 + 127*cos(2*pi*f2*x + Z3);

clf;
plot(im1);   
hold on
plot(im2);   
plot(im3);   
plot(im4);   
plot(im5);   
plot(im6);   
xlim([0 1000]);

%% Plotting with Respect to Another Variable - time
clc, clear, close all

%******************************************************
%PARAMETERS THAT CAN BE CHANGED
%******************************************************
f = 10; %Frequency of sinusoid
Sampling_Frequency = 1000; %Sampling frequency
Start_Time = 0;
End_Time = 5;
%******************************************************

Sampling_Period = 1/Sampling_Frequency;
t = (Start_Time: Sampling_Period: End_Time);
y = sin(2*pi*f*t);
subplot(2,1,1); plot(t,y,'-o');
title("Sinusoid Sampled at: " + num2str(Sampling_Frequency) + "Hz");
xlabel("t"); ylabel("Amplitude");
axis([Start_Time End_Time -1.2 1.2]);

Image_Height = 480;
im1 = repmat(y,Image_Height,1);
im2 = imresize(im1, [480 640]);
subplot(2,1,2); imshow(im2);
title("Resized Image to be 480px by 640px");
xlabel("Width: 640px"); ylabel("Height: 480px");

% pixel_values = impixel

mask = im2 < 0;
figure
subplot(2,1,1)
imshow(im2)
subplot(2,1,2)
imshow(mask)

%% Using the Number of Cycles to Simulate Sinusoids Frequency
clc, clear, close all

Number_Of_Cycles = 50;
f = 1; 
Sampling_Frequency = 1000;
Start_Time = 0;
End_Time = Number_Of_Cycles*1/f;
Sampling_Period = 1/Sampling_Frequency;
t = (Start_Time: Sampling_Period: End_Time);
y = cos(2*pi*f*t);
Image_Height = 480;
im1 = repmat(y,Image_Height,1);
im1 = imresize(im1, [480 640]);
subplot(2,3,1); imshow(im1);
title(num2str(Number_Of_Cycles) + " Cycles in Image");


%Main parameter to adjust
Number_Of_Cycles = 100;
End_Time = Number_Of_Cycles*1/f;
Sampling_Period = 1/Sampling_Frequency;
t = (Start_Time: Sampling_Period: End_Time);
y = cos(2*pi*f*t);
im2 = repmat(y,Image_Height,1);
im2 = imresize(im2, [480 640]);
subplot(2,3,2); imshow(im2);
title(num2str(Number_Of_Cycles) + " Cycles in Image");


%Main parameter to adjust%
Number_Of_Cycles = 5;
End_Time = Number_Of_Cycles*1/f;
t = (Start_Time: Sampling_Period: End_Time);
y = cos(2*pi*f*t);
im3 = repmat(y,Image_Height,1);
im3 = imresize(im3, [480 640]);
subplot(2,3,3); imshow(im3);
title(num2str(Number_Of_Cycles) + " Cycles in Image");


%Main parameter to adjust%
Number_Of_Cycles = 10;
End_Time = Number_Of_Cycles*1/f;
t = (Start_Time: Sampling_Period: End_Time);
y = cos(2*pi*f*t);
im4 = repmat(y,Image_Height,1);
im4 = imresize(im4, [480 640]);
subplot(2,3,4); imshow(im4);
title(num2str(Number_Of_Cycles) + " Cycles in Image");


%Main parameter to adjust%
Number_Of_Cycles = 20;
End_Time = Number_Of_Cycles*1/f;
t = (Start_Time: Sampling_Period: End_Time);
y = cos(2*pi*f*t);
im5 = repmat(y,Image_Height,1);
im5 = imresize(im5, [480 640]);
subplot(2,3,5); imshow(im5);
title(num2str(Number_Of_Cycles) + " Cycles in Image");


%Main parameter to adjust%
Number_Of_Cycles = 80;
End_Time = Number_Of_Cycles*1/f;
t = (Start_Time: Sampling_Period: End_Time);
y = cos(2*pi*f*t);
im6 = repmat(y,Image_Height,1);
im6 = imresize(im6, [480 640]);
subplot(2,3,6); imshow(im6);
title(num2str(Number_Of_Cycles) + " Cycles in Image");