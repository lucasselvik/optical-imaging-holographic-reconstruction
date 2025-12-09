%% PART 2

% Close all figures and clear variables
close all;
clear;
% Load rays
load('lightField.mat');

sensorWidth = 5e-3; % 5 mm
Npixels = 200; % Number of pixels
rays_x = rays(1, :); 
rays_y = rays(3, :);


%% No Propagation

% Render the image using the rays2img function
img = rays2img(rays_x, rays_y,sensorWidth, Npixels);

% Display the rendered image
figure;
imshow(img);
title(sprintf('Sensor Width = %.2f mm, Number of Pixels = %d', sensorWidth * 1e3, Npixels^2), 'FontSize', 16);


%% Changing the Sensor Width

% Render the image using the rays2img function
img_small_sensor = rays2img(rays_x, rays_y,sensorWidth/2, Npixels);

% Display the rendered image
figure;
hold on
subplot(1,2,1);
imshow(img_small_sensor);
title(sprintf('%.2f mm Sensor, %d Pixels', sensorWidth/2 * 1e3, Npixels^2), 'FontSize', 16);

% Render the image using the rays2img function
img_large_sensor = rays2img(rays_x, rays_y,sensorWidth*2, Npixels);

% Display the rendered image
subplot(1,2,2);
imshow(img_large_sensor);
title(sprintf('%.2f mm Sensor, %d Pixels', sensorWidth*2 * 1e3, Npixels^2), 'FontSize', 16);
hold off


%% Changing the Number of Pixels

% Render the image using the rays2img function
img_less_pixels = rays2img(rays_x, rays_y,sensorWidth, Npixels/5);

% Display the rendered image
figure;
hold on
subplot(1,2,1);
imshow(img_less_pixels);
title(sprintf('%.2f mm Sensor, %d Pixels', sensorWidth * 1e3, (Npixels/5)^2), 'FontSize', 16);

% Render the image using the rays2img function
img_more_pixels = rays2img(rays_x, rays_y,sensorWidth, Npixels*5);

% Display the rendered image
subplot(1,2,2);
imshow(img_more_pixels);
title(sprintf('%.2f mm Sensor, %d Pixels', sensorWidth * 1e3, (Npixels*5)^2), 'FontSize', 16);
hold off


%% Changing Distance

% Initialize width and pixels
width = .005;
Npixels = 200;
% Initialize 5 distances spaced evenly from 0.001 to 0.1 using linspace
d = linspace(0.001, 0.1, 5);

% Loop through each distance value
for i = 1:length(d)
    % Propagate rays through distance d
    prop_rays = propagateFreeSpace(rays, d(i));
    % Use helper function to generate image from rays
    [img, x, y] = rays2img(prop_rays(1, :), prop_rays(3, :), width, Npixels);

    % Plot figure
    figure;
    imagesc(x, y, img);
    % Axis image to ensure proper scale
    axis image;
    colormap gray;
    title(['Propagated Ray Traced Image (d = ' num2str(d(i)) ')']);
    xlabel('X Position (m)');
    ylabel('Y Position (m)');
end

% The result is a blurry image. We cannot create a clear, sharp image because 
% there is no lens, just diverging rays. This means that they will hit the sensor
% in a wide/spread-out area, but a sharp image requires the rays to converge.


%% HELPER FUNCTIONS

function [img,x,y] = rays2img(rays_x,rays_y,width,Npixels)
% rays2img - Simulates the operation of a camera sensor, where each pixel
% simply collects (i.e., counts) all of the rays that intersect it. The
% image sensor is assumed to be square with 100% fill factor (no dead
% areas) and 100% quantum efficiency (each ray intersecting the sensor is
% collected).
%
% inputs:
% rays_x: A 1 x N vector representing the x position of each ray in meters.
% rays_y: A 1 x N vector representing the y position of each ray in meters.
% width: A scalar that specifies the total width of the image sensor in 
%   meters.
% Npixels: A scalar that specifies the number of pixels along one side of 
%   the square image sensor.
%
% outputs:
% img: An Npixels x Npixels matrix representing a grayscale image captured 
%   by an image sensor with a total Npixels^2 pixels.
% x: A 1 x 2 vector that specifies the x positions of the left and right 
%   edges of the imaging sensor in meters.
% y: A 1 x 2 vector that specifies the y positions of the bottom and top 
%   edges of the imaging sensor in meters.
%
% Matthew Lew 11/27/2018
% 11/26/2021 - edited to create grayscale images from a rays_x, rays_y
% vectors
% 11/9/2022 - updated to fix axis flipping created by histcounts2()

% eliminate rays that are off screen
onScreen = abs(rays_x)<width/2 & abs(rays_y)<width/2;
x_in = rays_x(onScreen);
y_in = rays_y(onScreen);

% separate screen into pixels, calculate coordinates of each pixel's edges
mPerPx = width/Npixels;
Xedges = ((1:Npixels+1)-(1+Npixels+1)/2)*mPerPx;
Yedges = ((1:Npixels+1)-(1+Npixels+1)/2)*mPerPx;

% count rays at each pixel within the image
img = histcounts2(y_in,x_in,Yedges,Xedges);    % histcounts2 for some reason assigns x to rows, y to columns


% rescale img to uint8 dynamic range
img = uint8(round(img/max(img(:)) * 255));
x = Xedges([1 end]);
y = Yedges([1 end]);

% figure;
% image(x_edges([1 end]),y_edges([1 end]),img); axis image xy;
end


% Propagation through free space, Md
function [rays_out] = propagateFreeSpace(rays, d)

% Free space propagation matrix based on distance d
M = [1, d, 0, 0;
    0, 1, 0, 0;
    0, 0, 1, d;
    0, 0, 0, 1];

% Matrix multiplication to generate output rays
rays_out = M*rays;
end