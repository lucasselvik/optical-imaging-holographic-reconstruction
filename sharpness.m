%% PART 3 

% Close all figures and clear variables
close all;
clear;
% Load rays
load('lightField.mat');


%% Finding Objects w/ K-means

% Extracting the angle data (N x 2 matrix)
angle_data = [rays(2,:)' , rays(4,:)'];

% Running K-means to find 3 clusters. 
% idx is the cluster index (1, 2, or 3) for each ray.
% C is a 3x2 matrix containing the centers of the 3 clusters.
K = 3;
[idx, C] = kmeans(angle_data, K);

% Defining the angular width for filtering rays
d_theta = 0.01; 

% Extracting the angles from the rays data
theta_x = rays(2, :);
theta_y = rays(4, :);

for k = 1:K
    % Getting the center angle for the current object (k)
    theta_x_center = C(k, 1);
    theta_y_center = C(k, 2);
    
    % Designing the bounds using filtering logic
    lower_x = theta_x_center - d_theta/2;
    upper_x = theta_x_center + d_theta/2;
    lower_y = theta_y_center - d_theta/2;
    upper_y = theta_y_center + d_theta/2;

    % Creating the logical filter
    filter_x = (theta_x >= lower_x) & (theta_x <= upper_x);
    filter_y = (theta_y >= lower_y) & (theta_y <= upper_y);
    angle_filter = filter_x & filter_y;

    % Applying the filter
    filtered_rays = rays(:, angle_filter);

    N_filtered = size(filtered_rays, 2);
    % disp(['View Angle: (', num2str(theta_x_center, '%.4f'), ', ', num2str(theta_y_center, '%.4f'), ') rad']);
    %disp(['Number of rays selected: ', num2str(N_filtered)]);
    
    
%Convolution not used in final report

    % if N_filtered > 0
    %     % 1. Propagating the filtered rays through the lens system
    %     rays_at_image_plane = propagateSensor(filtered_rays, d1, d2, f, r_lens);
    % 
    %     % 2. Extracting the x and y positions at the image plane
    %     rays_x_out = rays_at_image_plane(1, :);
    %     rays_y_out = rays_at_image_plane(3, :);
    % 
    %     % 3. Generating the image
    %     img = rays2img(rays_x_out, rays_y_out, width, Npixels);
    % 
    %     % Using convolution - defining the kernel size
    %     kernel_size = 4;
    %     kernel = ones(kernel_size, kernel_size) / kernel_size^2;
    % 
    %     % 2. Applying convolution to the image matrix
    %     smoothed_img = conv2(img, kernel, 'same');
    % else
    %     disp('Warning: No rays were selected for this center. Try increasing d_theta.');
    % end
end


%% Tuning Focal Length

%%% Rough estimate of f

% Initialize parameters that are held constant
d1 = 0;
d2 = 0.5;
Npixels = 200;
r_lens = 100; % large value chosen so all rays go through lens

n = 100; % number of f values to test
f = linspace(0.1, 0.5, n);
deviation = zeros(1, n);

for i=1:length(f)
    % Creating new variables for each group of rays based on the cluster index
    group1rays = rays(:, idx == 1);
    group2rays = rays(:, idx == 2);
    group3rays = rays(:, idx == 3);
    
    % Propagating group rays through the lens system
    group1propagated = propagateSensor(group1rays, d1, d2, f(i), r_lens);
    group2propagated = propagateSensor(group2rays, d1, d2, f(i), r_lens);
    group3propagated = propagateSensor(group3rays, d1, d2, f(i), r_lens);
    
    % Calculating the center values for each group based on the maximum and minimum of the first row
    center1 = 0.5*(max(group1propagated(1, :)) + min(group1propagated(1, :)));
    center2 = 0.5*(max(group2propagated(1, :)) + min(group2propagated(1, :)));
    center3 = 0.5*(max(group3propagated(1, :)) + min(group3propagated(1, :)));
    
    deviation(i) = std([center1, center2, center3]);
end

% Plot deviation values for each f
figure;
plot(f, deviation);
title('Object Deviation vs. Focal Length');
xlabel('Focal Length (m)');
ylabel('Standard Deviation of Object Midpoints')


%%% Search for optimal f

% Find f value from the rough search that gave the minimum deviation
[~, min_index] = min(deviation);
f_estimate = f(min_index);

f = linspace(f_estimate-0.1, f_estimate+0.1, n);
deviation = zeros(1, n);

for i=1:length(f)
    % Creating new variables for each group of rays based on the cluster index
    group1rays = rays(:, idx == 1);
    group2rays = rays(:, idx == 2);
    group3rays = rays(:, idx == 3);
    
    % Propagating group rays through the lens system
    group1propagated = propagateSensor(group1rays, d1, d2, f(i), r_lens);
    group2propagated = propagateSensor(group2rays, d1, d2, f(i), r_lens);
    group3propagated = propagateSensor(group3rays, d1, d2, f(i), r_lens);
    
    % Calculating the center values for each group based on the maximum and minimum of the first row
    center1 = 0.5*(max(group1propagated(1, :)) + min(group1propagated(1, :)));
    center2 = 0.5*(max(group2propagated(1, :)) + min(group2propagated(1, :)));
    center3 = 0.5*(max(group3propagated(1, :)) + min(group3propagated(1, :)));
    
    deviation(i) = std([center1, center2, center3]);
end

[~, min_index] = min(deviation);
f_optimal = f(min_index);


%%% Plot final images

% Propagate each group
group1propagated = propagateSensor(group1rays, d1, d2, f_optimal, r_lens);
group2propagated = propagateSensor(group2rays, d1, d2, f_optimal, r_lens);
group3propagated = propagateSensor(group3rays, d1, d2, f_optimal, r_lens);


% Find optimal sensor width for each object by finding the range of the
% rays (either in the x or y direction, whichever is greater). Note that
% this strategy relies on the fact that the objects are roughly centered at
% the origin.
group1_width = max(range(group1propagated(1, :)), range(group1propagated(3, :)));
group2_width = max(range(group2propagated(1, :)), range(group2propagated(3, :)));
group3_width = max(range(group3propagated(1, :)), range(group3propagated(3, :)));

% Use rays2img to make final images
[group1_img, group1_x, group1_y] = rays2img(group1propagated(1, :), group1propagated(3, :), group1_width, Npixels);
[group2_img, group2_x, group2_y] = rays2img(group2propagated(1, :), group2propagated(3, :), group2_width, Npixels);
[group3_img, group3_x, group3_y] = rays2img(group3propagated(1, :), group3propagated(3, :), group3_width, Npixels);

% Plotting the three images
figure;

subplot(1, 3, 1); % all three images side by side
imagesc(group1_x, group1_y, group1_img);
axis image; % maintain proper dimensions
axis off; % hide axes to only show image
colormap gray;
title('Object 1');

subplot(1, 3, 2);
imagesc(group2_x, group2_y, group2_img);
axis image;
axis off;
colormap gray;
title('Object 2');

subplot(1, 3, 3);
imagesc(group3_x, group3_y, group3_img);
axis image;
axis off;
colormap gray;
title('Object 3');


% Printing values
fprintf('d1 = %f\n', d1);
fprintf('d2 = %f\n', d2);
fprintf('f = %f\n', f_optimal);
fprintf('object 1 sensor width = %f\n', group1_width);
fprintf('object 2 sensor width = %f\n', group2_width);
fprintf('object 3 sensor width = %f\n', group3_width);
fprintf('Npixels = %f\n', Npixels);


%% Demonstration of Multiple Possible Focusing Values

% Values held constant for both examples
d1 = 0;
Npixels = 200;
r_lens = 100;

% Example 1 (values determined by code from the previous section)
d2 = 0.5;
f = 0.222222;
width = 0.005;

example1Rays = propagateSensor(group1rays, d1, d2, f, r_lens);
[example1_img, example1_x, example1_y] = rays2img(example1Rays(1, :), example1Rays(3, :), width, Npixels);

% Example 2 (values determined by code from the previous section)
d2 = 1;
f = 0.286869;
width = 0.010049;

example2Rays = propagateSensor(group1rays, d1, d2, f, r_lens);
[example2_img, example2_x, example2_y] = rays2img(example2Rays(1, :), example2Rays(3, :), width, Npixels);

% Plotting images
figure;

subplot(1, 2, 1); % images side by side
imagesc(example1_x, example1_y, example1_img);
axis image; % maintain proper dimensions
axis off; % hide axes to only show image
colormap gray;
title('Example 1');

subplot(1, 2, 2);
imagesc(example2_x, example2_y, example2_img);
axis image;
axis off;
colormap gray;
title('Example 2');


%% Helper Functions

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


% Full imaging system propagation
function [rays_out] = propagateSensor(rays, d1, d2, f, r_lens)

rays_out = propagateFreeSpace(rays, d1);
rays_out = propagateLens(rays_out, f, r_lens, 0); % Ignore rays at lens
rays_out = propagateFreeSpace(rays_out, d2);

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


% Propagation through lens, Mf
function [rays_out] = propagateLens(rays, f, r_lens, only_lens_rays)

% Lens propagation matrix based on focal point f
M = [1, 0, 0, 0;
    -1/f, 1, 0, 0;
    0, 0, 1, 0;
    0, 0, -1/f, 1;];

% Matrix multiplication to generate output rays
rays_out = M*rays;

% Find rays that hit the lens using distance to the center
rays_at_lens = sqrt(rays(1, :).^2 + rays(3, :).^2) <= r_lens;

if ~only_lens_rays
    % Ignore rays not at lens
    rays_out = rays_out(:, rays_at_lens);
else
    % Use element-wise multiplication to maintain the angles of rays not at
    % the lens and use the adjusted values for rays at the lens.
    rays_out(2, :) = rays_at_lens.*rays_out(2, :) + ~rays_at_lens.*rays(2, :);
    rays_out(4, :) = rays_at_lens.*rays_out(4, :) + ~rays_at_lens.*rays(4, :);
end

end