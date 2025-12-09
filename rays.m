%% PART 1

% Close all figures and clear variables
close all;
clear;


%% Rays through free space

% Initialize number of rays and distance
num_rays = 8;
d = 0.2;

% Linearly generate angles within a range
x_angles = linspace(-pi/20, pi/20, num_rays);
% Initialize ray matrix with angles and other values set to 0
rays_in = [zeros(1, num_rays); x_angles; zeros(1, num_rays); zeros(1, num_rays)];
% Same as the previous but with an offset on the x axis
rays_in_offset = [0.01*ones(1, num_rays); x_angles; zeros(1, num_rays); zeros(1, num_rays)];

% Propagate both ray matrices
rays_out = propagateFreeSpace(rays_in, d);
rays_out_offset = propagateFreeSpace(rays_in_offset, d);

% Initialize z values for plotting
rays_z = [zeros(1, num_rays); d*ones(1, num_rays)];

figure;
hold on;
% Plot x coord vs. z coord to trace path of all 16 rays
plot(rays_z, [rays_in(1, :); rays_out(1, :)], 'r');
plot(rays_z, [rays_in_offset(1, :); rays_out_offset(1, :)], 'b');
xlabel('z (m)');
ylabel('x (m)');
hold off;


%% Rays through a lens

% Initialize focal length, lens radius, and d_2
f = 0.15;
r_lens = 0.02;
d_2 = 0.75;

% Reuse rays_out from previous section, propagating them through the lens
rays_out_lens = propagateLens(rays_out, f, r_lens, 0);
rays_out_offset_lens = propagateLens(rays_out_offset, f, r_lens, 0);

% Propagate rays again for d_2
rays_out_final = propagateFreeSpace(rays_out_lens, d_2);
rays_out_offset_final = propagateFreeSpace(rays_out_offset_lens, d_2);

% Recalculate num_rays (some are removed if not within the lens radius) and
% generate final z values for plotting. Different for each set of rays
% because of the aforementioned lens radius removal.
num_rays = size(rays_out_final, 2);
rays_z_final = [d*ones(1, num_rays); (d+d_2)*ones(1, num_rays)];
num_rays = size(rays_out_offset_final, 2);
rays_z_offset_final = [d*ones(1, num_rays); (d+d_2)*ones(1, num_rays)];

% Plot rays being propagated through d_1, the lens, and d_2
f = figure;
hold on;
plot(rays_z, [rays_in(1, :); rays_out(1, :)], 'r');
plot(rays_z, [rays_in_offset(1, :); rays_out_offset(1, :)], 'b');
plot(rays_z_final, [rays_out_lens(1, :); rays_out_final(1, :)], 'r');
plot(rays_z_offset_final, [rays_out_offset_lens(1, :); rays_out_offset_final(1, :)], 'b');
xlabel('z (m)');
ylabel('x (m)');
hold off;

% Fix to make the figure appear in the published pdf
exportgraphics(f, "rays_plot.png", "Resolution", 300);
imshow("rays_plot.png");


%% Helper Functions

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