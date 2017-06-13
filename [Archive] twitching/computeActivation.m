function [ activation ] = computeActivation( stim, dt )
%COMPUTEACTIVATION Summary of this function goes here
%   Detailed explanation goes here

past_activation = 0;
activation = zeros(length(stim), 1);

for i=1:length(stim)
    
    if(stim(i) > 0)
        tau = 15;
    else
        tau = 50;
    end
    
    activation_derivative = tau * ( -past_activation + stim(i) );
    activation(i) = past_activation + dt * activation_derivative;
    past_activation = activation(i);
    
end
end

