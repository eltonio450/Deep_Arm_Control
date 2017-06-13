% Twitching process



####To do: 
Rewrite Reset function
Generate length 0 (idem)
Anti-oja'as Rule = Bigger eigenvector

sensors = importdata('sensors_set4');
actuators = importdata('actuators_set4');
length_0 = importdata('length_0'); %infinity limit
length_0 = length_0(1:9);

% Compute the weights for only one leg
delta_length = sensors.data(:,1:9)-repmat(length_0,size(sensors.data,1),1);
rate_length = sensors.data(:,19:28);
stim = actuators.data;

weight_1 = zeros(9);
weight_2 = zeros(9);
all_weight = zeros(9);

dt = 0.01;
lr = 0.01;

for i=1:size(weight_1,1)
    
    % Filter the activation
    activation = computeActivation(stim(:,i), dt);
    s = find(activation > 0);
    
    % 40 replications of a specific twitching
    for l=1:40
        for k=1:length(s)
            for j=1:size(weight_1,2)
                
                % Anti-Oja rule
                weight_1(i,j) = weight_1(i,j) - lr * activation(s(k)) * (delta_length(s(k)+1,j)+activation(s(k))*weight_1(i,j));
                weight_2(i,j) = weight_2(i,j) - lr * activation(s(k)) * (rate_length(s(k)+1,j)+activation(s(k))*weight_2(i,j));
                
            end
        end
    end
end

% Weights' matrices for both legs
weight_1 =  blkdiag(weight_1,weight_1);
weight_2 =  blkdiag(weight_2,weight_2);

% Saving weights' matrices
dlmwrite('weight_1',weight_1,'\t');
dlmwrite('weight_2',weight_2,'\t');



%% Florin's data plot

time = linspace(0,length(actuators.data)/1000,length(actuators.data));

figure;

subplot(311)
plot(time(1:end/2),actuators.data(1:end/2,10:end))
ylabel('stim')
legend(actuators.textdata{10:end})

subplot(312)
plot(time(1:end/2),sensors.data(1:end/2,10:18))
ylabel('II')

subplot(313)
plot(time(1:end/2),sensors.data(1:end/2,18+9:end))
ylabel('Ia')
xlabel('time (s)')