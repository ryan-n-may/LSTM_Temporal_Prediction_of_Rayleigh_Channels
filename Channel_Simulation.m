clear all;
close all;

ray = Rayliegh_Simulation();
ray.SNR = 30;
ray.N = 2;
ray.fm  = 2e3;
ray.Nsym = 64;

ray = ray.Init();
ray = ray.create_channel();
ray = ray.run_channel();

LS = LS_Estimator();
LS = LS.load_simulation(ray);
LS = LS.LS_CE();

MMSE = MMSE_Estimator();
MMSE = MMSE.load_simulation(ray);
MMSE = MMSE.MMSE_CE();

LSTM = LSTM_Equaliser();
LSTM = LSTM.load_simulation(ray);
LSTM = LSTM.load_LS_estimate(LS);
LSTM = LSTM.Init();
LSTM.view_LS_estimations_alternative();
%{
load LSTM_network.mat
LSTM = LSTM.load_trained_networks(net_mag, net_phase, ...
                                    sigmaX_mag, sigmaX_phase, ...
                                    muX_mag, muX_phase);
%}
LSTM = LSTM.train_LSTM_network();
LSTM = LSTM.run_LSTM_network();

LSTM.net_mag

[LSTM_R, LSTM_I] = pol2cart(LSTM.M_phase, wrapToPi(LSTM.M_mag));
LSTM_H = LSTM_R + 1j.*LSTM_I;
for i=1:size(ray.Y, 1)
    Y_LSTM(i,:) = ray.Y(i,:) ./ LSTM_H;
end

figure("Name", "Results");
subplot(2,2,1);
scatter(real(ray.X), imag(ray.X));
xlim([-2,2])
ylim([-2,2])
title("Modulated Symbols");

subplot(2,2,2);
scatter(real(MMSE.Y_MMSE), imag(MMSE.Y_MMSE));
xlim([-2,2])
ylim([-2,2])
title("MMSE Equalised output");

subplot(2,2,3);
scatter(real(LS.Y_LS), imag(LS.Y_LS));
xlim([-2,2])
ylim([-2,2])
title("LS Equalised output");

subplot(2,2,4);
scatter(real(Y_LSTM), imag(Y_LSTM));
xlim([-2,2])
ylim([-2,2])
title("LSTM Equalised output");

