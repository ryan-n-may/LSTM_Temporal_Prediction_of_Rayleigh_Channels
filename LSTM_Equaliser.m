classdef LSTM_Equaliser
    properties
        Nsym;
        
        layers;
        options;
   
        LS; % holds the current LS object from estimate
        H_IDEAL;
        H_LS_mag;
        H_LS_phase;

        data_mag;
        data_phase;

        net_mag;
        net_phase;

        % Simulation properties
        numHiddenUnits = 250; %complexity
        numChannels;

        muX_phase;
        muX_mag;
        sigmaX_phase;
        sigmaX_mag;

        muT_phase;
        muT_mag;
        sigmaT_phase;
        sigmaT_mag;

        M_mag;
        M_phase;
    end
    methods
        function Obj = Init(Obj)
            Obj.layers =[
                        sequenceInputLayer(Obj.numChannels)
                        lstmLayer(Obj.numHiddenUnits)
                        fullyConnectedLayer(Obj.numChannels)
                        regressionLayer
                        ];
            Obj.options = trainingOptions("adam",...
                            MaxEpochs=100, ...
                            SequencePaddingDirection="left", ...
                            Shuffle='every-epoch', ...
                            Plots="training-progress", ...
                            Verbose=0);
        end

        function Obj = load_simulation(Obj, ray)
            Obj.Nsym = ray.Nsym;
            Obj.H_IDEAL = ray.H;
        end

        function Obj = load_LS_estimate(Obj, LS)
            % The H_LS input is a matrix containing as many 
            % LS estimates as there are symbols, therefore, 
            % we are effectively training for multiple LS estimates 
            % ... except in one batch simulation... 
            Obj.LS = LS;
            Obj.H_LS_mag = abs(Obj.LS.H_LS);
            % we unwrap the phase plot to make it easier to follow in the
            % time series. 
            Obj.H_LS_phase = angle(Obj.LS.H_LS);
                  
            
            Obj.data_mag = {Obj.H_LS_mag(1,:)};
            for nsym = 2:Obj.Nsym
                Obj.data_mag{end+1,1} = Obj.H_LS_mag(nsym, :);
            end

            Obj.data_phase = {unwrap(Obj.H_LS_phase(1,:))};
            for nsym = 2:Obj.Nsym
                Obj.data_phase{end+1,1} = unwrap(Obj.H_LS_phase(nsym, :));
            end
            

            Obj.numChannels = 1;
        end

        function [Obj] = load_trained_networks(Obj, net_mag, net_phase, ...
                                                sigmaX_mag, sigmaX_phase, ...
                                                muX_mag, muX_phase)
            Obj.net_mag = net_mag;
            Obj.net_phase = net_phase;
            Obj.sigmaX_mag = sigmaX_mag;
            Obj.sigmaX_phase = sigmaX_phase;
            Obj.muX_mag = muX_mag;
            Obj.muX_phase = muX_phase;
        end

        function view_LS_estimations(Obj)
            E = 10*log10(Obj.H_LS_mag);
            figure("Name", "LSTM LS Training Estimations");
            square = round(sqrt(Obj.Nsym));
            t = tiledlayout(square, square);
            title(t, "LS estimations for all symbols");
            for nsym=1:Obj.Nsym
                nexttile
                plot(E(nsym, :));
                hold on;
                plot(10*log10(abs(Obj.H_IDEAL)), "--");
                hold off;
                ylabel("Nfft");
                xlabel("Power (dB)");
                title("Symbol " + nsym);
            end
        end

        function view_LS_estimations_alternative(Obj)
            E_draw_m = cell2mat(Obj.data_mag);
            E_draw_p = cell2mat(Obj.data_phase);
            figure("Name", "LSTM LS Training Estimations (Alternative)");
            subplot(2,1,1);
            for(i = 1:size(E_draw_m,1))
                plot(10*log10(E_draw_m(i,:)));
                hold on;
            end
            title("Magnitude");
            xlabel("1:Nfft");
            ylabel("H (power dB)");
            hold off;
            subplot(2,1,2);
            for(i = 1:size(E_draw_p,1))
                plot(unwrap(E_draw_p(i,:)));
                hold on;
            end
            title("Phase");
            xlabel("1:Nfft");
            ylabel("H angle");
            hold off;
            
        end
    
        function Obj = train_LSTM_network(Obj)            
            dataTrain_mag = Obj.data_mag;
            dataTrain_phase = Obj.data_phase;

            save("LSTM_training_data.mat", "dataTrain_mag");
            save("LSTM_training_data.mat", "dataTrain_phase", "-append");
                       
            % Assiging cell contents to X and T 
            for n = 1:numel(dataTrain_mag)
                X_mag = dataTrain_mag{n}; % does this ignore other channels?
                X_phase = dataTrain_phase{n};

                XTrain_mag{n} = X_mag(:, 1:end-1);
                TTrain_mag{n} = X_mag(:, 2:end);

                XTrain_phase{n} = X_phase(:, 1:end-1);
                TTrain_phase{n} = X_phase(:, 2:end);
            end

            Obj.muX_phase = mean(cat(2, XTrain_phase{:}),2);
            Obj.sigmaX_phase = std(cat(2, XTrain_phase{:}), 0, 2);
            Obj.muT_phase = mean(cat(2, TTrain_phase{:}),2);
            Obj.sigmaT_phase = std(cat(2, TTrain_phase{:}), 0, 2);

            Obj.muX_mag = mean(cat(2, XTrain_mag{:}),2);
            Obj.sigmaX_mag = std(cat(2, XTrain_mag{:}), 0, 2);
            Obj.muT_mag = mean(cat(2, TTrain_mag{:}),2);
            Obj.sigmaT_mag = std(cat(2, TTrain_mag{:}), 0, 2);

            for n = 1:numel(XTrain_mag)
                XTrain_mag{n} = (XTrain_mag{n} - Obj.muX_mag) ./ Obj.sigmaX_mag ./ 2;
                TTrain_mag{n} = (TTrain_mag{n} - Obj.muT_mag) ./ Obj.sigmaT_mag ./ 2;
                XTrain_phase{n} = (XTrain_phase{n} - Obj.muX_phase) ./ Obj.sigmaX_phase ./2;
                TTrain_phase{n} = (TTrain_phase{n} - Obj.muT_phase) ./ Obj.sigmaT_phase ./2;
            end
    
            save("LSTM_training_data.mat", "XTrain_mag", "-append");
            save("LSTM_training_data.mat", "XTrain_phase", "-append");
            save("LSTM_training_data.mat", "TTrain_mag", "-append");
            save("LSTM_training_data.mat", "TTrain_phase", "-append");

           

            net_mag = trainNetwork(XTrain_mag, TTrain_mag, Obj.layers, Obj.options);
            net_phase = trainNetwork(XTrain_phase, TTrain_phase, Obj.layers, Obj.options);
            
            Obj.net_mag = net_mag; Obj.net_phase = net_phase;
            save("LSTM_network.mat", "net_mag");
            save("LSTM_network.mat", "net_phase", "-append");
            muX_mag = Obj.muX_mag;
            sigmaX_mag = Obj.sigmaX_mag;
            muX_phase = Obj.muX_phase;
            sigmaX_phase = Obj.sigmaX_phase;
            save("LSTM_network.mat", "muX_mag", "-append");
            save("LSTM_network.mat", "muX_phase", "-append");
            save("LSTM_network.mat", "sigmaX_mag", "-append");
            save("LSTM_network.mat", "sigmaX_phase", "-append");
        end

        function [Obj] = run_LSTM_network(Obj)
            %% Closed loop forecasting 
            H_primer_mag = Obj.H_LS_mag(1, 1:8); 
            H_primer_phase = unwrap(Obj.H_LS_phase(1, 1:8));

            muX_mag = Obj.muX_mag;
            muX_phase = Obj.muX_phase;
            sigmaX_mag = Obj.sigmaX_mag;
            sigmaX_phase = Obj.sigmaX_phase;

            save("LSTM_testing_data.mat", "H_primer_mag");
            save("LSTM_testing_data.mat", "H_primer_phase", "-append");
            save("LSTM_testing_data.mat", "muX_mag", "-append");
            save("LSTM_testing_data.mat", "muX_phase", "-append");
            save("LSTM_testing_data.mat", "sigmaX_mag", "-append");
            save("LSTM_testing_data.mat", "sigmaX_phase", "-append");

                       
            for n = 1:length(H_primer_mag)
                H_primer_mag(n) = (H_primer_mag(n) - muX_mag(1)) ./ sigmaX_mag(1) ./2;
                H_primer_phase(n) = (H_primer_phase(n) - muX_phase(1)) ./ sigmaX_phase(1) ./2;
            end

            inc = 4;
            INPUT = H_primer_mag;
            [~, Y] = forecast(Obj, INPUT, Obj.net_mag, inc);
            X = [INPUT Y];
            for i = (2*inc):inc:(64-inc)   
                [~, Y] = forecast(Obj, X(1:i), Obj.net_mag, inc);
                X = [X(1:i) Y];
            end
            X_mag = X;

            inc = 4;
            INPUT = H_primer_phase;
            [~, Y] = forecast(Obj, INPUT, Obj.net_phase, inc);
            X = [INPUT Y];
            for i = (2*inc):inc:(64-inc)   
                [~, Y] = forecast(Obj, X(1:i), Obj.net_phase, inc);
                X = [X(1:i) Y];
            end
            X_phase = X;
            
      
            for n = 1:length(X_mag)
                Obj.M_mag(n) = (X_mag(n) .* sigmaX_mag .* 2) + muX_mag;
                Obj.M_phase(n) = (X_phase(n) .* sigmaX_phase .* 2) + muX_phase;
            end

            load LSTM_training_data.mat;

            figure(10);
            subplot(1,3,1);
            plot(abs(Obj.H_IDEAL));
            hold on;
            plot(Obj.M_mag, '--');
            plot(XTrain_mag{1,1}, '-^');
            legend("Rayeligh Channel", "LSTM magnitude stimation");
            
            subplot(1,3,3);
            plot(unwrap(angle(Obj.H_IDEAL)));
            hold on;
            plot(angle(Obj.H_IDEAL));
            plot(Obj.M_phase, "--");
            plot(XTrain_phase{1,1});
            plot(wrapToPi(Obj.M_phase));
            legend("Rayleigh Channel phase", "Rayleigh Channel phase wrapped to pi", ...
                "LSTM phase estimation", "LSTM Phase estimation prior to scaling", ...
                "LSTM phase estimation wrapped to pi");
    
        end

        function [Obj, Y] = forecast(Obj, X, net, numPredictionTimeSteps)
            net = resetState(net);
            [net, Z] = predictAndUpdateState(net, X);
            
            Xt = Z(:, end);
            
            Y = zeros(1, numPredictionTimeSteps);
            
            for t = 1:numPredictionTimeSteps
                [net, Y(:, t)] = predictAndUpdateState(net, Xt);
                Xt = Y(:,t);
            end
        end
    end
end

