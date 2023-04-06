
classdef Rayliegh_Simulation
    % Properties set my user
    properties 
        Nfft = 64;          % FFT count
        Nsym = 1;           % Number of symbols
        Nps = 2;            % Pilot spacing
        Nbps = 2;           % Number of bits per symbol (Modulated symbol)
        
        Es = 1; 
       
        SNR = 70;           % Signal to noise ratio
        N   = 5;            % Number of channel paths
        fm  = 2e6;          % Maximum doppler shift
    
    % Properites initialised
    
        Ng;
        Nofdm;
        Np;
        M;
        A;
        scale;
        ts_mu;
        ts;
        fs;
    
    % Properties valuable for output
   
        pilot_loc;
        h;
        H;
        X;
        Y;
    end
   
    methods
        function [Obj] = Init(Obj)
            Obj.Ng = Obj.Nfft/8;        
            Obj.Nofdm = Obj.Nfft+Obj.Ng;
            Obj.Np = Obj.Nfft/Obj.Nps;
            Obj.M = Obj.Nbps^2;     
            Obj.A = sqrt(3/2/(Obj.M-1)*Obj.Es); 
            Obj.scale = 1e-5;   
            Obj.ts_mu = 50;
            Obj.ts  = Obj.scale*Obj.ts_mu;
            Obj.fs  = 1/(Obj.ts_mu*Obj.scale);
        end

        function [Obj] = create_channel(Obj)
            Nfft    = 2^max(3, nextpow2(2*Obj.fm/Obj.fs*Obj.N));

            Nifft   = ceil(Nfft*Obj.fs/(2*Obj.fm));
        
            GI = randn(1, Nfft);
            GQ = randn(1, Nfft);
        
            CGI = fft(GI);
            CGQ = fft(GQ);
        
            doppler_coef = Doppler_spectrum(Obj, Obj.fm, Nfft);
        
            f_CGI = CGI.*sqrt(doppler_coef);
            f_CGQ = CGQ.*sqrt(doppler_coef);
        
            Filtered_CGI = [f_CGI(1:Nfft/2) zeros(1,Nifft-Nfft) f_CGI(Obj.Nfft/2+1:Nfft)];
            Filtered_CGQ = [f_CGQ(1:Nfft/2) zeros(1,Nifft-Nfft) f_CGQ(Obj.Nfft/2+1:Nfft)];
        
            hI = ifft(Filtered_CGI);
            hQ = ifft(Filtered_CGQ);
        
            rayEnvolope = sqrt(abs(hI).^2 + abs(hQ).^2);
            
            rayRMS = sqrt(mean(rayEnvolope(1:Obj.N).*rayEnvolope(1:Obj.N)));
        
            Obj.h = complex(real(hI(1:Obj.N)), -real(hQ(1:Obj.N)))/rayRMS;
        end
        %% Internal method, not called by class. 
        function y = Doppler_spectrum(Obj, fd, Nfft)
            df = 2*fd/Nfft;
        
            f(1) = 0; 
            y(1) = 1.5/(pi*fd);
        
            for i = 2:Nfft/2
                f(i) = (i-1)*df;
                y([i Nfft-i+2]) = 1.5/(pi*fd*sqrt(1-(f(i)/fd)^2));
        
            end
        
            nFitPoints = 3;
            kk = [Nfft/2-nFitPoints:Nfft/2];
        
            polyFreq = polyfit(f(kk), y(kk), nFitPoints);
            y((Nfft/2)+1) = polyval(polyFreq, f(Nfft/2)+df);
        end


        function [Obj] = run_channel(Obj)
            for nsym=1:Obj.Nsym
                pilots = [1+1j, -1-1j];
                Xp = pilots(randi([1,2], [1,Obj.Np]));
                msgint=randi(Obj.M,Obj.Nfft-Obj.Np,Obj.M)-1; % bit generation
                Data = Obj.A*qammod(msgint, Obj.M,'gray');
                                
                ip = 0; Obj.pilot_loc = [];
                for k=1:Obj.Nfft
                    if mod(k,Obj.Nps)==1
                        Obj.X(nsym, k)=Xp(floor(k/Obj.Nps)+1); 
                        Obj.pilot_loc=[Obj.pilot_loc k]; 
                        ip = ip+1;
                    else 
                        Obj.X(nsym, k) = Data(k-ip);
                    end
                end
    
                x = ifft(Obj.X(nsym, :),Obj.Nfft); 
                xt = [x(Obj.Nfft-Obj.Ng+1:Obj.Nfft) x]; % IFFT and add CP
                              
                                               
                Obj.H = fft(Obj.h,Obj.Nfft); 
                               
                y_channel = conv(xt,Obj.h); % Channel path (convolution)
            
                yt = awgn(y_channel,Obj.SNR,'measured');
            
                y = yt(Obj.Ng+1:Obj.Nofdm); 
                Obj.Y(nsym,:) = fft(y); % Remove CP and FFT    
            end
        end
    end
end

