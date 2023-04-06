
classdef MMSE_Estimator
    properties
        SNR;
        Nfft;
        Nsym;
        Nps;
        int_opt = "spline"; %default
        pilot_loc;
        X;
        Y;
        Xp;
        h;

        H_MMSE;
        Y_MMSE;
    end
    methods
        function [Obj] = load_simulation(Obj, ray)
            Obj.Nfft = ray.Nfft;
            Obj.Nsym = ray.Nsym;
            Obj.Nps = ray.Nps;
            Obj.pilot_loc = ray.pilot_loc;
            Obj.X = ray.X;
            Obj.Y = ray.Y;
            Obj.SNR = ray.SNR;
            Obj.h = ray.h;
        end

        %Y,Xp,pilot_loc,Nfft,Nps,h,SNR
        function [Obj] = MMSE_CE(Obj)
            snr = 10^(Obj.SNR*0.1);
            for nsym = 1:Obj.Nsym
                Obj.Xp(nsym, :) = Obj.X(nsym, Obj.pilot_loc);

                Np=Obj.Nfft/Obj.Nps; 
                k=1:Np; 
                H_tilde = Obj.Y(nsym, Obj.pilot_loc(k))./Obj.Xp(nsym, k); 
                k=0:length(Obj.h)-1;  
                hh = Obj.h*Obj.h'; 
                tmp = Obj.h.*conj(Obj.h).*k; 
                r = sum(tmp)/hh;    r2 = tmp*k.'/hh; 
                tau_rms = sqrt(r2-r^2);    
                df = 1/Obj.Nfft;  
                j2pi_tau_df = 1j*2*pi*tau_rms*df;
                K1 = repmat([0:Obj.Nfft-1].',1,Np); 
                K2 = repmat([0:Np-1],Obj.Nfft,1); 
                rf = 1./(1+j2pi_tau_df*(K1-K2*Obj.Nps));
                K3 = repmat([0:Np-1].',1,Np);  
                K4 = repmat([0:Np-1],Np,1); 
                rf2 = 1./(1+j2pi_tau_df*Obj.Nps*(K3-K4));  
                Rhp = rf;
                Rpp = rf2 + eye(length(H_tilde),length(H_tilde))/snr;
                Obj.H_MMSE(nsym, :) = transpose(Rhp*inv(Rpp)*H_tilde.');  

                Obj.Y_MMSE(nsym, :) = Obj.Y(nsym, :) ./ Obj.H_MMSE(nsym, :);
            end
        end
    end
end