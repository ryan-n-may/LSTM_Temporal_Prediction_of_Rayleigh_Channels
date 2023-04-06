
classdef LS_Estimator
    properties
        Nfft;
        Nsym;
        Nps;
        int_opt = "spline"; %default
        pilot_loc;
        X;
        Y;
        Xp;

        H_LS;
        Y_LS;

        H_LS32;
    end
    methods
        function [Obj] = load_simulation(Obj, ray)
            Obj.Nfft = ray.Nfft;
            Obj.Nsym = ray.Nsym;
            Obj.Nps = ray.Nps;
            Obj.pilot_loc = ray.pilot_loc;
            Obj.X = ray.X;
            Obj.Y = ray.Y;
        end

        function [Obj] = LS_CE(Obj)
            for nsym=1:Obj.Nsym
                Obj.Xp = Obj.X(nsym, Obj.pilot_loc);
                Np=Obj.Nfft/Obj.Nps; 
                k=1:Np;
                Obj.H_LS32(nsym, k) = Obj.Y(nsym, Obj.pilot_loc(k))./Obj.Xp(k); 
                if lower(Obj.int_opt(1))=='1'
                    method='linear'; 
                else 
                    method='spline'; 
                end
                % Linear/Spline interpolation
                Obj = interpolate(Obj, method, nsym);
                
                Obj.Y_LS(nsym, :) = Obj.Y(nsym, :) ./ Obj.H_LS(nsym, :); 
            end
        end

        function [Obj] = interpolate(Obj, method, nsym)
            
            if lower(method(1))=='l'
                Obj.H_LS(nsym, :) = interp1(Obj.pilot_loc,Obj.H_LS32(nsym, :), [1:Obj.Nfft]);
            else 
                Obj.H_LS(nsym, :) = interp1(Obj.pilot_loc,Obj.H_LS32(nsym, :), [1:Obj.Nfft],'spline');
            end
        end
    end
end