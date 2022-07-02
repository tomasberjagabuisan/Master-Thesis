#!/bin/bash
#SBATCH --job-name=DMF_UWS       
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-101
#SBATCH --output=DMF_UWS%A_%a.out
#SBATCH --error=DMF_UWS%A_%a.err

#Load Matlab 2017a module
ml MATLAB

matlab -nojvm -nodisplay<<-EOF

s=str2num(getenv('SLURM_ARRAY_TASK_ID'));

we_range = 0:0.025:2.5;
we = we_range(s); % this is the G parameter

%load data
load('DataSleepW_N3.mat')
load('ts_coma24_AAL_symm.mat')

C=SC; %Normalized
N=90;  % # parcells
NSUB=10; % # Subjects in the empirical data set 
NSUBSIM=10; % # simulation
Tmax=192;  % time points of each subject in the dataset %Truncado porque queremos
indexsub=1:NSUB;

TR=2.4;
slide=3;
window=30;
Isubdiag = find(tril(ones(N),-1)); %tril(:,-1) 

% rename the timeseries as ts
ts=timeseries_UWS24_symm;
%%%%%%%%%%%%%% create filter parameters

flp = .008;           % lowpass frequency of filter
fhi = .08;            % highpass
delt = TR;            % sampling interval
k=2;                  % 2nd order butterworth filter
fnq=1/(2*delt);       % Nyquist frequency
Wn=[flp/fnq fhi/fnq]; % butterworth bandpass non-dimensional frequency
[bfilt2,afilt2]=butter(k,Wn);   % construct the filter

FCtdata2=zeros(NSUB,N,N);
stau=zeros(NSUB,N);

%%%%%%%%%%%%%%  Compute FCD
kk=1;
for nsub=1:NSUB
    ts{nsub} = ts{nsub}(:,1:Tmax);
    tsdata(:,:,nsub)=ts{nsub};
    BOLDdata=(squeeze(tsdata(:,:,nsub)));
    FCdata(nsub,:,:)=corrcoef(BOLDdata');
    for seed=1:N
        x= demean(detrend(BOLDdata(seed,:)));
        x(find(x>3*std(x)))  = 3*std(x);
        x(find(x<-3*std(x))) = -3*std(x);
        timeseriedata(seed,:)= filtfilt(bfilt2,afilt2,x); 
    end

    ii2=1;
    for t=1:slide:Tmax-window 
        jj2=1;
        cc=corrcoef((timeseriedata(:,t:t+window))');
        for t2=1:slide:Tmax-window
            cc2=corrcoef((timeseriedata(:,t2:t2+window))');
            ca=corrcoef(cc(Isubdiag),cc2(Isubdiag));
            if jj2>ii2
                cotsamplingdata(kk)=ca(2);   %% this accumulate all elements of the FCD empirical
                kk=kk+1;
            end
            jj2=jj2+1;
        end
        ii2=ii2+1;
    end
end

FC_emp=squeeze(mean(FCdata,1)); %squeeze del mean de la primera dimensió
FCemp2=FC_emp-FC_emp.*eye(N); %This eliminates self connections of areas
GBCemp=mean(FCemp2,2); %mean of the second dimension of the FCemp2

%% Parameters for the mean field model
dtt   = 1e-3;   % Sampling rate of simulated neuronal activity (seconds)
dt=0.1;

taon=100;
taog=10;
gamma=0.641;
sigma=0.01;
JN=0.15;
I0=0.382;
Jexte=1.;
Jexti=0.7;
w=1.4;

%%%%%%%%%%%%
%% Optimize
%%
gain=1;
Jbal = Balance_J9(we,C);

%% Model FC and FCD

% the model itself
for ii=1:10
    kk=1;
    for nsub=1:NSUBSIM
        neuro_act=zeros(round(1000*(Tmax+10)*TR+1),N);
        sn=0.001*ones(N,1);
        sg=0.001*ones(N,1);
        nn=1;
        for t=0:dt:(1000*(Tmax+10)*TR)
            xn=I0*Jexte+JN*w*sn+we*JN*C*sn-Jbal.*sg;
            xg=I0*Jexti+JN*sn-sg;
            rn=phie_gain(xn,gain);
            rg=phii_gain(xg,gain);
            sn=sn+dt*(-sn/taon+(1-sn)*gamma.*rn./1000.)+sqrt(dt)*sigma*randn(N,1);
            sn(sn>1) = 1;
            sn(sn<0) = 0;
            sg=sg+dt*(-sg/taog+rg./1000.)+sqrt(dt)*sigma*randn(N,1);
            sg(sg>1) = 1;
            sg(sg<0) = 0;
            j=j+1;
            if abs(mod(t,1))<0.01
                neuro_act(nn,:)=rn';
                nn=nn+1;
            end
        end
        nn=nn-1;
        
       %%%% BOLD empirical
        % Friston BALLOON MODEL
        T = nn*dtt; % Total time in seconds
        
        B = BOLD(T,neuro_act(1:nn,1)'); % B=BOLD activity, bf=Fourier transform, f=frequency range)
        BOLD_act = zeros(length(B),N);
        BOLD_act(:,1) = B;
        
        for nnew=2:N
            B = BOLD(T,neuro_act(1:nn,nnew));
            BOLD_act(:,nnew) = B;
        end   
    
        bds=BOLD_act(4800:2400:end,:); % the simulated signal  %% Que passa amb això?¿
    
        % built the modeled observables
         
        Tmax2=size(bds,1);
        Phase_BOLD_sim=zeros(N,Tmax2);
        BOLDsim=bds';
        FC_simul2(nsub,:,:)=corrcoef(bds); 
    
        for seed=1:N
            tsbds = detrend(demean(bds(1:Tmax,seed)'));
            tsbds(find(tsbds>3*std(tsbds))) = 3*std(tsbds);
            tsbds(find(tsbds<-3*std(tsbds))) = -3*std(tsbds);
            timeserie(seed,:)= filtfilt(bfilt2,afilt2,tsbds); 
        end 
    
        %%
        ii2=1;
        for t=1:slide:Tmax2-window
            jj2=1;
            cc=corrcoef((timeserie(:,t:t+window))');
            for t2=1:slide:Tmax2-window 
                cc2=corrcoef((timeserie(:,t2:t2+window))');
                ca=corrcoef(cc(Isubdiag),cc2(Isubdiag));
                if jj2>ii2
                    cotsamplingsim(kk)=ca(2);  %% FCD simulation
                    kk=kk+1;
                end
                jj2=jj2+1;
            end
            ii2=ii2+1;
        end
    end
    FC_simul=squeeze(mean(FC_simul2,1));
    cc=corrcoef(atanh(FC_emp(Isubdiag)),atanh(FC_simul(Isubdiag)));

    FC(ii)=cc(2); %% FC fitting

    FC_euclidian(ii) = sum((FC_emp(Isubdiag)-FC_simul(Isubdiag)).^2,'all');
    FC_euclidian2(ii) = sum(abs(FC_emp(Isubdiag)-FC_simul(Isubdiag)),'all');
    FC_Ssim(ii) = ssim(FC_simul, FC_emp);

    FCsim2=FC_simul-FC_simul.*eye(N);

    [hh(ii) pp(ii) FCD(ii)]=kstest2(cotsamplingdata,cotsamplingsim);  %% FCD fitting
end

% compute fitting metrics between empirical and simulated observables

%%%
save(sprintf('DMF_TS_UWS_%d.mat',s),'FC','FCD','FC_euclidian','FC_euclidian2','FC_Ssim','hh','pp');

EOF