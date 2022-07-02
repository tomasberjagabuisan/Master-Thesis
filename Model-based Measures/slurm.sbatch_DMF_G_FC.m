#!/bin/bash
#SBATCH --job-name=Int_DMF       
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-101
#SBATCH --output=DMF_G%A_%a.out
#SBATCH --error=DMF_G%A_%a.err

#Load Matlab 2017a module
ml MATLAB

matlab -nojvm -nodisplay<<-EOF

s=str2num(getenv('SLURM_ARRAY_TASK_ID'));

we_range = 0:0.025:2.5;
we = we_range(s); % this is the G parameter

%load data
load('DataSleepW_N3.mat')

C=SC; %Normalized
N=90;  % # parcells
NSUB=15;
NSUBSIM=15; % # simulation
Tmax=200;  % time points of each subject in the dataset %Truncado porque queremos
indexsub=1:NSUB;

TR=2;
Isubdiag = find(tril(ones(N),-1)); %tril(:,-1) 

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
for ii = 1:10
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
    
        bds=BOLD_act(2000:2000:end,:); % the simulated signal  %% Que passa amb això?¿
    
    % built the modeled observables
         
        Tmax2=size(bds,1);
        Phase_BOLD_sim=zeros(N,Tmax2);
        BOLDsim=bds';
        FC_simul2(nsub,:,:)=corrcoef(bds); 
    end
    
    % compute fitting metrics between empirical and simulated observables
    
    FC_simul=squeeze(mean(FC_simul2,1));

    pp = 1;
    PR = 0:0.01:0.99;
    cs=zeros(1,length(PR));
    
    %Integration v1
    for p = PR
        A = abs(FC_simul)>p;
        [~, csize] = get_components(double(A));
        cs(pp) = max(csize);
        pp = pp+1;
    end

    integ(ii) = sum(cs)*0.01/N;

    %Integration v2
    meanFC(ii) = mean(abs(FC_simul),"all")

    %Segregation
    [~, Q2(ii)] = community_louvain(FC_simul,[],[],'negative_sym');
    [~, Q3(ii)] = community_louvain(FC_simul,[],[],'negative_asym');
end 

%%%
save(sprintf('netvars_%d.mat',s),'integ','Q2','Q3','meanFC');

EOF