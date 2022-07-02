%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%      MODEL PERTURBATION     %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
load('DataSleepW_N3.mat')
load('Jbal_t.mat')
C=SC; 
N=90;  % # parcells
Tmax=1; % Change depending the time simulated (in seconds)
TR=2; % Will depen on the model perturbed

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
w=1.4;  % this is the G parameter
gain=1;
hhh=1;

we = 2.1;% Change depending state to perturbate
index = round(we*100);
Jbal = Jbal_all(:,index); % Jbal previosly calcultaed and stored for all we

Amplitude_perturb = 0.15; %(nA)
Time_perturb = 10; %(ms)
Num_subs_peturbed = 100; %Number of subjects we want to perturb

%% Simulation
for sub = 1:Num_subs_peturbed
kk=1;
sim_time = 1000*1*TR;
neuro_act=zeros(round(sim_time),N);
sn=0.001*ones(N,1);
sg=0.001*ones(N,1);
nn=1;
for t=0:dt:sim_time
    Perturb = zeros(N,1);
    if t >= sim_time/2 
        if t<(sim_time/2+Time_perturb)
            Perturb=Amplitude_perturb*ones(90,1); %Change for whatever perturbation if needed
        end
    end
    xn=I0*Jexte+JN*w*sn+we*JN*C*sn-Jbal.*sg + Perturb;
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
save(sprintf('PerturbPCI_%d.mat',sub),'neuro_act');
end