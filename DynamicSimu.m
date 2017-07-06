function [jointEnergy, oracleEnergy, dynamicEnergy, DMSEnergy, DVSEnergy] = DynamicSimu(bMin, bMax, sMin, sMax, Rs, Ce, Cs, Cl, D, M, W, numberOfTasks, powerRatioIndex)

maxCPUIterations = 5;
maxRadioIterations = 5;
PRI = 9; 

EJoint (1:PRI) = 0;
EOracle (1:PRI) = 0;
EDynamic (1:PRI) = 0;
EJustDMS (1:PRI) = 0;
EJustDVS (1:PRI) = 0;
EMax (1:PRI) = 0;

jointEnergy (1:9, 1:9) = 0;
oracleEnergy (1:9, 1:9) = 0;
dynamicEnergy (1:9, 1:9) = 0;
DMSEnergy (1:9, 1:9) = 0;
DVSEnergy (1:9, 1:9) = 0;
MaxEnergy (1:9, 1:9) = 0;

sOracle(1:W, 1:M) = 0; bOracle(1:W, 1:M) = 0; bDynamic(1:W, 1:M) = 0;

tic

StrPowerRatio = int2str(powerRatioIndex);
C = Cl * 10 ^ ((powerRatioIndex - 5)/2)
% Rs = Rs * 10 ^ (powerRatioIndex - 4)
% Cs = Cs * 4 ^ (powerRatioIndex - 4)
% PRadio = Rs*(Cs*(2^bMax-1)+Ce);
% Alpha = 3;
% Cl = PRadio /(sMax^Alpha);
% C = Cl;

jointEnergyFile = strcat('simResults/jointE', StrPowerRatio);
oracleEnergyFile = strcat('simResults/oracleE', StrPowerRatio);
dynamicEnergyFile = strcat('simResults/dynamicE', StrPowerRatio);
DMSEnergyFile = strcat('simResults/DMSE', StrPowerRatio);
DVSEnergyFile = strcat('simResults/DVSE', StrPowerRatio);
MaxEnergyFile = strcat('simResults/MaxE', StrPowerRatio);

e_cmOracle (1:W, 1:M) = 0;
e_cpOracle (1:W, 1:M) = 0;
e_cmMax (1:M) = 0;
e_cpMax (1:W) = 0;


%     MaxCPUWorkload and MaxRadioWorkload are in terms of ms. We want to
%     find them in terms of the number of bits and cycles.
MaxCPUWorkload = 10;
for CPUIndex = 1:maxCPUIterations * 2 - 1
    
    StrCPU=int2str(CPUIndex);
    jointSpeedFile = strcat('simResults/jointS', StrPowerRatio, StrCPU);
    OracleSpeedFile = strcat('simResults/OracleS', StrPowerRatio, StrCPU);
    DMSSpeedFile = strcat('simResults/DMSS', StrPowerRatio, StrCPU);
    DVSSpeedFile = strcat('simResults/DVSS', StrPowerRatio, StrCPU);
    
    %         Calculate maxCpWl = the maximum computational workload in terms of
    %         CYCLES. 1e-3 is because the MaxCPUWorkload is in miliseconds.
    maxCpWl = MaxCPUWorkload * 1e-3 * sMax;
    MaxRadioWorkload = 10;
    for radioIndex = 1:10 - CPUIndex
        
        %             Calculate the maximum communicational workload in terms of
        %             BITS. 1e-3 is because the MaxRadioWorkload is in miliseconds.
        maxCmWl = Rs * bMax * MaxRadioWorkload * 1e-3;
        Rho = (maxCmWl / M);
        L = (maxCpWl / W);
        %             Run the main algorithm
        
        Lp (1:W) = 1; 
        Lm (1:M) = 1; 
        [sJoint, bJoint] = initialization(maxCmWl, D, C, Rs, maxCpWl, Lm, Lp, bMin, bMax, sMin, sMax, Ce, Cs, M, W);
        dlmwrite(jointSpeedFile, sJoint, 'delimiter', ' ', '-append');
        dlmwrite(jointSpeedFile, bJoint, 'delimiter', ' ', '-append');
        dlmwrite(jointSpeedFile, ' ', '-append');
        
        tCPU = 0;
        clear Lp; Lp (1:1) = 0;
        for cpuReal = 1:W
            if MaxCPUWorkload ~= 0 
                tCPU = tCPU + L/sJoint(cpuReal);
            end
            [sMeh, bDynamic(cpuReal, :)] = initialization(maxCmWl, D - tCPU, C, Rs, 0, Lm, Lp, bMin, bMax, sMin, sMax, Ce, Cs, M, 1);
        end
        dlmwrite(OracleSpeedFile, bDynamic, 'delimiter', ' ', '-append');
        dlmwrite(OracleSpeedFile, ' ', '-append');
                        
        LpOracle = 1;
        LmOracle = 1;
        for s = 1:W
            for t = 1:M
                [sOracle(s, t), bOracle(s, t)] = initialization(maxCmWl*t/M, D, C, Rs, maxCpWl*s/W, LmOracle, LpOracle, bMin, bMax, sMin, sMax, Ce, Cs, 1, 1);
            end
        end
        dlmwrite(OracleSpeedFile, sOracle, 'delimiter', ' ', '-append');
        dlmwrite(OracleSpeedFile, bOracle, 'delimiter', ' ', '-append');
        dlmwrite(OracleSpeedFile, ' ', '-append');
        
        Lp (1:W) = 0; 
        Lm (1:M) = 1;
        [sJustDMS, bJustDMS] = initialization(maxCmWl, D, C, Rs, maxCpWl, Lm, Lp, bMin, bMax, sMin, sMax, Ce, Cs, M, W);
        dlmwrite(DMSSpeedFile, sJustDMS, 'delimiter', ' ', '-append');
        dlmwrite(DMSSpeedFile, bJustDMS, 'delimiter', ' ', '-append');
        dlmwrite(DMSSpeedFile, ' ', '-append');
        
        Lp (1:W) = 1;
        Lm (1:M) = 0;
        [sJustDVS, bJustDVS] = initialization(maxCmWl, D, C, Rs, maxCpWl, Lm, Lp, bMin, bMax, sMin, sMax, Ce, Cs, M, W);
        dlmwrite(DVSSpeedFile, sJustDVS, 'delimiter', ' ', '-append');
        dlmwrite(DVSSpeedFile, bJustDVS, 'delimiter', ' ', '-append');
        dlmwrite(DVSSpeedFile, ' ', '-append');
        
        %             Evaluate the energy using each scheme.
                
        G= uniformDistFunc(W);
        F= uniformDistFunc(M);
        
        e_cmJoint (1:M) = Rho .* (Ce + Cs .* (2 .^ bJoint(1:M) - 1)) ./ bJoint(1:M);
        e_cpJoint (1:W) = L .* (C .* sJoint(1:W) .^ 2);
        jj = sum (e_cmJoint.*F) + sum (e_cpJoint.*G);
        
        e_cmMax (1:M) = Rho .* (Ce + Cs .* (2 .^ bMax - 1)) ./ bMax;
        e_cpMax (1:W) = L .* (C .* sMax .^ 2);
        mx = sum (e_cmMax.*F) + sum (e_cpMax.*G);
        
        for s = 1:W
            for t = 1:M
                e_cmOracle (s, t) = (t) .* Rho .* (Ce + Cs .* (2 .^ bOracle(s, t) - 1)) ./ bOracle(s, t);
                e_cpOracle (s, t) = (s) .* L .* (C .* sOracle(s, t) .^ 2);
            end
        end
        
        e_cmDynamic (1:W, 1:M) = Rho .* (Ce + Cs .* (2 .^ bDynamic(1:W, 1:M) - 1)) ./ bDynamic(1:W, 1:M);
                
        e_cmJustDMS (1:M) = Rho .* (Ce + Cs .* (2 .^ bJustDMS(1:M) - 1)) ./ bJustDMS(1:M);
        e_cpJustDMS (1:W) = L .* (C .* sJustDMS(1:W) .^ 2);
        
        e_cmJustDVS (1:M) = Rho .* (Ce + Cs .* (2 .^ bJustDVS(1:M) - 1)) ./ bJustDVS(1:M);
        e_cpJustDVS (1:W) = L .* (C .* sJustDVS(1:W) .^ 2);
        
        E1 = 0; E2 = 0; E3 = 0; E4 = 0; E5 = 0; E6 = 0;
        
               
        for i=1:numberOfTasks
            %                 In terms of maxCpWl/M
%             y = pdf('norm',1:M, (1+M)/2, 1);
%             radioActualRequiredBits = randsample(1:M,1,true, y);
            radioActualRequiredBits = random('unid', M);
            %                 In terms of maxCmWl/W
%             CPUActualRequiredCycles = randsample(1:W,1,true, y);
            CPUActualRequiredCycles = random('unid', W);
            
            E1 = E1 + sum(e_cmJoint(1:radioActualRequiredBits)) + sum (e_cpJoint(1:CPUActualRequiredCycles));
            E4 = E4 + e_cmOracle(CPUActualRequiredCycles, radioActualRequiredBits) + e_cpOracle(CPUActualRequiredCycles, radioActualRequiredBits);
            E5 = E5 + sum(e_cmDynamic(CPUActualRequiredCycles, 1:radioActualRequiredBits)) + sum (e_cpJoint(1:CPUActualRequiredCycles));
            E2 = E2 + sum(e_cmJustDMS(1:radioActualRequiredBits)) + sum (e_cpJustDMS(1:CPUActualRequiredCycles));
            E3 = E3 + sum(e_cmJustDVS(1:radioActualRequiredBits)) + sum (e_cpJustDVS(1:CPUActualRequiredCycles));
            E6 = E6 + sum(e_cmMax(1:radioActualRequiredBits)) + sum (e_cpMax(1:CPUActualRequiredCycles));
        end
        EJoint (radioIndex) = E1 / numberOfTasks;
        EOracle (radioIndex) = E4 / numberOfTasks;
        EDynamic (radioIndex) = E5 / numberOfTasks;
        EJustDMS (radioIndex) = E2 / numberOfTasks;
        EJustDVS (radioIndex) = E3 / numberOfTasks;
        EMax (radioIndex) = E6 / numberOfTasks;
        
        MaxRadioWorkload = MaxRadioWorkload + 10;
    end
    dlmwrite(jointEnergyFile, EJoint, 'delimiter', ' ', '-append');
    dlmwrite(oracleEnergyFile, EOracle, 'delimiter', ' ', '-append');
    dlmwrite(dynamicEnergyFile, EDynamic, 'delimiter', ' ', '-append');
    dlmwrite(DMSEnergyFile, EJustDMS, 'delimiter', ' ', '-append');
    dlmwrite(DVSEnergyFile, EJustDVS, 'delimiter', ' ', '-append');
    dlmwrite(MaxEnergyFile, EMax, 'delimiter', ' ', '-append');
    
    jointEnergy(CPUIndex, :) = EJoint;
    oracleEnergy(CPUIndex, :) = EOracle;
    dynamicEnergy(CPUIndex, :) = EDynamic;
    DMSEnergy(CPUIndex, :) = EJustDMS;
    DVSEnergy(CPUIndex, :) = EJustDVS;
    MaxEnergy(CPUIndex, :) = EMax;
    
    
    MaxCPUWorkload = MaxCPUWorkload + 10;
end
toc
% end

