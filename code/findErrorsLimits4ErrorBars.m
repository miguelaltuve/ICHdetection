function [errhigh, errlow, ConfidenceIntervalOfMean] = findErrorsLimits4ErrorBars(metric)
% Computing high and low errors as the difference between the mean of the
% metric and the 95% confidence interval  
%
% Author: Miguel Altuve
% Email: miguelaltuve@gmail.com

% Fiting normal probability distribution to metric
pd = fitdist(metric,'Normal'); 

% 95% Confidence intervals for probability distribution parameters
ci = paramci(pd); 

% high error
errhigh = ci(2,1)- mean(metric); 

% low error
errlow = mean(metric)-ci(1,1); 

ConfidenceIntervalOfMean = ci(:,1);