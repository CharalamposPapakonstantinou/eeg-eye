clear all
close all
clc;
d=importdata('eye.mat'); % https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State#

%%
t1=1;
t2=length(d);

%%
close all;
org=d(t1:t2,1:end-1);
[a,b]=hampel(org,5,10);

figure(1) 
subplot(3,1,1)
plot(d(t1:t2,end))
xlabel('Time')
ylabel('Eye Open/Close (0/1)')

subplot(3,1,2)
plot(a)
d=[a, d(:,end)];
xlabel('Time')
ylabel('Amplitude')
legend('\bf AF3','\bf F7','\bf F3','\bf FC5','\bf T7','\bf P7','\bf O1','\bf O2','\bf P8','\bf T8','\bf FC6','\bf F4','\bf F8','\bf AF4')
%%
figure,
N=7
subplot(N,1,1)
plot(d(t1:t2,end))
for i=2:N-1
subplot(N,1,i)
plot(d(t1:t2,i-1))
subplot(N,1,i+1)
plot(d(t1:t2,i))
end

%%
aa=d(:,1:end-1)-mean(d(:,1:end-1));
bb=aa./max(aa);

figure(1) 
subplot(3,1,3)
plot(bb);
xlabel('Time')
ylabel('Scaled Amplitude')


DD=[bb, d(:,end)];
[coeff,score,latent,~,explained] = pca(DD(t1:t2,1:end-1));
disp(explained)
% Xcentered = score*coeff'
 
% biplot(coeff(:,1:3),'scores',score(:,1:3),'varlabels',{'v_1','v_2','v_3','v_4'});

covarianceMatrix = cov(DD(t1:t2,1:end-1));
[V,D] = eig(covarianceMatrix);

correlation=corr(DD(t1:t2,1:end-1));

figure,
imagesc(correlation)
colormap gray
colorbar
xticks([1:14])
xticklabels({'\bf AF3','\bf F7','\bf F3','\bf FC5','\bf T7','\bf P7','\bf O1','\bf O2','\bf P8','\bf T8','\bf FC6','\bf F4','\bf F8','\bf AF4'})
yticks([1:14])
yticklabels({'\bf AF3','\bf F7','\bf F3','\bf FC5','\bf T7','\bf P7','\bf O1','\bf O2','\bf P8','\bf T8','\bf FC6','\bf F4','\bf F8','\bf AF4'})
title('Correlation')

x = repmat(1:14,14,1); % generate x-coordinates
y = x'; % generate y-coordinates
t = num2cell(round(correlation,2)); % extact values into cells
t = cellfun(@num2str, t, 'UniformOutput', false); % convert to string
text(x(:), y(:), t, 'HorizontalAlignment', 'Center')



dataInPrincipalComponentSpace = DD(t1:t2,1:end-1)*coeff; % = score

var(dataInPrincipalComponentSpace)' % eigvalues of covariance matrix = latent

figure(4)

scatter3(dataInPrincipalComponentSpace(t1:t2,1),dataInPrincipalComponentSpace(t1:t2,2),dataInPrincipalComponentSpace(t1:t2,3),[],d(t1:t2,end),'o','fill')
colormap(hsv(2))
cc=colorbar()
cc.YTick = [0.25 0.75];
cc.YTickLabel = {'Open', 'Close'};

grid on
axis square
xlabel('PC1')
ylabel('PC2')
zlabel('PC3')
title('PCA - Dimensionality Reduction')

%% gif creation

% filename='eeg_eye.gif'
% ff=figure(4);
% for tt=20:2:360
%     view(tt,30)
%     pause(0.002);
%     frame = getframe(ff);
%     im = frame2im(frame);
%     [imind,cm] = rgb2ind(im,256);
%     if tt == 20
%         imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
%     else
%         imwrite(imind,cm,filename,'gif','WriteMode','append');
%     end
% end

%%

inn=dataInPrincipalComponentSpace(t1:t2,1:3);
outn=d(t1:t2,end);

eyepca=[inn,outn];
writetable(array2table(eyepca),'eyepca.xlsx')






