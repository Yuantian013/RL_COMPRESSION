a=abs(mu_com)<1e-2;
b=sum(sum(a))
c=b/200
%%
figure(1);imagesc(Actor_a_kernel);figure(2);imagesc(Actor_a_kernel_com)
%%
plot(Actor_a_kernel,'r');hold on;plot(Actor_a_kernel_com,'b');legend('Second layer kernel','Compressed second layer kernel')