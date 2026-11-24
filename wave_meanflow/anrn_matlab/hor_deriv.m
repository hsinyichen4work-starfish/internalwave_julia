function [phix,phiy,phiz] = hor_deriv(phi,z,dx,dy)
%% Oladeji Siyanbola, USM, 08/01/2023
%% [phix,phiy,phiz] = hor_deriv(phi,z,dx,dy) computes the horizontal and vertical derivatives of phi
%% input
% phi represents the state variable
% z represents the z levels for the sigma coordinates
% dx and dy represent the grid spacings in the x and y directions, respectively
% phi, z have the same dimensions (ll,mm,nn)
% dx has dimensions [ll-1,mm,nn] while dy has dimensions [ll,mm-1,nn]

%% output
% phix and phiy are the hor. derivatives in the x and y directions, respectively
% phiz is derivative in the z direction.

nz = size(phi,3);

% vertical gradient of phi
phiz = 0*phi;

phiz(:,:,2:end-1) = (phi(:,:,3:end)-phi(:,:,1:end-2))./(z(:,:,3:end)-z(:,:,1:end-2));
phiz(:,:,1) = phiz(:,:,2) - (z(:,:,2)-z(:,:,1)).*(phiz(:,:,3)-phiz(:,:,2))./(z(:,:,3)-z(:,:,2));
phiz(:,:,nz) = phiz(:,:,nz-1) - (z(:,:,nz-1)-z(:,:,nz)).*(phiz(:,:,nz-2)-phiz(:,:,nz-1))./(z(:,:,nz-2)-z(:,:,nz-1));

% x-gradient of phi at rho points
zx = 0*z;
zx(2:end-1,:,:) = (z(3:end,:,:) - z(1:end-2,:,:))./(dx(1:end-1,:,:) + dx(2:end,:,:));
phix = 0*phi;
phix(2:end-1,:,:) = (phi(3:end,:,:) - phi(1:end-2,:,:))./(dx(1:end-1,:,:) + dx(2:end,:,:)) - ...
                            phiz(2:end-1,:,:).*zx(2:end-1,:,:);
phix(:,1,:) = 0; phix(:,end,:) = 0;                         
                        
% y-gradient of phi at rho points
zy = 0*z;
zy(:,2:end-1,:) = (z(:,3:end,:) - z(:,1:end-2,:))./(dy(:,1:end-1,:) + dy(:,2:end,:));
phiy = 0*phi;
phiy(:,2:end-1,:) = (phi(:,3:end,:) - phi(:,1:end-2,:))./(dy(:,1:end-1,:) + dy(:,2:end,:)) - ...
                            phiz(:,2:end-1,:).*zy(:,2:end-1,:);
phiy(1,:,:) = 0; phiy(end,:,:) = 0;

end

