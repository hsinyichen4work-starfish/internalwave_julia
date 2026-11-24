%%
%% This script computes the advection of IW Kinetic energy by background flow hor. velocities (U,V): An1
%%
%%
%%
%% grid properties

chd_dir    = '/data3/olad/';
chd_grd    = 'CCS2k_grd.nc';
chd_thetas = 6;
chd_thetab = 4.5;
chd_hc     = 350;
chd_N      = 100;
chd_ang    = 'rad';
chdscoord  = 'new2008';                 % child 'new' or 'old' type scoord
chdgrd     = [chd_dir, chd_grd];

% Child and parent s-coord parameters into chdscd and parscd
chdscd.theta_s = chd_thetas;
chdscd.theta_b = chd_thetab;
chdscd.hc      = chd_hc;
chdscd.N       = chd_N;
chdscd.scoord  = chdscoord;

% Get S-coordinate params for child grid
theta_b_c = chdscd.theta_b;
theta_s_c = chdscd.theta_s;
hc_c      = chdscd.hc;
N_c       = chdscd.N;
scoord_c  = chdscd.scoord;

% lon and lat
lon = ncread(chdgrd,'lon_rho'); [ll,mm] = size(lon);
lat = ncread(chdgrd,'lat_rho');

h = ncread(chdgrd,'h');
mask = ncread(chdgrd,'mask_rho');
h(mask == 0) = NaN;

%%
delt = 24*3;

nm = 20;tsub = 10;

%%

h1 = h(2:end-1,2:end-1);

%%

rho0 = 1027.4;

%%

fnm_amp = '/data3/olad/ROMS/output/usw_2k/March_rev/d2/uvp_amp_d2.nc'; % contains u,v modal amplitudes for baroclinic modes
fnm_amp0 = '/data3/olad/ROMS/output/usw2k_mar/d2/VEL_bar.nc'; % contains u,v modal amplitudes for barotropic mode (n = 0)

fnm_damp = '/data3/olad/ROMS/output/usw_2k/March_rev/d2/NABla_uvp_amp_d2.nc'; % contains horizontal gradients of u,v modal amplitudes for baroclinic modes 
fnm_damp0 = '/data3/olad/ROMS/output/usw2k_mar/d2/NABla_uvp_amp_d2_bar.nc'; % contains horizontal gradients of u,v modal amplitudes for barotropic mode (n = 0)

fnm_eig = '/data3/olad/ROMS/output/usw_2k/March_rev/total/eig_MAR.nc'; % contains eigen functions phi for u,v

fnm_velm = '/data3/olad/ROMS/output/usw2k_mar/total/uvw_tavg.nc'; % contains background flows (72 hours averaged u,v)

fnm_zN2 = '/data3/olad/ROMS/output/usw2k_mar/total/zN2.nc'; % contains z-levels for the sigma layer faces

fnm_out = '/data3/olad/ROMS/output/usw_2k/March_rev/d2/bal/An/An1_d2.nc'; % output file
nccreate(fnm_out,'An1','Dimensions',{'xi_rho',ll-2,'eta_rho',mm-2,'modes_m',nm+1,'modes_n',nm+1,'time',tsub},'Datatype','single')

%%

An1 = single(NaN(ll-2,mm-2,nm+1,nm+1,tsub));

for i_t = 1:tsub
    
    phi = ncread(fnm_eig,'phi',[1,1,1,1,i_t],[ll,mm,chd_N,nm+1,1]);
    phi = phi(2:end-1,2:end-1,:,:);
    
    U  = ncread(fnm_velm,'u_m',[1,1,1,i_t],[ll-1,mm,chd_N,1]);
    U = U(1:end-1,:,:)./2 + U(2:end,:,:)./2; U = U(:,2:end-1,:);
    
    V  = ncread(fnm_velm,'v_m',[1,1,1,i_t],[ll,mm-1,chd_N,1]);
    V = V(:,1:end-1,:)./2 + V(:,2:end,:)./2; V = V(2:end-1,:,:);
    
    u = ncread(fnm_amp,'u_amp',[1,1,(i_t-1)*delt+1,1],[ll-1,mm-1,delt,nm]);
    u = u(1:end-1,:,:,:)./2 + u(2:end,:,:,:)./2; u = u(:,1:end-1,:,:)./2 + u(:,2:end,:,:)./2;
    
    v = ncread(fnm_amp,'v_amp',[1,1,(i_t-1)*delt+1,1],[ll-1,mm-1,delt,nm]);
    v = v(1:end-1,:,:,:)./2 + v(2:end,:,:,:)./2; v = v(:,1:end-1,:,:)./2 + v(:,2:end,:,:)./2;
    
    u0 = ncread(fnm_amp0,'ubar',[1,1,(i_t-1)*delt+1],[ll-1,mm-1,delt]);
    u0 = u0(1:end-1,:,:)./2 + u0(2:end,:,:)./2; u0 = u0(:,1:end-1,:)./2 + u0(:,2:end,:)./2;
    
    v0 = ncread(fnm_amp0,'vbar',[1,1,(i_t-1)*delt+1],[ll-1,mm-1,delt]);
    v0 = v0(1:end-1,:,:)./2 + v0(2:end,:,:)./2; v0 = v0(:,1:end-1,:)./2 + v0(:,2:end,:)./2;
    
    u_x = ncread(fnm_damp,'u_ampx',[1,1,(i_t-1)*delt+1,1],[ll,mm,delt,nm]);
    u_x = u_x(2:end-1,2:end-1,:,:);
    
    u_y = ncread(fnm_damp,'u_ampy',[1,1,(i_t-1)*delt+1,1],[ll,mm,delt,nm]);
    u_y = u_y(2:end-1,2:end-1,:,:);
    
    v_x = ncread(fnm_damp,'v_ampx',[1,1,(i_t-1)*delt+1,1],[ll,mm,delt,nm]);
    v_x = v_x(2:end-1,2:end-1,:,:);
    
    v_y = ncread(fnm_damp,'v_ampy',[1,1,(i_t-1)*delt+1,1],[ll,mm,delt,nm]);
    v_y = v_y(2:end-1,2:end-1,:,:);
    
    u0_x = ncread(fnm_damp0,'u_ampx',[1,1,(i_t-1)*delt+1],[ll,mm,delt]);
    u0_x = u0_x(2:end-1,2:end-1,:);
    
    u0_y = ncread(fnm_damp0,'u_ampy',[1,1,(i_t-1)*delt+1],[ll,mm,delt]);
    u0_y = u0_y(2:end-1,2:end-1,:);
    
    v0_x = ncread(fnm_damp0,'v_ampx',[1,1,(i_t-1)*delt+1],[ll,mm,delt]);
    v0_x = v0_x(2:end-1,2:end-1,:);
    
    v0_y = ncread(fnm_damp0,'v_ampy',[1,1,(i_t-1)*delt+1],[ll,mm,delt]);
    v0_y = v0_y(2:end-1,2:end-1,:);
    
    zw_m = ncread(fnm_zN2,'zw_m',[1,1,1,i_t],[ll,mm,chd_N+1,1]);
    dz = zw_m(:,:,1:end-1) - zw_m(:,:,2:end); dz = dz(2:end-1,2:end-1,:);
    
    for i_n = 0:nm
        
        if i_n == 0
            un = u0;
            vn = v0;
            
        else
            
            un = u(:,:,:,i_n);
            vn = v(:,:,:,i_n);
            
        end
        
        phi_n = phi(:,:,:,i_n+1);
        
        for i_m = 0:nm
            disp([i_t,i_n,i_m])
            if i_m == 0
                
                dum_dx = u0_x;
                dum_dy = u0_y;
                dvm_dx = v0_x;
                dvm_dy = v0_y;
            else
                
                dum_dx = u_x(:,:,:,i_m);
                dum_dy = u_y(:,:,:,i_m);
                dvm_dx = v_x(:,:,:,i_m);
                dvm_dy = v_y(:,:,:,i_m);
            end
            
            phi_m = phi(:,:,:,i_m+1);
            
            
            Uphi_mn = sum(U.*phi_m.*phi_n.*dz,3); % ∫ U φm φn dz
            Vphi_mn = sum(V.*phi_m.*phi_n.*dz,3);
            
            
            An1(:,:,i_m+1,i_n+1,i_t) = rho0.*(mean(un.*dum_dx,3).*Uphi_mn) + rho0.*(mean(un.*dum_dy,3).*Vphi_mn) +...
                rho0.*(mean(vn.*dvm_dx,3).*Uphi_mn) + rho0.*(mean(vn.*dvm_dy,3).*Vphi_mn);
                % ∂uₘ/∂x = dum_dx ; un.*dum_dx = uₙ ∂ₓuₘ
        end
    end
end

An1 = -An1;

ncwrite(fnm_out,'An1',An1)


%%
i1 = 200; i2 = 600;
j1 = 400; j2 = 800;

lonc = lon(i1-1:i2-1,j1-1:j2-1);
latc = lat(i1-1:i2-1,j1-1:j2-1);
hc = h(i1-1:i2-1,j1-1:j2-1);

figure
pcolor(lonc,latc,An1(i1:i2,j1:j2,2,3,1))
shading flat
caxis([-0.02 0.02])
colormap(bluewhitered(256))

