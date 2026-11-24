%%
%% This script computes the hor. background flow advection of IW modal energy (new term usually small): An5
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

fnm_zN2 = '/data3/olad/ROMS/output/usw2k_mar/total/zN2.nc'; % contains z-levels for the sigma layer faces and buoyancy frequency N2

fnm_eig = '/data3/olad/ROMS/output/usw_2k/March_rev/total/eig_MAR.nc'; % contains eigen functions PHI for w and eigen speed Ce
fnm_deig = '/data3/olad/ROMS/output/usw_2k/March_rev/total/NABla_eig_MAR.nc'; % contains horizontal gradients of eigen speed Ce

fnm_velm = '/data3/olad/ROMS/output/usw2k_mar/total/uvw_tavg.nc'; % contains background flows (72 hours averaged u,v)

fnm_amp = '/data3/olad/ROMS/output/usw_2k/March_rev/d2/uvp_amp_d2.nc'; % contains p modal amplitudes for baroclinic modes
fnm_amp0 = '/data3/olad/ROMS/output/usw2k_mar/d2/PRE_bar.nc'; % contains p modal amplitudes for barotropic mode (n = 0)

fnm_out = '/data3/olad/ROMS/output/usw_2k/March_rev/d2/bal/An/An5_d2.nc'; % output file
nccreate(fnm_out,'An5','Dimensions',{'xi_rho',ll-2,'eta_rho',mm-2,'modes_m',nm+1,'modes_n',nm+1,'time',tsub},'Datatype','single')

%%

An5 = single(NaN(ll-2,mm-2,nm+1,nm+1,tsub));

for i_t = 1:tsub
    
    U  = ncread(fnm_velm,'u_m',[1,1,1,i_t],[ll-1,mm,chd_N,1]);
    U = U(1:end-1,:,:)./2 + U(2:end,:,:)./2; U = U(:,2:end-1,:);
    
    V  = ncread(fnm_velm,'v_m',[1,1,1,i_t],[ll,mm-1,chd_N,1]);
    V = V(:,1:end-1,:)./2 + V(:,2:end,:)./2; V = V(2:end-1,:,:);
    
    PHI = ncread(fnm_eig,'PHI',[1,1,1,1,i_t],[ll,mm,chd_N,nm+1,1]);
    PHI = PHI(2:end-1,2:end-1,:,:);
        
    N2 = ncread(fnm_zN2,'N2',[1,1,1,i_t],[ll,mm,chd_N+1,1]);
    N2 = N2(:,:,1:end-1)./2 + N2(:,:,2:end)./2;
    N2 = N2(2:end-1,2:end-1,:);
    
    ce = ncread(fnm_eig,'Ce_n',[1,1,1,i_t],[ll,mm,nm+1,1]);
    
    ce_x = ncread(fnm_deig,'Ce_nx',[1,1,1,i_t],[ll,mm,nm+1,1]);
        
    ce_y = ncread(fnm_deig,'Ce_ny',[1,1,1,i_t],[ll,mm,nm+1,1]);
        
    p = ncread(fnm_amp,'p_amp',[1,1,(i_t-1)*delt+1,1],[ll-1,mm-1,delt,nm]);
    p = p(1:end-1,:,:,:)./2 + p(2:end,:,:,:)./2; p = p(:,1:end-1,:,:)./2 + p(:,2:end,:,:)./2;
    
    p0 = ncread(fnm_amp0,'pbar_zeta',[1,1,(i_t-1)*delt+1],[ll-1,mm-1,delt]);        
    p0 = p0(1:end-1,:,:)./2 + p0(2:end,:,:)./2; p0 = p0(:,1:end-1,:)./2 + p0(:,2:end,:)./2;
    
    zw_m = ncread(fnm_zN2,'zw_m',[1,1,1,i_t],[ll,mm,chd_N+1,1]);
    dz = zw_m(:,:,1:end-1) - zw_m(:,:,2:end); dz = dz(2:end-1,2:end-1,:);
    
    for i_n = 0:nm       
                
        if i_n == 0
            
            pn = p0;
        else
            
            pn = p(:,:,:,i_n);
            
        end
        
        PHI_n = PHI(:,:,:,i_n+1);
        
        cn = ce(2:end-1,2:end-1,i_n+1);
                
        for i_m = 0:nm
            disp([i_t,i_n,i_m])
            
            if i_m == 0
                
                pm = p0;
                
            else
                
                pm = p(:,:,:,i_m);
                
            end
            
            PHI_m = PHI(:,:,:,i_m+1);
            
            cm = ce(2:end-1,2:end-1,i_m+1);            
                                                          
            dcm_dx = ce_x(2:end-1,2:end-1,i_m+1);
            dcm_dy = ce_y(2:end-1,2:end-1,i_m+1);              
            
            
            UPHI_mn = sum(U.*N2.*PHI_m.*PHI_n.*dz,3)./(cm.*cm); VPHI_mn = sum(V.*N2.*PHI_m.*PHI_n.*dz,3)./(cm.*cm);
            
            
            An5(:,:,i_m+1,i_n+1,i_t) = (2.*mean(pn.*pm,3).*dcm_dx.*UPHI_mn)./(rho0.*cn.*cn.*cm) + ...
                (2.*mean(pn.*pm,3).*dcm_dy.*VPHI_mn)./(rho0.*cn.*cn.*cm);
        end
    end
end

ncwrite(fnm_out,'An5',An5)


%%
i1 = 200; i2 = 600;
j1 = 400; j2 = 800;

lonc = lon(i1-1:i2-1,j1-1:j2-1); 
latc = lat(i1-1:i2-1,j1-1:j2-1); 
hc = h(i1-1:i2-1,j1-1:j2-1); 

ab = An5(i1:i2,j1:j2,2,3,5); ab(hc < 50) = NaN;

figure
pcolor(lonc,latc,ab)
shading flat
caxis([-0.02 0.02])
colormap(bluewhitered(256))
colorbar

