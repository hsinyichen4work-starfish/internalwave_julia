%%
%% This script computes the rate of energy exchange between tidal flow and 
%% the varying background buoyancy, i.e., buoyancy production: Rn3
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

fnm_eig = '/data3/olad/ROMS/output/usw_2k/March_rev/total/eig_MAR.nc'; % contains eigen functions phi for u,v, PHI for w, and eigen speed Ce

fnm_zN2 = '/data3/olad/ROMS/output/usw2k_mar/total/zN2.nc'; % contains z-levels for the sigma layer faces

fnm_dvelm = '/data3/olad/ROMS/output/usw2k_mar/total/NABla_UV_mar.nc'; % contains horizontal gradients of background buoyancy  (72 hours averaged b)

fnm_amp = '/data3/olad/ROMS/output/usw_2k/March_rev/d2/uvp_amp_d2.nc'; % contains u,v,p modal amplitudes for baroclinic modes
fnm_amp0p = '/data3/olad/ROMS/output/usw2k_mar/d2/PRE_bar.nc'; % contains p modal amplitudes for barotropic mode (n = 0)
fnm_amp0v = '/data3/olad/ROMS/output/usw2k_mar/d2/VEL_bar.nc'; % contains u,v modal amplitudes for barotropic mode (n = 0)


fnm_out = '/data3/olad/ROMS/output/usw_2k/March_rev/d2/bal/Rn/Rn3_d2.nc'; % output file
nccreate(fnm_out,'Rn3','Dimensions',{'xi_rho',ll-2,'eta_rho',mm-2,'modes_m',nm+1,'modes_n',nm+1,'time',tsub},'Datatype','single')


Rn3 = single(NaN(ll-2,mm-2,nm+1,nm+1,tsub));

for i_t = 1:tsub
        
    phi = ncread(fnm_eig,'phi',[1,1,1,1,i_t],[ll,mm,chd_N,nm+1,1]);
    phi = phi(2:end-1,2:end-1,:,:);
    
    PHI = ncread(fnm_eig,'PHI',[1,1,1,1,i_t],[ll,mm,chd_N,nm+1,1]);
    PHI = PHI(2:end-1,2:end-1,:,:);
    
    ce = ncread(fnm_eig,'Ce_n',[1,1,1,i_t],[ll,mm,nm+1,1]);
    
    B_x = ncread(fnm_dvelm,'B_x',[1,1,1,i_t],[ll,mm,chd_N,1]);
    B_x = B_x(2:end-1,2:end-1,:);
    
    B_y = ncread(fnm_dvelm,'B_y',[1,1,1,i_t],[ll,mm,chd_N,1]);
    B_y = B_y(2:end-1,2:end-1,:); 
    
    p = ncread(fnm_amp,'p_amp',[1,1,(i_t-1)*delt+1,1],[ll-1,mm-1,delt,nm]);
    p = p(1:end-1,:,:,:)./2 + p(2:end,:,:,:)./2; p = p(:,1:end-1,:,:)./2 + p(:,2:end,:,:)./2;
    
    p0 = ncread(fnm_amp0p,'pbar_zeta',[1,1,(i_t-1)*delt+1],[ll-1,mm-1,delt]);        
    p0 = p0(1:end-1,:,:)./2 + p0(2:end,:,:)./2; p0 = p0(:,1:end-1,:)./2 + p0(:,2:end,:)./2;
    
    u = ncread(fnm_amp,'u_amp',[1,1,(i_t-1)*delt+1,1],[ll-1,mm-1,delt,nm]);
    u = u(1:end-1,:,:,:)./2 + u(2:end,:,:,:)./2; u = u(:,1:end-1,:,:)./2 + u(:,2:end,:,:)./2;    
    
    v = ncread(fnm_amp,'v_amp',[1,1,(i_t-1)*delt+1,1],[ll-1,mm-1,delt,nm]);
    v = v(1:end-1,:,:,:)./2 + v(2:end,:,:,:)./2; v = v(:,1:end-1,:,:)./2 + v(:,2:end,:,:)./2;
    
    u0 = ncread(fnm_amp0v,'ubar',[1,1,(i_t-1)*delt+1],[ll-1,mm-1,delt]);
    u0 = u0(1:end-1,:,:)./2 + u0(2:end,:,:)./2; u0 = u0(:,1:end-1,:)./2 + u0(:,2:end,:)./2;
    
    v0 = ncread(fnm_amp0v,'vbar',[1,1,(i_t-1)*delt+1],[ll-1,mm-1,delt]);
    v0 = v0(1:end-1,:,:)./2 + v0(2:end,:,:)./2; v0 = v0(:,1:end-1,:)./2 + v0(:,2:end,:)./2; 
           
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
                
                um = u0;                
                vm = v0;               
                             
            else        
                
                um = u(:,:,:,i_m);                
                vm = v(:,:,:,i_m);                
                               
            end 
            
            phi_m = phi(:,:,:,i_m+1);
            
                      
            Bnm1 = sum(B_x.*PHI_n.*phi_m.*dz,3); Bnm2 = sum(B_y.*PHI_n.*phi_m.*dz,3);
            
            Rn3(:,:,i_m+1,i_n+1,i_t) = mean(pn.*um,3).*Bnm1./(cn.*cn) + mean(pn.*vm,3).*Bnm2./(cn.*cn);
        end
    end
end

ncwrite(fnm_out,'Rn3',Rn3)

%%
i1 = 200; i2 = 600;
j1 = 400; j2 = 800;

lonc = lon(i1-1:i2-1,j1-1:j2-1); 
latc = lat(i1-1:i2-1,j1-1:j2-1); 
hc = h(i1-1:i2-1,j1-1:j2-1); 

ab = Rn3(i1:i2,j1:j2,2,3,1); % ab(hc < 50) = NaN;

figure
pcolor(lonc,latc,ab)
shading flat
caxis([-0.02 0.02])
colormap(bluewhitered(256))
colorbar