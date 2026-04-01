 # input
debug = False
analyze = False
nodiag = False
nstep_analysis = 10

orb =  "modpot"  # HF or modpot

ne = 3
tdoc_frozen = 0

tbasis = {'N': 'aug-ccpvdz' }
tgeom = "N 0 0 0.0 ; "
talp = [0.0,2.422,2.422]  # for modpot  V = sum_i c_i * exp(-alp_i*(r-r_i)) * (r-r_i)**n_i tcoef=c_i; talp_i=alp_i, center=r_i, power=n_i
tcoef = [-5.0,-4.0,-0.461]
tcenter = [[0,0,0],[0,0,0],[0,0,0]]
tpower = [-1,-1,0]
tcharge = 0
tspin = 1

pbasis = {'H@2': 'ccpvdz'}
elp = "H@2"
xp = 0
yp = 0
zp = -1000.0
palp = [0.0]
pcoef = [-1.0]
ppower = [-1]
pcenter = [[0,0,zp]]
pgeom = elp + " " + str(xp) + " " + str(yp) + " " + str(zp)
pcharge = 0
pspin = 1

i_init = 11
dtime = 0.05

zmax = 60.0
ngrid = 100
gridtype = 'exp'  #lin or exp
vproj = 0.6
bmin =  0.5
bmax =  8.5
nbb = 16
xmlfile = 'csfs_n3+h.xml'
