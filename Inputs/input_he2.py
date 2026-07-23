 # input
debug = False
analyze = False
nodiag = True
nstep_analysis = 10

orb =  "HF"  # HF or modpot

ne = 4
tdoc_frozen = 1
pdoc_frozen = 1

tbasis = {'Be': 'cc-pvdz' }
tgeom = "Be 0 0 0.0 ; "
talp = []  # for modpot  V = sum_i c_i * exp(-alp_i*(r-r_i)) * (r-r_i)**n_i tcoef=c_i; talp_i=alp_i, center=r_i, power=n_i
tcoef = []
tcenter = []
tpower = []
tcharge = 0
tspin = 0

pbasis = {'Be@2': 'cc-pvdz'}
elp = "Be@2"
xp = 0
yp = 0
zp = -1000.0
palp = []
pcoef = []
ppower = []
pcenter = [[0,0,zp]]
pgeom = elp + " " + str(xp) + " " + str(yp) + " " + str(zp)
pcharge = 0
pspin = 0

i_init = 11
dtime = 0.05

zmin = 3.0 # only for main_pec.py
zmax = 63.0
ngrid = 60
gridtype = 'exp'  #lin or exp
vproj = 0.2
bmin =  0.7
bmax =  8.5
nbb = 1
xmlfile = 'csfs_he2.xml'
