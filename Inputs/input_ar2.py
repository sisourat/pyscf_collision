 # input
debug = True
analyze = False
nodiag = False
nstep_analysis = 10

orb =  "HF"  # HF or modpot

ne = 14
tdoc_frozen = 5
pdoc_frozen = 5

tbasis = {'Ar': 'aug-cc-pvdz' }
tgeom = "Ar 0 0 0.0 ; "
talp = []  # for modpot  V = sum_i c_i * exp(-alp_i*(r-r_i)) * (r-r_i)**n_i tcoef=c_i; talp_i=alp_i, center=r_i, power=n_i
tcoef = []
tcenter = []
tpower = []
tcharge = 1
tspin = 1

pbasis = {'Ar@2': 'aug-cc-pvdz'}
elp = "Ar@2"
xp = 0
yp = 0
zp = -1000.0
palp = []
pcoef = []
ppower = []
pcenter = [[0,0,zp]]
pgeom = elp + " " + str(xp) + " " + str(yp) + " " + str(zp)
pcharge = 1
pspin = 1

i_init = 11
dtime = 0.05

fpec = 'pec_cis.txt' # only for main_pec.py
zmin = 6.0 # only for main_pec.py
zmax = 16.0
ngrid = 20
gridtype = 'exp'  #lin or exp
vproj = 0.2
bmin =  0.7
bmax =  8.5
nbb = 1
xmlfile = 'csfs_ar2.xml'
