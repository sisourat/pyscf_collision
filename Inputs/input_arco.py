 # input
debug = False
analyze = False
nodiag = False
nstep_analysis = 10

orb =  "HF"  # HF or modpot

ne = 12
tdoc_frozen = 3
pdoc_frozen = 6

tbasis = {'C': 'cc-pvdz', 'O': 'cc-pvdz', }
tgeom = "C 0 0 0.0 ; O 0 0 2.1156; "
talp = []  # for modpot  V = sum_i c_i * exp(-alp_i*(r-r_i)) * (r-r_i)**n_i tcoef=c_i; talp_i=alp_i, center=r_i, power=n_i
tcoef = []
tcenter = []
tpower = []
tcharge = 2
tspin = 0

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
pcharge = 0
pspin = 0

i_init = 11
dtime = 0.05

fpec = 'pec_arcopp_trip.txt' # only for main_pec.py
zmin = 12.0 # only for main_pec.py
zmax = 26.0
ngrid = 10
gridtype = 'exp'  #lin or exp
vproj = 0.2
bmin =  0.7
bmax =  8.5
nbb = 1
xmlfile = 'csfs_arcopp_trip.xml'
