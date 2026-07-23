from pyscf import gto
import numpy as np
import sys
import os
import pathlib
import matplotlib.pyplot as plt
import importlib

from libcollision import *
from libdyn import *
from libanalysis import *
from generate_csfs import *
from cimat import *
from scipy.interpolate import interp1d

if __name__ == "__main__":

  pdir = pathlib.Path().resolve()
  sys.path.append(pdir)
  module_name = sys.argv[1]
  module = importlib.import_module(module_name)
  # Copy all names from the module into the global namespace
  globals().update(vars(module))
  sgeom = tgeom + pgeom
  sbasis = tbasis | pbasis
  scharge = tcharge + pcharge
  sspin = tspin + pspin

  tmol = gto.M(atom=tgeom,basis=tbasis,charge=tcharge,spin=tspin,symmetry = True)
  tmol.build( unit = 'Bohr')
  pmol = gto.M(atom=pgeom,basis=pbasis,charge=pcharge,spin=pspin,symmetry = True)
  pmol.build( unit = 'Bohr')
  mol = gto.M(atom=sgeom,basis=sbasis,charge=scharge,spin=sspin)
  mol.build( unit = 'Bohr')

# Computes target and projectile HF orbitals and ""model"" orbitals
  print("TARGET")
  if orb == 'modpot':
   tmo, tmo_e = modpot_system(tmol,talp,tcoef,tcenter,tpower,debug)
  elif orb == 'HF':
   tmo, tmo_e = system(tmol,debug)
  else:
      raise NotImplementedError("Only HF or modpot orbitals implemented")
  ntmo = len(tmo)

  #print()
  #print(tmo.T)
  #print()

  print("PROJECTILE")
  if orb == 'modpot':
   pmo, pmo_e = modpot_system(pmol,palp,pcoef,pcenter,ppower,debug)
  elif orb == 'HF':
   pmo, pmo_e = system(pmol,debug)
  else:
      raise NotImplementedError("Only HF or modpot orbitals implemented")
  npmo = len(pmo)
  xp = 0
  yp = 0
  zp = -zmax
  pgeom = elp + " " + str(xp) + " " + str(yp) + " " + str(zp)
  pmol = gto.M(atom=pgeom,basis=pbasis,charge=pcharge,spin=pspin)
  pmol.build( unit = 'Bohr')

  zlist =  np.linspace(zmin,zmax,ngrid)
  nmo = ntmo + npmo

  smo = np.zeros((nmo,nmo))
  smo[0:ntmo,0:ntmo] = tmo
  smo[ntmo:nmo,ntmo:nmo] = pmo

  csfs = process_xml_csf(xmlfile)
  ncsfs = len(csfs)
  for i in range(ncsfs):
       print(i,csfs[i])

  fout = open(fpec,'w')
  for zp in zlist:
   xp = 0
   yp = 0
   pgeom = elp + " " + str(xp) + " " + str(yp) + " " + str(zp)
   sgeom = tgeom + pgeom
   mol = gto.M(atom=sgeom,basis=sbasis,charge=scharge,spin=sspin)
   mol.build( unit = 'Bohr')

   if orb == 'modpot':
    zpcenter = [np.repeat([0,0,zp], len(pcenter))]
    salp = np.concatenate((talp, palp), axis=0)
    scoef = np.concatenate((tcoef, pcoef), axis=0)
    scenter = np.concatenate((tcenter, zpcenter), axis=0)
    spower = np.concatenate((tpower, ppower), axis=0)

  # Asymptotic Energies
   phase = np.ones(ncsfs)
   if orb == 'modpot':
    ovmo, kin, pot = hcore_modpot(salp, scoef, scenter, spower, mol, smo)
   elif orb == 'HF':
    ovmo, kin, pot = hcore(mol, smo)
   else:
    raise NotImplementedError("Only HF or modpot orbitals implemented")
   r12mo = twoeints(mol,smo)
   r12mo_antisym = r12mo - r12mo.transpose(0, 2, 1, 3)
  # Sum over core orbitals
   eecore = 0.0
   enuc = 0.0
   for i in range(tdoc_frozen):
     #eecore += 2.0*r12mo[i, :, i, :]  # Sum over (p i | q i)
     eecore += 2.0*r12mo[i, i, :, :]  # Sum over (p i | q i)
     enuc += 2.0*pot[i,i]
   for i in range(ntmo,ntmo+pdoc_frozen):
     #eecore += 2.0*r12mo[i, :, i, :]  # Sum over (p i | q i)
     eecore += 2.0*r12mo[i, i, :, :]  # Sum over (p i | q i)
     enuc += 2.0*pot[i,i]

   erep_frozen = 0.0
   for i in range(tdoc_frozen):
    for j in range(ntmo,ntmo+pdoc_frozen):
       erep_frozen += 4.0*r12mo[i, i, j, j]

   h1emo = kin + pot + eecore
   mat, smat = cimat(ovmo, h1emo, r12mo, r12mo_antisym, ne, nmo, csfs, phase)
   hmat = np.linalg.inv(smat)@mat

  # Create a mask for the diagonal
   diagonal_mask = np.eye(hmat.shape[0], dtype=bool)
   if nodiag:
    # Apply the mask to keep only diagonal elements
    diagonal_hmat = hmat * diagonal_mask
    hmat = diagonal_hmat

   eig, eigv = np.linalg.eig(hmat)
   idx = eig.argsort()[::-1]
   eig = np.real(eig[idx])
   eigv = eigv[:,idx]
   print(zp,*eig+mol.energy_nuc()+enuc+erep_frozen,file=fout)

   # Get the absolute values of the eigenvectors
   abs_eigv = np.abs(eigv)
   # Find the index of the largest component for each eigenvector (column)
   largest_component_indices = np.argmax(abs_eigv, axis=0)
   # Find the values of the largest components
   largest_component_values = abs_eigv[largest_component_indices, range(eigv.shape[1])]
   for i in range(len(eig)):
     print(i, eig[i].real)
     for j in range(ncsfs):
      if(np.abs(eigv[j,i].real)>0.3):
       print("      ",j, eigv[j,i].real, csfs[j])
     print()
  fout.close()














