#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
uncert.py
---------

'''

from __future__ import division, print_function, absolute_import, unicode_literals
import vplot as vpl
import matplotlib
import matplotlib.pyplot as pl
import numpy as np
import emcee
import subprocess
import re
import os, sys
import random
from pool import Pool
from scipy.stats import norm, lognorm
import corner
import argparse
PATH = os.path.dirname(os.path.abspath(__file__))

# From Ribas et al. (2016), discarding 
# the flare corrections.
# F   lambda(nm)
# --------------
# 130 [0.6 - 10]
# 89  [10 - 40]
# 13  [40 - 92]
# 20  [92 - 118]
# --------------
# FXUV = 252 erg/s/cm^2
# LXUV = 1.67e27 erg/s = 4.34e-7 LSUN
# log(LXUV / LSUN) = -6.36
# log( sig LXUV / LSUN) = 0.3 (Arbitrary, but roughly based on observational spred of X-ray values)

# Constants
LSUN = 3.846e26
YEARSEC = 3.15576e7
BIGG = 6.67428e-11
DAYSEC = 86400
MSUN = 1.988416e30
AUCM = 1.49598e11

# Observational constraints    
M = 0.120                 # Anglada-Escude et al. (2016)
sigM = 0.015              # Anglada-Escude et al. (2016) 
L = 0.00165               # Demory et al. (2009)
sigL = 0.00015            # Demory et al. (2009)
logLXUV = -6.36           # Ribas et al. (2016; see above)
siglogLXUV = 0.3          # See above
P = 11.186                # Anglada-Escude et al. (2016)
sigP = 0.002              # Anglada-Escude et al. (2016)
Mpsini = 1.27             # Anglada-Escude et al. (2016)
sigMpsini = 0.18          # Anglada-Escude et al. (2016)
Age = 4.8                 # Barnes et al. (2017)
sigAge = 1.4              # TODO: should be skewed normal, sigma- = 1.4, sigma+ = 1.1
Beta = -1.23              # Ribas et al. (2005) 
sigBeta = 0.1             # Arbitrary

class FunctionWrapper(object):
  '''
  A simple function wrapper class. Stores :py:obj:`args` and :py:obj:`kwargs` and
  allows an arbitrary function to be called with a single parameter :py:obj:`x`
  
  '''
  
  def __init__(self, f, *args, **kwargs):
    '''
    
    '''
    
    self.f = f
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, x):
    '''
    
    '''
    
    return self.f(x, *self.args, **self.kwargs)

def LnPrior(x, **kwargs):
  '''
  
  '''
  
  # Get the current vector
  dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = x
  
  # Log-flat prior on fsat
  dSatXUVFrac = 10 ** dSatXUVFrac
  
  # Gyr -> years
  dStopTime *= 1.e9
  
  # Reduce the amount of text output
  dOutputTime = dStopTime
  
  # Log-flat prior on tsat
  dSatXUVTime = 10 ** dSatXUVTime
  
  # This fixes the units in vplanet
  dSatXUVTime *= -1

  # Generous bounds
  if (dMass < 0.1) or (dMass > 0.15):
    return -np.inf
  elif (dSatXUVFrac < 1.e-5) or (dSatXUVFrac > 1.e-2):
    return -np.inf
  elif (-dSatXUVTime < 0.5) or (-dSatXUVTime > 10.):
    return -np.inf
  elif (dStopTime < 1e6) or (dStopTime > 1.e10):
    return -np.inf
  
  # Age prior.
  lnprior = norm.logpdf(dStopTime, Age * 1e9, sigAge * 1e9)
  
  # Beta prior.
  lnprior += norm.logpdf(dXUVBeta, Beta, sigBeta)
  
  return lnprior

def LnLike(x, **kwargs):
  '''
  
  '''
  
  # Get the current vector
  dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = x
  dSatXUVFrac = 10 ** dSatXUVFrac
  dStopTime *= 1.e9
  dOutputTime = dStopTime
  dSatXUVTime = 10 ** dSatXUVTime
  dSatXUVTime *= -1
  
  # Get the period (all prior); convert to semi-major axis
  dOrbPeriod = (P + sigP * np.random.randn()) * DAYSEC
  dSemi = (dOrbPeriod ** 2 * BIGG * (dMass * MSUN) / (4 * np.pi ** 2)) ** (1./3.) / AUCM
  
  # Get the planet mass (all prior)
  dPlanetMass = -np.inf
  while -dPlanetMass > 10:
    inc = np.arccos(1 - np.random.random())
    msini = Mpsini + sigMpsini * np.random.randn()
    dPlanetMass = -msini / np.sin(inc)
  
  # Get the prior probability
  lnprior = LnPrior(x, **kwargs)
  if np.isinf(lnprior):
    return -np.inf, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
  
  # Get the data
  star_in = kwargs.get('star_in')
  planet_in = kwargs.get('planet_in')
  vpl_in = kwargs.get('vpl_in')
  L = kwargs.get('L')
  sigL = kwargs.get('sigL')
  logLXUV = kwargs.get('logLXUV')
  siglogLXUV = kwargs.get('siglogLXUV')
  InitH2O = kwargs.get('InitH2O')
  InitH = kwargs.get('InitH')
  EpsH2O = kwargs.get('EpsH2O')
  EpsH = kwargs.get('EpsH')
  InstantO2Sink = kwargs.get('InstantO2Sink')
  
  # Randomize file names
  sysname = 'vpl.%015x' % random.randrange(16**15)
  starname = 'star.%015x' % random.randrange(16**15)
  planetname = 'planet.%015x' % random.randrange(16**15)
  sysfile = sysname + '.in'
  starfile = starname + '.in'
  planetfile = planetname + '.in'
  logfile = sysname + '.log'
  starfwfile = '%s.star.forward' % sysname
  planetfwfile = '%s.planet.forward' % sysname
  
  # Populate the star input file
  for param in ['dMass', 'dSatXUVFrac', 'dSatXUVTime']:
    star_in = re.sub('%s(.*?)#' % param, '%s %.5e #' % (param, eval(param)), star_in)
  with open(os.path.join(PATH, 'output', starfile), 'w') as f:
    print(star_in, file = f)
  
  # Populate the planet input file
  planet_in = re.sub('%s(.*?)#' % 'dSemi', '%s %.5e #' % ('dSemi', dSemi), planet_in)
  planet_in = re.sub('%s(.*?)#' % 'dMass', '%s %.5e #' % ('dMass', dPlanetMass), planet_in)
  planet_in = re.sub('%s(.*?)#' % 'dSurfWaterMass', '%s %.5e #' % ('dSurfWaterMass', InitH2O), planet_in)
  planet_in = re.sub('%s(.*?)#' % 'dEnvelopeMass', '%s %.5e #' % ('dEnvelopeMass', InitH), planet_in)
  planet_in = re.sub('%s(.*?)#' % 'dAtmXAbsEffH2O', '%s %.5e #' % ('dAtmXAbsEffH2O', EpsH2O), planet_in)
  planet_in = re.sub('%s\s(.*?)#' % 'dAtmXAbsEffH', '%s %.5e #' % ('dAtmXAbsEffH', EpsH), planet_in)
  planet_in = re.sub('%s\s(.*?)#' % 'bInstantO2Sink', '%s %d #' % ('bInstantO2Sink', InstantO2Sink), planet_in)
  with open(os.path.join(PATH, 'output', planetfile), 'w') as f:
    print(planet_in, file = f)
  
  # Populate the system input file
  for param in ['dStopTime', 'dOutputTime']:
    vpl_in = re.sub('%s(.*?)#' % param, '%s %.5e #' % (param, eval(param)), vpl_in)
  vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysname, vpl_in)
  vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s %s #' % (starfile, planetfile), vpl_in)
  with open(os.path.join(PATH, 'output', sysfile), 'w') as f:
    print(vpl_in, file = f)
  
  # Run VPLANET and get the output, then delete the logfile
  subprocess.call(['vplanet', sysfile], cwd = os.path.join(PATH, 'output'))
  output = vpl.GetOutput(os.path.join(PATH, 'output'), logfile = logfile)
  
  try:
    os.remove(os.path.join(PATH, 'output', starfile))
    os.remove(os.path.join(PATH, 'output', planetfile))
    os.remove(os.path.join(PATH, 'output', sysfile))
    os.remove(os.path.join(PATH, 'output', starfwfile))
    os.remove(os.path.join(PATH, 'output', planetfwfile))
    os.remove(os.path.join(PATH, 'output', logfile))
  except FileNotFoundError:
    # Run failed!
    return -np.inf, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

  # Ensure we ran for as long as we set out to
  if not output.log.final.system.Age / YEARSEC >= dStopTime:
    return -np.inf, [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
  
  dEnvMass = float(output.log.final.planet.EnvelopeMass)
  dWaterMass = float(output.log.final.planet.SurfWaterMass)
  dOxygenMass = float(output.log.final.planet.OxygenMass)
  dLuminosity = float(output.log.final.star.Luminosity)
  dLXUV = float(output.log.final.star.LXUVStellar)
  dRGTime = float(output.log.final.planet.RGDuration)
  
  # Compute the likelihood
  lnlike = -0.5 * ((dLuminosity - L) / sigL) ** 2 + \
           -0.5 * ((np.log10(dLXUV) - logLXUV) / siglogLXUV) ** 2 + \
           lnprior

  # Return likelihood and blobs
  return lnlike, np.array([dSemi, -dPlanetMass, dLuminosity, dLXUV, dRGTime, dEnvMass, dWaterMass, dOxygenMass])

def GetEvol(x, **kwargs):
  '''
  
  '''

  # Get the current vector
  dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = x
  dSatXUVFrac = 10 ** dSatXUVFrac
  dStopTime *= 1.e9
  dOutputTime = 1.e6
  dSatXUVTime = 10 ** dSatXUVTime
  dSatXUVTime *= -1
  
  # Get the period (all prior); convert to semi-major axis
  dOrbPeriod = (P + sigP * np.random.randn()) * DAYSEC
  dSemi = (dOrbPeriod ** 2 * BIGG * (dMass * MSUN) / (4 * np.pi ** 2)) ** (1./3.) / AUCM
  
  # Get the planet mass (all prior)
  dPlanetMass = -np.inf
  while -dPlanetMass > 10:
    inc = np.arccos(1 - np.random.random())
    msini = Mpsini + sigMpsini * np.random.randn()
    dPlanetMass = -msini / np.sin(inc)
  
  # Get the data
  star_in = kwargs.get('star_in')
  planet_in = kwargs.get('planet_in')
  vpl_in = kwargs.get('vpl_in')
  L = kwargs.get('L')
  sigL = kwargs.get('sigL')
  logLXUV = kwargs.get('logLXUV')
  siglogLXUV = kwargs.get('siglogLXUV')
  InitH2O = kwargs.get('InitH2O')
  InitH = kwargs.get('InitH')
  EpsH2O = kwargs.get('EpsH2O')
  EpsH = kwargs.get('EpsH')
  InstantO2Sink = kwargs.get('InstantO2Sink')
  
  # Randomize file names
  sysname = 'vpl.%015x' % random.randrange(16**15)
  starname = 'star.%015x' % random.randrange(16**15)
  planetname = 'planet.%015x' % random.randrange(16**15)
  sysfile = sysname + '.in'
  starfile = starname + '.in'
  planetfile = planetname + '.in'
  logfile = sysname + '.log'
  starfwfile = '%s.star.forward' % sysname
  planetfwfile = '%s.planet.forward' % sysname
  
  # Populate the star input file
  for param in ['dMass', 'dSatXUVFrac', 'dSatXUVTime']:
    star_in = re.sub('%s(.*?)#' % param, '%s %.5e #' % (param, eval(param)), star_in)
  with open(os.path.join(PATH, 'output', starfile), 'w') as f:
    print(star_in, file = f)
  
  # Populate the planet input file
  planet_in = re.sub('%s(.*?)#' % 'dSemi', '%s %.5e #' % ('dSemi', dSemi), planet_in)
  planet_in = re.sub('%s(.*?)#' % 'dMass', '%s %.5e #' % ('dMass', dPlanetMass), planet_in)
  planet_in = re.sub('%s(.*?)#' % 'dSurfWaterMass', '%s %.5e #' % ('dSurfWaterMass', InitH2O), planet_in)
  planet_in = re.sub('%s(.*?)#' % 'dEnvelopeMass', '%s %.5e #' % ('dEnvelopeMass', InitH), planet_in)
  planet_in = re.sub('%s(.*?)#' % 'dAtmXAbsEffH2O', '%s %.5e #' % ('dAtmXAbsEffH2O', EpsH2O), planet_in)
  planet_in = re.sub('%s\s(.*?)#' % 'dAtmXAbsEffH', '%s %.5e #' % ('dAtmXAbsEffH', EpsH), planet_in)
  planet_in = re.sub('%s\s(.*?)#' % 'bInstantO2Sink', '%s %d #' % ('bInstantO2Sink', InstantO2Sink), planet_in)
  with open(os.path.join(PATH, 'output', planetfile), 'w') as f:
    print(planet_in, file = f)
  
  # Populate the system input file
  for param in ['dStopTime', 'dOutputTime']:
    vpl_in = re.sub('%s(.*?)#' % param, '%s %.5e #' % (param, eval(param)), vpl_in)
  vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysname, vpl_in)
  vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s %s #' % (starfile, planetfile), vpl_in)
  with open(os.path.join(PATH, 'output', sysfile), 'w') as f:
    print(vpl_in, file = f)
  
  # Run VPLANET and get the output, then delete the logfile
  subprocess.call(['vplanet', sysfile], cwd = os.path.join(PATH, 'output'))
  output = vpl.GetOutput(os.path.join(PATH, 'output'), logfile = logfile)

  try:
    os.remove(os.path.join(PATH, 'output', starfile))
    os.remove(os.path.join(PATH, 'output', planetfile))
    os.remove(os.path.join(PATH, 'output', sysfile))
    os.remove(os.path.join(PATH, 'output', starfwfile))
    os.remove(os.path.join(PATH, 'output', planetfwfile))
    os.remove(os.path.join(PATH, 'output', logfile))
  except FileNotFoundError:
    # Run failed!
    return None

  return output

def RunMCMC(oceans = 10, hydrogen = 0, epswater = 0.15, epshydro = 0.15,
            nwalk = 16, nsteps = 5000, nburn = 500, name = 'test', magma = False,
            pool = None, **kwargs):
  '''
  
  '''
  
  print("Running MCMC...")
  
  # Get the input files
  with open(os.path.join(PATH, 'star.in'), 'r') as f:
    star_in = f.read()
  with open(os.path.join(PATH, 'planet.in'), 'r') as f:
    planet_in = f.read()
  with open(os.path.join(PATH, 'vpl.in'), 'r') as f:
    vpl_in = f.read()  

  # MCMC kwargs
  kwargs = {'star_in': star_in, 'planet_in': planet_in, 'vpl_in': vpl_in,
            'L': L, 'sigL': sigL, 'logLXUV': logLXUV, 'siglogLXUV': siglogLXUV,
            'InitH2O': -oceans, 'InitH': -hydrogen, 'EpsH2O': epswater, 'EpsH': epshydro,
            'InstantO2Sink': magma}

  # Set up MCMC
  ndim = 5
  x0 = np.array([0.123, -3, 0.1, Age, Beta])
  _, blobs0 = LnLike(x0, **kwargs)
  def get_guess():
    while True:
      guess = [0.123 + 0.005 * np.random.randn(),
               -3. + 0.1 * np.random.randn(),
               0.25 + 0.1 * np.random.randn(),
               Age + sigAge * np.random.randn(),
               Beta + sigBeta * np.random.randn()]
      if not np.isinf(LnPrior(guess, **kwargs)):
        return guess
  x0 = np.array([get_guess() for w in range(nwalk)])

  # Run MCMC
  sampler = emcee.EnsembleSampler(nwalk, ndim, LnLike, kwargs = kwargs, pool = pool)
  width = 50
  for i, result in enumerate(sampler.sample(x0, blobs0 = blobs0, iterations = nsteps)):
    print("MCMC: %d/%d..." % (i + 1, nsteps))
  
  # Save
  blobs = np.array(sampler.blobs)
  chain = np.array(sampler.chain)
  np.savez(os.path.join(PATH, 'output', '%s.mcmc.npz' % name), blobs = blobs, chain = chain, nwalk = nwalk, 
           nburn = nburn, nsteps = nsteps, name = name, oceans = oceans,
           hydrogen = hydrogen, epswater = epswater, epshydro = epshydro, magma = magma)

def PlotChains(name = 'test', **kwargs):
  '''
  
  '''
  
  print("Plotting chains...")
  
  # Load
  data = np.load(os.path.join(PATH, 'output', '%s.mcmc.npz' % name))
  blobs = data['blobs']
  chain = data['chain']
  nwalk = data['nwalk']
  nburn = kwargs.get('nburn', data['nburn'])
  ndim = chain.shape[-1]
  
  # Plot the chains
  fig = pl.figure(figsize = (16, 10))
  fig.subplots_adjust(bottom = 0.05, top = 0.95, hspace = 0.1)
  axc = [pl.subplot2grid((6, 22), (0, 0), colspan = 8, rowspan = 1),
         pl.subplot2grid((6, 22), (1, 0), colspan = 8, rowspan = 1),
         pl.subplot2grid((6, 22), (2, 0), colspan = 8, rowspan = 1),
         pl.subplot2grid((6, 22), (3, 0), colspan = 8, rowspan = 1),
         pl.subplot2grid((6, 22), (4, 0), colspan = 8, rowspan = 1),
         pl.subplot2grid((6, 22), (5, 0), colspan = 8, rowspan = 1),
         pl.subplot2grid((6, 22), (0, 12), colspan = 8, rowspan = 1),
         pl.subplot2grid((6, 22), (1, 12), colspan = 8, rowspan = 1),
         pl.subplot2grid((6, 22), (2, 12), colspan = 8, rowspan = 1),
         pl.subplot2grid((6, 22), (3, 12), colspan = 8, rowspan = 1),
         pl.subplot2grid((6, 22), (4, 12), colspan = 8, rowspan = 1),
         pl.subplot2grid((6, 22), (5, 12), colspan = 8, rowspan = 1)]
  axh = [pl.subplot2grid((6, 22), (0, 8), colspan = 2, rowspan = 1, sharey = axc[0]),
         pl.subplot2grid((6, 22), (1, 8), colspan = 2, rowspan = 1, sharey = axc[1]),
         pl.subplot2grid((6, 22), (2, 8), colspan = 2, rowspan = 1, sharey = axc[2]),
         pl.subplot2grid((6, 22), (3, 8), colspan = 2, rowspan = 1, sharey = axc[3]),
         pl.subplot2grid((6, 22), (4, 8), colspan = 2, rowspan = 1, sharey = axc[4]),
         pl.subplot2grid((6, 22), (5, 8), colspan = 2, rowspan = 1, sharey = axc[5]),
         pl.subplot2grid((6, 22), (0, 20), colspan = 2, rowspan = 1, sharey = axc[6]),
         pl.subplot2grid((6, 22), (1, 20), colspan = 2, rowspan = 1, sharey = axc[7]),
         pl.subplot2grid((6, 22), (2, 20), colspan = 2, rowspan = 1, sharey = axc[8]),
         pl.subplot2grid((6, 22), (3, 20), colspan = 2, rowspan = 1, sharey = axc[9]),
         pl.subplot2grid((6, 22), (4, 20), colspan = 2, rowspan = 1, sharey = axc[10]),
         pl.subplot2grid((6, 22), (5, 20), colspan = 2, rowspan = 1, sharey = axc[11])]
         
  # Plot!
  params = [r'$M_\star$ (M$_\odot$)', r'$\log\ f_{sat}$', r'$\log\ t_{sat}$ (Gyr)', 'Age (Gyr)']
  for p, param in enumerate(params):  
    for k in range(nwalk):
      axc[p].plot(chain[k,:,p],alpha = 0.2,lw=1)
    axc[p].set_ylabel(param)
    axh[p].hist(chain[:,nburn:,p].flatten(), bins=30, 
                orientation="horizontal", histtype='step', 
                fill=False, color = 'k', lw = 1)
    pl.setp(axh[p].get_yticklabels(), visible=False)
  
  # Take the log of the LXUV for plotting
  blobs[:,:,3] = np.log10(blobs[:,:,3])
  
  # Plot the model outputs
  params = [r'a ($10^{-2}$ AU)', r'$m$ (M$_\oplus$)', r'$L_\star$ ($10^{-3}$ L$_\odot$)', r'$\log\ L_{xuv}$ (L$_\odot$)', r'RG Dur. (Gyr)', r'Hydrogen (M$_\oplus$)', 'Water (TO)', 'Oxygen (bar)']
  scale = [1e2, 1, 1e3, 1, 1e-9, 1, 1, 1]
  for p, param in enumerate(params):
    for k in range(nwalk):
      axc[ndim + p - 1].plot(blobs[:,k,p] * scale[p], alpha = 0.2, lw = 1)
    axc[ndim + p - 1].set_ylabel(param)
    values = blobs[nburn:,:,p].flatten() * scale[p]
    values = np.delete(values, np.where(np.isnan(values)))
    axh[ndim + p - 1].hist(values, bins=30, 
                           orientation="horizontal", histtype='step', 
                           fill=False, color = 'k', lw = 1)
    pl.setp(axh[ndim + p - 1].get_yticklabels(), visible=False)
    
  # Plot the age prior
  x = np.linspace(axh[3].get_ylim()[0], axh[3].get_ylim()[1], 1000)
  y = norm.pdf(x, Age, sigAge)
  y *= (axh[3].get_xlim()[1] - axh[3].get_xlim()[0]) / (y.max() - y.min())
  axh[3].plot(y, x, 'r-', lw = 2)
  
  # Plot the planet mass prior
  x = np.linspace(axh[5].get_ylim()[0], axh[5].get_ylim()[1], 300)
  inc = np.arccos(1 - np.random.random(1000000))
  msini = Mpsini + sigMpsini * np.random.randn(1000000)
  m = msini / np.sin(inc)
  y = np.array(np.histogram(m, x)[0], dtype = float)
  y *= (axh[5].get_xlim()[1] - axh[5].get_xlim()[0]) / (y.max() - y.min())
  axh[5].plot(y, x[:-1], 'r-', lw = 2)
  
  # Plot the luminosity constraint
  x = np.linspace(axh[6].get_ylim()[0], axh[6].get_ylim()[1], 1000)
  y = norm.pdf(x, 1e3 * L, 1e3 * sigL)
  y *= (axh[6].get_xlim()[1] - axh[6].get_xlim()[0]) / (y.max() - y.min())
  axh[6].plot(y, x, 'b-', lw = 2)

  # Plot the lxuv constraint
  x = np.linspace(axh[7].get_ylim()[0], axh[7].get_ylim()[1], 1000)
  y = norm.pdf(x, logLXUV, siglogLXUV)
  y *= (axh[7].get_xlim()[1] - axh[7].get_xlim()[0]) / (y.max() - y.min())
  axh[7].plot(y, x, 'b-', lw = 2)

  # Appearance
  for i, axis in enumerate(axc):
    axis.margins(0, 0)
    axis.set_xticklabels([])
    axis.get_yaxis().set_label_coords(-0.12, 0.5)
    axis.axvline(nburn, color = 'b', ls = '--', alpha = 0.5, lw = 2)
  for i, axis in enumerate(axh):
    axis.set_xticklabels([])
    axis.set_xlim(0, axis.get_xlim()[1] * 1.1)
  
  fig.savefig(os.path.join(PATH, 'output', '%s.chains.pdf' % name), bbox_inches = 'tight')

def PlotCorner(name = 'test', **kwargs):
  '''
  
  '''
  
  print("Plotting corner plot...")
  
  # Load
  data = np.load(os.path.join(PATH, 'output', '%s.mcmc.npz' % name))
  blobs = data['blobs']
  chain = data['chain']
  nwalk = data['nwalk']
  nburn = kwargs.get('nburn', data['nburn'])
  ndim = chain.shape[-1]
  nsteps = chain.shape[1]
  
  # Is there any hydrogen, anywhere?
  if blobs[:,:,-3].max() > blobs[:,:,-3].min():
    labels=[r"RG Dur. (Myr)", r"Hydrogen (M$_\oplus$)", r"Water (TO)", r"Oxygen (bar)"]
    blobs = blobs[nburn:,:,np.array([-4,-3,-2,-1])].reshape(nwalk * (nsteps - nburn), 4)
  else:
    labels=[r"RG Dur. (Myr)", r"Water (TO)", r"Oxygen (bar)"]
    blobs = blobs[nburn:,:,np.array([-4,-2,-1])].reshape(nwalk * (nsteps - nburn), 3)

  # Fix units
  blobs[:,0] /= 1.e6
  
  # Plot
  matplotlib.rcParams['lines.linewidth'] = 1
  fig = corner.corner(blobs, labels = labels, bins = 50)
  fig.savefig(os.path.join(PATH, 'output', '%s.corner.pdf' % name), bbox_inches = 'tight')

def RunEvol(name = 'test', nsamples = 100, pool = None, **kwargs):
  '''
  
  '''
  
  print("Running evolution...")
  
  # Load
  data = np.load(os.path.join(PATH, 'output', '%s.mcmc.npz' % name))
  blobs = data['blobs']
  chain = data['chain']
  nwalk = data['nwalk']
  nburn = kwargs.get('nburn', data['nburn'])
  ndim = chain.shape[-1]
  nsteps = data['nsteps']
  oceans = data['oceans']
  hydrogen = data['hydrogen']
  epswater = data['epswater']
  epshydro = data['epshydro']
  magma = data['magma']
  name = data['name']

  # Get kwargs
  with open('star.in', 'r') as f:
    star_in = f.read()
  with open('planet.in', 'r') as f:
    planet_in = f.read()
  with open('vpl.in', 'r') as f:
    vpl_in = f.read()
  kwargs = {'star_in': star_in, 'planet_in': planet_in, 'vpl_in': vpl_in,
            'L': L, 'sigL': sigL, 'logLXUV': logLXUV, 'siglogLXUV': siglogLXUV,
            'InitH2O': -oceans, 'InitH': -hydrogen, 'EpsH2O': epswater, 
            'EpsH': epshydro, 'InstantO2Sink': magma}

  # Flatten
  chain = chain[:,nburn:,:].reshape(nwalk * (nsteps - nburn), -1)

  # Run!
  func = FunctionWrapper(GetEvol, **kwargs)
  x = [chain[np.random.randint(0, chain.shape[0])] for i in range(nsamples)]
  if pool is None:
    outputs = list(map(func, x))
  else:
    outputs = pool.map(func, x)
  
  # Save
  np.savez(os.path.join(PATH, 'output', '%s.evol.npz' % name), 
           Time = [o.star.Time for o in outputs],
           Luminosity = [o.star.Luminosity for o in outputs],
           LXUVStellar = [o.star.LXUVStellar for o in outputs],
           EnvelopeMass = [o.planet.EnvelopeMass for o in outputs],
           SurfWaterMass = [o.planet.SurfWaterMass for o in outputs],
           OxygenMass = [o.planet.OxygenMass for o in outputs])
    
def PlotEvol(name = 'test', **kwargs):
  '''
  
  '''
  
  print("Plotting evolution...")
  
  # Load
  data = np.load(os.path.join(PATH, 'output', '%s.evol.npz' % name))
  Time = data['Time']
  Luminosity = data['Luminosity']
  LXUVStellar = data['LXUVStellar']
  EnvelopeMass = data['EnvelopeMass']
  SurfWaterMass = data['SurfWaterMass']
  OxygenMass = data['OxygenMass']
  
  # Plot
  fig_star, ax_star = pl.subplots(1,2, figsize = (10, 4))
  fig_star.subplots_adjust(bottom = 0.15, wspace = 0.3)
  fig_planet, ax_planet = pl.subplots(1,3, figsize = (16, 4))
  fig_planet.subplots_adjust(bottom = 0.15, wspace = 0.3)
  for i in range(len(Time)):
    ax_star[0].plot(Time[i], Luminosity[i], lw = 1, alpha = 0.2)
    ax_star[1].plot(Time[i], LXUVStellar[i], lw = 1, alpha = 0.2)
    ax_planet[0].plot(Time[i], EnvelopeMass[i], lw = 1, alpha = 0.2)
    ax_planet[1].plot(Time[i], SurfWaterMass[i], lw = 1, alpha = 0.2)
    ax_planet[2].plot(Time[i], OxygenMass[i], lw = 1, alpha = 0.2)
  
  # Appearance
  for axis in np.append(ax_star, ax_planet):
    axis.set_xscale('log')
    axis.set_xlabel('Time (yr)')
  for axis in ax_star:
    axis.set_yscale('log')
  ax_star[0].set_ylabel(r'$L_\star$ (L$_\odot$)')
  ax_star[1].set_ylabel(r'$L_{xuv}$ (L$_\odot$)')
  ax_planet[0].set_ylabel(r'Hydrogen (M$_\oplus$)')
  ax_planet[1].set_ylabel('Water (TO)')
  ax_planet[2].set_ylabel('Oxygen (bar)')

  # Save
  fig_star.savefig(os.path.join(PATH, 'output', '%s.star.pdf' % name), bbox_inches = 'tight')
  fig_planet.savefig(os.path.join(PATH, 'output', '%s.planet.pdf' % name), bbox_inches = 'tight')

def Submit(name = 'test', nsteps = 5000, nwalk = 40, nburn = 500, nsamples = 1000,
           oceans = 10., hydrogen = 0.01, epsilon = 0.15, magma = False, 
           walltime = 10, nodes = 2, ppn = 16, mpn = 250):
  '''
  
  '''
  
  # Submit the cluster job      
  pbsfile = os.path.join(PATH, 'uncert.pbs')
  str_w = 'walltime=%d:00:00' % walltime
  str_v = 'NAME=%s,NSTEPS=%d,NWALK=%d,NBURN=%d,NSAMPLES=%d,OCEANS=%.5f,HYDROGEN=%.5f,EPSILON=%.5f,MAGMA=%d' % \
          (name, nsteps, nwalk, nburn, nsamples, oceans, hydrogen, epsilon, magma)
  str_n = 'nodes=%d:ppn=%d,feature=%dcore,mem=%dgb' % (nodes, ppn, ppn, mpn * nodes)
  str_name = name
  str_out = os.path.join(PATH, 'output', str_name + '.log')
  qsub_args = ['qsub', pbsfile,
               '-v', str_v, 
               '-o', str_out,
               '-j', 'oe', 
               '-N', str_name,
               '-l', str_n,
               '-l', str_w]
  print("Submitting the job...")
  subprocess.call(qsub_args)

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(prog = 'uncert', add_help = False)
  parser.add_argument("-n", "--name", type = str, default = 'test', help = 'The run name')
  parser.add_argument("-s", "--nsteps", type = int, default = 5000, help = 'The number of MCMC steps')
  parser.add_argument("-w", "--nwalk", type = int, default = 40, help = 'The number of MCMC walkers')
  parser.add_argument("-b", "--nburn", type = int, default = 500, help = 'The number of burn-in steps')
  parser.add_argument("-e", "--nsamples", type = int, default = 1000, help = 'The number of evolution samples')
  parser.add_argument("-o", "--oceans", type = float, default = 10.0, help = 'The initial number of oceans (TO)')
  parser.add_argument("-h", "--hydrogen", type = float, default = 0.01, help = 'The initial mass of hydrogen (M_E)')
  parser.add_argument("-x", "--epsilon", type = float, default = 0.15, help = 'XUV efficiency')
  parser.add_argument("-m", "--magma", type = int, default = 0, help = 'Magma present?')
  parser.add_argument("-r", "--run", action = 'store_true', help = 'Run the evolution?')
  parser.add_argument("-p", "--plot", action = 'store_true', help = 'Plot the results?')
  args = parser.parse_args()
  
  with Pool() as pool:
    
    # Options
    kwargs = dict(name = args.name, nsteps = args.nsteps, nburn = args.nburn, nsamples = args.nsamples,
                  nwalk = args.nwalk, oceans = args.oceans, hydrogen = args.hydrogen, epshydro = args.epsilon, 
                  epswater = args.epsilon, magma = bool(args.magma), pool = pool)
    
    # Check for output dir
    if not os.path.exists(os.path.join(PATH, 'output')):
      os.makedirs(os.path.exists(os.path.join(PATH, 'output')))
    
    # Run
    if args.run:
      RunMCMC(**kwargs)
      RunEvol(**kwargs)
    
    # Plot
    if args.plot:
      PlotChains(**kwargs)
      PlotCorner(**kwargs)
      PlotEvol(**kwargs)