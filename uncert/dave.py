#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
dave.py
-------
Reshapes the output for Dave's pandas stuff.

'''

import numpy as np

name = 'output/w5h0m0'
data = np.load(name + '.mcmc.npz')
chain = data['chain']
blobs = data['blobs']

# Reshape and merge
blobs = np.swapaxes(blobs,0,1)
chain = np.concatenate([chain, blobs], axis = -1)
chain = chain.reshape((chain.shape[0] * chain.shape[1], -1))

# Labels
params = ['dMass (Solar)', 'dSatXUVFrac', 'dSatXUVTime (Gyr)', 'dStopTime (Gyr)', 'dXUVBeta',
          'dSemi (AU)', 'dPlanetMass (Earth)', 'dLuminosity (Solar)', 'dLXUV (Solar)', 'dRGTime (Myr)', 
          'dEnvMass (Earth)', 'dWaterMass (TO)', 'dOxygenMass (bar)']