"""
compare_damage_spectrum.py
==========================
Compares two methods for computing the radiation damage energy spectrum in
OpenMC, i.e. damage energy as a function of incident neutron energy:

  Method 1 – Traditional "damage-energy" score
    Uses pre-computed MT 444 (damage-energy) cross sections processed by
    NJOY, which encode the expected damage energy per interaction averaged
    over the full angular distribution of each reaction channel. Evaluated
    via the collision or tracklength estimator.

  Method 2 – ParticleProductionFilter with damage_model='nrt'
    Applies the Lindhard/Robinson partition function on-the-fly to each
    sampled recoil secondary banked during transport. Requires
    settings.recoil_production = True. Evaluated via the analog "events"
    score.

Both tallies use 500 equal-lethargy energy bins so the spectra can be
compared directly on a log-log plot of damage energy per unit lethargy
versus incident neutron energy.

Model: monoenergetic point source (D-T neutrons) at the center
       of a 10 cm radius sphere of pure iron (Fe-56) at 7.87 g/cm³.
"""

import matplotlib.pyplot as plt
import numpy as np
import openmc

# =============================================================================
# 1. MATERIAL
# =============================================================================
iron = openmc.Material(name='iron')
iron.set_density('g/cm3', 7.87)
iron.add_element('Fe', 1.0)

materials = openmc.Materials([iron])

# =============================================================================
# 2. GEOMETRY
# =============================================================================
sphere = openmc.Sphere(r=10.0, boundary_type='vacuum')
cell = openmc.Cell(fill=iron, region=-sphere)
geometry = openmc.Geometry(openmc.Universe(cells=[cell]))

# =============================================================================
# 3. SETTINGS
# =============================================================================
settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.batches = 50
settings.particles = 100_000

# Monoenergetic neutron point source at the origin
E_source = 14.1e6
source = openmc.IndependentSource()
source.space = openmc.stats.Point()
source.angle = openmc.stats.Isotropic()
source.energy = openmc.stats.delta_function(E_source)
source.particle = 'neutron'
settings.source = source

# Enable recoil production so that heavy recoil nuclei are banked as
# secondary particles during transport. Required for Method 2.
settings.recoil_production = True
settings.recoil = {
    'multi_neutron_mode': 'one_particle',
    'missing_products': 'phase_space',
    'bank_emitted_ions': True,
    'capture_photons': 'banked'
}

# =============================================================================
# 4. TALLIES
# =============================================================================

# Restrict all tallies to the iron sphere interior
mat_filter = openmc.MaterialFilter(iron)

# 500 equal-lethargy bins from 1 meV to 20 MeV
E_min_plot, E_max_plot = 1e-3, 20e6  # eV
energy_bins = np.geomspace(E_min_plot, E_max_plot, 501)  # 501 edges → 500 bins
energy_filter = openmc.EnergyFilter(energy_bins)

# -----------------------------------------------------------------------
# Tally 1: Traditional damage-energy score
# -----------------------------------------------------------------------
# OpenMC looks up the MT 444 macroscopic cross section at each particle
# energy and scores it as a standard reaction rate via the collision or
# tracklength estimator. The MT 444 cross section was precomputed by NJOY
# as the integral of (σ_reaction × Lindhard-partition of recoil energy)
# averaged over all reaction angles, summed over all reaction channels.
#
# Result units: eV per source neutron
tally_traditional = openmc.Tally(name='damage-energy (traditional, MT 444)')
tally_traditional.filters = [mat_filter, energy_filter]
tally_traditional.scores = ['damage-energy']

# -----------------------------------------------------------------------
# Tally 2: ParticleProductionFilter with NRT damage model
# -----------------------------------------------------------------------
# For each collision event, the filter iterates over secondaries banked in
# the current event that match the listed particle types. The weight applied
# to the "events" score for each secondary is:
#
#   weight = site.wgt * lindhard_partition(E_R, Z_R, A_R, Z_L, A_L)
#
# where E_R is the recoil kinetic energy, Z_R/A_R are the recoil identity
# (from the PDG number of the secondary), and Z_L/A_L are the lattice
# identity (from the collision target nuclide).
#
# The recoil types below correspond to the main reaction channels on Fe-56:
#
#   Fe56  – elastic (n,n) and inelastic (n,n'γ) scattering
#   Fe55  – (n,2n) reaction
#   Mn56  – (n,p)  reaction   [Z=25, A=56]
#   Mn55  – (n,d) / (n,np)   [Z=25, A=55]
#   Cr53  – (n,α)             [Z=24, A=53]
#
# Minor channels such as (n,3n)→Fe54 and (n,he3)→Cr54 are omitted here;
# including them would bring Method 2 into closer agreement with Method 1.
#
# Result units: eV per source neutron
recoil_types = ['Fe56', 'Fe55', 'Fe54', 'Mn56', 'Mn55', 'Cr53', 'Cr54', 'H1', 'H2', 'He4', 'He3']

ppf_nrt = openmc.ParticleProductionFilter(recoil_types, damage_model='nrt')

tally_nrt = openmc.Tally(name='damage-energy (ParticleProductionFilter/NRT)')
tally_nrt.filters = [mat_filter, ppf_nrt, energy_filter]
tally_nrt.scores = ['events']

tallies = openmc.Tallies([tally_traditional, tally_nrt])

# =============================================================================
# 5. RUN
# =============================================================================
model = openmc.Model(geometry=geometry, materials=materials,
                     settings=settings, tallies=tallies)
sp_path = model.run()

# =============================================================================
# 6. POST-PROCESSING AND COMPARISON
# =============================================================================
N_E = len(energy_bins) - 1  # 500
N_R = len(recoil_types)    # number of recoil types

with openmc.StatePoint(sp_path) as sp:

    t_trad = sp.get_tally(name='damage-energy (traditional, MT 444)')
    t_nrt  = sp.get_tally(name='damage-energy (ParticleProductionFilter/NRT)')

    # Per-energy-bin spectra (eV / source neutron per bin)
    # t_trad filters: [mat(1), energy(N_E)]  → mean shape (N_E, 1, 1)
    spec_trad = t_trad.mean[:, 0, 0]
    # t_nrt filters: [mat(1), ppf(N_R), energy(N_E)] → mean shape (N_R*N_E, 1, 1)
    spec_nrt  = t_nrt.mean[:, 0, 0].reshape(N_R, N_E).sum(axis=0)

    # Totals (sum over energy bins)
    mean_trad = float(spec_trad.sum())
    unc_trad  = float(np.sqrt((t_trad.std_dev[:, 0, 0]**2).sum()))
    mean_nrt  = float(spec_nrt.sum())
    unc_nrt   = float(np.sqrt((t_nrt.std_dev[:, 0, 0].reshape(N_R, N_E)**2).sum()))

hline = '=' * 72

print()
print(hline)
print(f'  Damage Energy Comparison: Fe-56 sphere, {E_source/1e6:.1f} MeV neutron source')
print(hline)
print(f'\n  {"Method":<48} {"Mean [eV/src]":>13}  {"±1σ [eV/src]":>13}')
print('  ' + '-' * 78)
print(f'  {"Traditional (damage-energy score, MT 444)":<48} '
      f'{mean_trad:>13.4e}  {unc_trad:>13.4e}')
print(f'  {"ParticleProductionFilter (NRT, on-the-fly)":<48} '
      f'{mean_nrt:>13.4e}  {unc_nrt:>13.4e}')
print()

if mean_trad > 0:
    ratio_nrt = mean_nrt / mean_trad

    print(f'  NRT tally / Traditional         = {ratio_nrt:.4f}')
    print()
    print('  Interpretation:')
    print(f'    The NRT / Traditional ratio of {ratio_nrt:.4f} indicates how well the')
    print('    listed recoil types cover the total damage cross section.')
    print('    A ratio < 1 typically reflects missing minor reaction channels.')

print()
print('  Recoil channels included in the ParticleProductionFilter tally:')
for rtype in recoil_types:
    print(f'    {rtype}')
print()
print('  Note: To improve the NRT / Traditional agreement, add minor channels:')
print('    Fe54 [(n,3n)], Cr54 [(n,he3)], Cr55 [(n,2p)], etc.')
print(hline)
print()

# =============================================================================
# 7. DAMAGE ENERGY SPECTRUM (log-log)
# =============================================================================
# Plot damage energy per unit lethargy vs. incident neutron energy.
# Equal-lethargy bins have constant Δu = ln(E_max/E_min)/N_E.
du    = np.log(E_max_plot / E_min_plot) / N_E
E_mid = np.sqrt(energy_bins[:-1] * energy_bins[1:])  # geometric midpoints (eV)

# Damage energy per unit lethargy (eV / source neutron / Δu)
de_trad_dl = spec_trad / du
de_nrt_dl  = spec_nrt  / du

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(E_mid, de_trad_dl, label='Traditional (MT 444)', color='C0')
ax.loglog(E_mid, de_nrt_dl,  label='ParticleProductionFilter/NRT',
          color='C1', linestyle='--')

ax.set_xlim(left=1.0)
ax.set_xlabel('Incident neutron energy (eV)')
ax.set_ylabel('Damage energy per unit lethargy (eV / source neutron)')
ax.set_title(f'Damage energy spectrum — Fe sphere, {E_source/1e6:.1f} MeV neutrons')
ax.legend()
ax.grid(True, which='both', linestyle='--', alpha=0.4)

plt.tight_layout()
out_path = 'damage_energy_spectrum.png'
fig.savefig(out_path, dpi=150)
print(f'  Spectrum plot saved to {out_path}')
plt.close(fig)
