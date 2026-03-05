"""
compare_damage_energy.py
========================
Compares two methods for computing radiation damage energy in OpenMC:

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

Both methods are built on the NRT-DPA model (Lindhard/Robinson partition
function), so their results should agree in the limit of sufficient
statistics. The comparison is instructive because:

  * Method 1 is a deterministic (expected-value) estimator applied at
    every collision: it never has zero-weight events from a missing recoil.
  * Method 2 is a purely analog estimator: each sampled recoil contributes
    its actual partition-weighted energy. Coverage depends on which recoil
    particle types are requested in the filter.

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
tally_traditional.filters = [mat_filter]
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
tally_nrt.filters = [mat_filter, ppf_nrt]
tally_nrt.scores = ['events']

# -----------------------------------------------------------------------
# Tally 3: Recoil kinetic energy (no damage partition — reference)
# -----------------------------------------------------------------------
# Same as Tally 2 but with damage_model='recoil-energy', which weights
# each secondary by its full kinetic energy (wgt × E_recoil) with no
# electronic loss correction. This gives an upper bound on damage energy.
# as if all recoil energy went into atomic displacements.
# The ratio Tally2 / Tally3 is the effective damage fraction.
#
# Result units: eV per source neutron
ppf_ke = openmc.ParticleProductionFilter(recoil_types, damage_model='recoil-energy')

tally_ke = openmc.Tally(name='recoil kinetic energy (recoil-energy model)')
tally_ke.filters = [mat_filter, ppf_ke]
tally_ke.scores = ['events']

tallies = openmc.Tallies([tally_traditional, tally_nrt, tally_ke])

# =============================================================================
# 5. RUN
# =============================================================================
model = openmc.Model(geometry=geometry, materials=materials,
                     settings=settings, tallies=tallies)
sp_path = model.run()

# =============================================================================
# 6. POST-PROCESSING AND COMPARISON
# =============================================================================
with openmc.StatePoint(sp_path) as sp:

    t_trad = sp.get_tally(name='damage-energy (traditional, MT 444)')
    t_nrt  = sp.get_tally(name='damage-energy (ParticleProductionFilter/NRT)')
    t_ke   = sp.get_tally(name='recoil kinetic energy (recoil-energy model)')

    mean_trad, unc_trad = t_trad.mean.flat[0], t_trad.std_dev.flat[0]
    # NRT and recoil-energy tallies have one bin per recoil type; sum them all
    mean_nrt  = t_nrt.mean.sum()
    unc_nrt   = float(np.sqrt((t_nrt.std_dev**2).sum()))
    mean_ke   = t_ke.mean.sum()
    unc_ke    = float(np.sqrt((t_ke.std_dev**2).sum()))

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
print(f'  {"Recoil kinetic energy (upper bound)":<48} '
      f'{mean_ke:>13.4e}  {unc_ke:>13.4e}')
print()

if mean_trad > 0:
    ratio_nrt  = mean_nrt / mean_trad
    ratio_ke   = mean_ke  / mean_trad
    frac_damage = mean_nrt / mean_ke if mean_ke > 0 else float('nan')

    print(f'  NRT tally / Traditional         = {ratio_nrt:.4f}')
    print(f'  Recoil KE / Traditional         = {ratio_ke:.4f}')
    print(f'  NRT / Recoil KE (damage fraction) = {frac_damage:.4f}')
    print()
    print('  Interpretation:')
    print(f'    The NRT / Traditional ratio of {ratio_nrt:.4f} indicates how well the')
    print('    listed recoil types cover the total damage cross section.')
    print('    A ratio < 1 typically reflects missing minor reaction channels.')
    print(f'    The damage fraction ({frac_damage:.3f}) is the Lindhard-partitioned')
    print('    share of recoil kinetic energy that contributes to atomic')
    print('    displacements (the rest is lost to electronic excitation).')

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
# 7. DAMAGE ENERGY HISTOGRAM BY RECOIL TYPE
# =============================================================================
# Extract per-recoil contributions from the NRT tally.
# t_nrt has filters [mat_filter, ppf_nrt]; ppf_nrt has one bin per recoil type,
# so t_nrt.mean flattens to one value per entry in recoil_types.
per_recoil = t_nrt.mean.flatten()   # eV per source neutron, one entry per recoil type
total_de   = per_recoil.sum()

# Separate contributors that are ≥ 1 % of total; lump the rest into "other"
threshold  = 0.01 * total_de
major      = [(rt, v) for rt, v in zip(recoil_types, per_recoil) if v >= threshold]
other_val  = sum(v for v in per_recoil if v < threshold)

# Sort major contributors descending
major_sorted = sorted(major, key=lambda x: x[1], reverse=True)
if other_val > 0:
    major_sorted.append(('other', other_val))

bar_labels = [rt for rt, _ in major_sorted]
bar_pcts   = [100 * v / total_de for _, v in major_sorted]

colors = [f'C{i}' for i in range(len(bar_labels))]

fig, ax = plt.subplots(figsize=(max(6, len(bar_labels) * 0.9), 5))
ax.bar(bar_labels, bar_pcts, color=colors, width=0.6)

ax.set_ylim(0, 100)
ax.set_xlabel('Recoil species')
ax.set_ylabel('Percentage of total NRT damage energy (%)')
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
out_path = 'recoil_damage_energy.png'
fig.savefig(out_path, dpi=150)
print(f'  Histogram saved to {out_path}')
plt.close(fig)
