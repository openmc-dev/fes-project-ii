#!/usr/bin/env python3
"""
Plot comparison of OpenMC recoil spectrum vs SPECTRA-PKA JSON spectrum.

- Loads an OpenMC statepoint (default: newest statepoint.*.h5)
- Looks for a tally named 'recoil_distribution' (or a user-specified tally name)
- Loads the JSON produced by compare_pka.py (default: Fe_OpenMC_recoil_spectra.json)
- Extracts the spectrum matching the given (reaction, product) combination
- Plots both spectra on the same axes for visual comparison

Usage examples:
  python3 damage/plot_recoil_compare.py \
      --statepoint statepoint.20.h5 \
      --json Fe_OpenMC_recoil_spectra.json \
      --reaction "(n,elastic)" --product Fe56

  python3 damage/plot_recoil_compare.py \
      --reaction "(n,2n)" --product Fe55
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import openmc


def find_latest_statepoint() -> Optional[str]:
    """Return the path to the newest statepoint.*.h5 in CWD, or None if not found."""
    candidates = sorted(glob.glob("statepoint.*.h5"))
    if not candidates:
        return None
    # Sort by mtime descending
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_openmc_recoil_spectrum(
    statepoint_path: str,
    tally_name: str,
    reaction: str,
    product: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load recoil spectrum from an OpenMC statepoint tally.

    Parameters
    ----------
    statepoint_path : str
        Path to the statepoint HDF5 file.
    tally_name : str
        Name of the recoil-distribution tally.
    reaction : str
        OpenMC reaction name, e.g. ``'(n,elastic)'``.
    product : str
        Nuclide name of the reaction product, e.g. ``'Fe56'``.

    Returns
    -------
    energies, values
        Bin edges (eV) and per-bin event rates.
    """
    with openmc.StatePoint(statepoint_path) as sp:
        tally = sp.get_tally(name=tally_name)
        prod_filter = tally.filters[0]
        reaction_filter = tally.filters[1]
        reactions = [str(x) for x in reaction_filter.bins]
        energies = prod_filter.energies
        values = tally.get_reshaped_data(expand_dims=True) # (particle, energy, reaction, nuc, score)

        try:
            particle_index = prod_filter.particles.index(product)
        except ValueError:
            avail = list(prod_filter.particles)
            raise ValueError(
                f"Product '{product}' not found in ParticleProductionFilter. "
                f"Available products: {avail}"
            )

        try:
            reaction_index = reactions.index(reaction)
        except ValueError:
            raise ValueError(
                f"Reaction '{reaction}' not found in ReactionFilter. "
                f"Available reactions: {reactions}"
            )

        values = values[particle_index, :, reaction_index, 0, 0]

        # Divide by volume to get per-unit-volume rates
        values /= sp.summary.materials[0].volume

        return energies, values


def load_json_spectrum(
    json_path: str,
    reaction: str,
    product: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load a spectrum from the SPECTRA-PKA JSON by (reaction, product).

    Searches the ``spectra`` section for an entry whose ``reaction`` and
    ``product`` fields match the requested values.

    Returns (energies, rates, matched_key).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    spectra = data.get('spectra', {})

    matches = []
    for key, entry in spectra.items():
        if entry.get('reaction') == reaction and entry.get('product') == product:
            matches.append((key, entry))

    if len(matches) == 1:
        key, entry = matches[0]
        return np.asarray(entry['energies']), np.asarray(entry['pka_rates']), key
    elif len(matches) > 1:
        keys = [k for k, _ in matches]
        raise KeyError(
            f"Multiple entries match reaction='{reaction}', product='{product}': {keys}"
        )

    # No match found — build helpful error message
    available = [
        (e.get('reaction'), e.get('product'))
        for e in spectra.values()
        if e.get('reaction') is not None
    ]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for pair in available:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    raise KeyError(
        f"No entry with reaction='{reaction}', product='{product}' in {json_path}.\n"
        f"Available (reaction, product) pairs:\n"
        + "\n".join(f"  {r}  {p}" for r, p in unique[:20])
        + ("\n  ..." if len(unique) > 20 else "")
    )


def plot_compare(openmc_E: np.ndarray, openmc_Y: np.ndarray, json_E: np.ndarray, json_Y: np.ndarray,
                 reaction: str, product: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.stairs(openmc_Y, openmc_E, label='OpenMC')
    plt.stairs(json_Y, json_E, label='SPECTRA-PKA')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Recoil Energy [eV]')
    plt.ylabel('Rate [PKAs/s/cm³]')
    title = f'Recoil Spectrum: {reaction} \u2192 {product}'
    plt.title(title)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('recoil_spectrum_comparison.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare OpenMC recoil tally to SPECTRA-PKA JSON spectrum.')
    parser.add_argument('--statepoint', type=str, default=None, help='Path to OpenMC statepoint.*.h5 (default: newest in CWD)')
    parser.add_argument('--tally-name', type=str, default='recoil_distribution', help='Name of OpenMC tally to use')
    parser.add_argument('--list-tallies', action='store_true', help='List tallies in the statepoint and exit')
    parser.add_argument('--json', type=str, default='Fe_OpenMC_recoil_spectra.json', help='Path to recoil spectra JSON file')
    parser.add_argument('--reaction', type=str, default='(n,elastic)', help='OpenMC reaction name, e.g. "(n,elastic)"')
    parser.add_argument('--product', type=str, default='Fe56', help='Nuclide name of the reaction product, e.g. "Fe56"')
    args = parser.parse_args()

    statepoint = args.statepoint or find_latest_statepoint()
    if statepoint is None or not os.path.exists(statepoint):
        raise FileNotFoundError("Could not find a statepoint file. Pass --statepoint explicitly.")

    if not os.path.exists(args.json):
        raise FileNotFoundError(f"JSON file not found: {args.json}")

    # Optionally list tallies and exit
    if args.list_tallies:
        with openmc.StatePoint(statepoint) as sp:
            tallies = list(sp.tallies.values())
            print(f"Tallies in {statepoint}:")
            for t in tallies:
                filters = [getattr(f, 'type', type(f).__name__) for f in (getattr(t, 'filters', []) or [])]
                scores = list(getattr(t, 'scores', []) or [])
                print(f"  - id:{t.id} name:'{getattr(t, 'name', '')}' filters:{filters} scores:{scores}")
        return

    # Load data
    E_openmc, Y_openmc = load_openmc_recoil_spectrum(
        statepoint, args.tally_name, args.reaction, args.product
    )
    E_json, Y_json, found_key = load_json_spectrum(
        args.json, args.reaction, args.product
    )

    # Plot
    plot_compare(E_openmc, Y_openmc, E_json, Y_json, args.reaction, args.product)


if __name__ == '__main__':
    main()
