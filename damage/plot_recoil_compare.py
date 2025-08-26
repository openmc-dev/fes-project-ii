#!/usr/bin/env python3
"""
Plot comparison of OpenMC recoil spectrum vs SPECTRA-PKA JSON spectrum.

- Loads an OpenMC statepoint (default: newest statepoint.*.h5)
- Looks for a tally named 'recoil_distribution' (or a user-specified tally name)
- Loads the JSON produced by compare_pka.py (default: Fe_OpenMC_recoil_spectra.json)
- Extracts the 'Fe56(n,elastic)' spectrum (or a user-specified key)
- Plots both spectra on the same axes for visual comparison

Usage examples:
  python3 damage/plot_recoil_compare.py \
      --statepoint statepoint.20.h5 \
      --tally-name recoil_distribution \
      --json Fe_OpenMC_recoil_spectra.json \
      --json-key "Fe56(n,elastic)"
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


def load_openmc_recoil_spectrum(statepoint_path: str, tally_name: str, tally_id: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load recoil spectrum from an OpenMC statepoint tally.

    Returns (energies, values) as 1D arrays.
    """
    with openmc.StatePoint(statepoint_path) as sp:
        tally = sp.get_tally(name=tally_name)
        energies = tally.filters[1].values
        values = tally.mean.ravel()

        # Divide by volume to get per-unit-volume rates
        values /= sp.summary.materials[0].volume

        return energies, values


def load_json_spectrum(json_path: str, key: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load a spectrum from the SPECTRA-PKA JSON. Returns (energies, rates, found_key)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    spectra = data.get('spectra', {})
    if key in spectra:
        entry = spectra[key]
        return np.asarray(entry['energies']), np.asarray(entry['pka_rates']), key
    # Fallback: find a key that starts with the base (handles '#2' duplicates)
    for k in spectra.keys():
        if k == key or k.startswith(key + '#'):
            entry = spectra[k]
            return np.asarray(entry['energies']), np.asarray(entry['pka_rates']), k
    # Last resort: case-insensitive search
    key_low = key.lower()
    for k in spectra.keys():
        if key_low == k.lower() or k.lower().startswith(key_low + '#'):
            entry = spectra[k]
            return np.asarray(entry['energies']), np.asarray(entry['pka_rates']), k
    raise KeyError(f"Key '{key}' not found in JSON {json_path}. Available keys include e.g.: {list(spectra.keys())[:8]} …")


def plot_compare(openmc_E: np.ndarray, openmc_Y: np.ndarray, json_E: np.ndarray, json_Y: np.ndarray,
                 json_key: str, title_suffix: str = "") -> None:
    plt.figure(figsize=(8, 6))

    plt.stairs(openmc_Y, openmc_E, label='OpenMC tally (recoil_distribution)', lw=2)
    plt.stairs(json_Y, json_E, label=f'SPECTRA-PKA JSON ({json_key})', lw=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Recoil Energy [eV]')
    plt.ylabel('Rate [PKAs/s/cm³]')
    title = 'Recoil Spectrum Comparison'
    if title_suffix:
        title += f' — {title_suffix}'
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
    parser.add_argument('--tally-id', type=int, default=None, help='ID of OpenMC tally to use (overrides name if provided)')
    parser.add_argument('--list-tallies', action='store_true', help='List tallies in the statepoint and exit')
    parser.add_argument('--json', type=str, default='Fe_OpenMC_recoil_spectra.json', help='Path to recoil spectra JSON file')
    parser.add_argument('--json-key', type=str, default='Fe56(n,elastic)', help='Key in JSON to plot')
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
    E_openmc, Y_openmc = load_openmc_recoil_spectrum(statepoint, args.tally_name, args.tally_id)
    E_json, Y_json, found_key = load_json_spectrum(args.json, args.json_key)

    # Plot
    title_suffix = os.path.basename(statepoint)
    plot_compare(E_openmc, Y_openmc, E_json, Y_json, found_key, title_suffix=title_suffix)


if __name__ == '__main__':
    main()
