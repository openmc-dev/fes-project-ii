import os
import re
import subprocess
from math import pi

import openmc
import matplotlib.pyplot as plt
import numpy as np


def run_openmc_simulation():
    """Run OpenMC simulation and extract neutron flux.

    Extended to support multi-nuclide materials (e.g., natural Fe).
    Returns the OpenMC model so we can query the nuclides present.
    """
    print("Running OpenMC simulation...")

    # Create natural Fe material (multi-nuclide)
    mat = openmc.Material()
    # First example: natural Fe (user request)
    mat.add_element('Fe', 1.0)
    mat.set_density('g/cm3', 7.874)

    # Create spherical geometry
    R = 10.0
    sph = openmc.Sphere(r=R, boundary_type='vacuum')
    cell = openmc.Cell(fill=mat, region=-sph)
    cell.volume = 4/3 * pi * R**3
    model = openmc.Model()
    model.geometry = openmc.Geometry([cell])

    # Settings
    model.settings.batches = 20
    model.settings.particles = 1000
    model.settings.run_mode = 'fixed source'
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(),
        energy=openmc.stats.delta_function(14.0e6),
        strength=1.0e12,
    )

    # Create tally for neutron flux
    tally = openmc.Tally()
    energy_filter = openmc.EnergyFilter.from_group_structure('VITAMIN-J-175')
    tally.filters = [energy_filter]
    tally.scores = ['flux']
    model.tallies.append(tally)

    # Run simulation
    model.run(apply_tally_results=True)

    # Extract results
    flux = tally.mean.ravel()
    flux /= cell.volume  # Convert to flux per unit volume
    energies = energy_filter.values  # Energy bin boundaries in eV

    return flux, energies, model


def create_spectra_pka_flux_file(flux, energies, filename="openmc_flux.dat"):
    """Create SPECTRA-PKA compatible flux file from OpenMC results"""
    print(f"Creating SPECTRA-PKA flux file: {filename}")

    # Convert energies from eV to MeV for SPECTRA-PKA
    energies_mev = energies / 1e6

    # Number of energy groups
    num_groups = len(flux)

    with open(filename, 'w') as f:
        # Header line (description)
        f.write("OpenMC-generated neutron flux for target material\n")

        # Control line: itype=2, dummy=0, igroup=2 (n/s/cm2), dummy=0, acnm=1.0, time=1.0
        f.write("2 0 2 0 1.0 1.0\n")

        # Number of groups and ksail (error flag, -2 means no errors)
        f.write(f"{num_groups} -2\n")

        # Energy bin boundaries (num_groups + 1 values in MeV)
        for energy_mev in energies_mev:
            f.write(f"{energy_mev:.10E}\n")

        # Flux values (n/s/cm2)
        for flux_val in flux:
            f.write(f"{flux_val:.10E}\n")

    print(f"Flux file created with {num_groups} energy groups")
    return filename


def _parse_nuclide_name(nuc: str):
    """Parse a nuclide string like 'Fe56' -> ('Fe', 56)."""
    i = 0
    while i < len(nuc) and not nuc[i].isdigit():
        i += 1
    symbol = nuc[:i]
    mass_str = nuc[i:]
    try:
        A = int(mass_str)
    except ValueError:
        A = None
    return symbol, A


def _pka_filepath_for_nuclide(nuc: str, base_dir: str) -> str:
    """Build the SPECTRA-PKA file path for a nuclide name like 'Fe56'."""
    sym, A = _parse_nuclide_name(nuc)
    if sym and A:
        return os.path.join(base_dir, f"{sym}{A:03d}s.asc")
    return ""


def _atomic_mass_amu(nuc: str) -> float:
    """Best-effort atomic mass (amu) for a nuclide string like 'Fe56'.
    Tries openmc.data.atomic_mass, else falls back to mass number.
    """
    try:
        from openmc.data import atomic_mass
        return float(atomic_mass(nuc))
    except Exception:
        # Fallback to mass number if available
        _, A = _parse_nuclide_name(nuc)
        return float(A) if A else 0.0


def create_spectra_pka_input_file(
    flux_filename,
    results_stub="Fe_OpenMC",
    model: openmc.Model | None = None,
    pka_base_dir: str = "/opt/data/spectra-pka/tendl2019/pka",
):
    """Create SPECTRA-PKA input file.

    - Uses the provided OpenMC model to find all nuclides present in geometry.
    - Writes one pka_filename row per nuclide using TENDL2019 SPECTRA-PKA data.
    """
    input_filename = f"{results_stub}.in"
    print(f"Creating SPECTRA-PKA input file: {input_filename}")

    # Collect nuclides present in the model geometry
    nuclides = []
    if model is not None and model.geometry is not None:
        try:
            # Can return dict-like of nuclides; normalize to strings
            found = model.geometry.get_all_nuclides()
            # found may be dict of {Nuclide/str: float}; take the keys
            for k in (found.keys() if hasattr(found, 'keys') else list(found)):
                name = getattr(k, 'name', None) or (str(k) if not isinstance(k, str) else k)
                nuclides.append(name)
        except Exception:
            pass

    # Fallback: if we couldn't query, assume Fe natural set
    if not nuclides:
        nuclides = ["Fe054", "Fe056", "Fe057", "Fe058"]  # strings may not match 'Fe56'
        # Normalize to standard 'Fe54' etc
        nuclides = [f"Fe{int(n[2:]):d}" if n.startswith("Fe") else n for n in nuclides]

    # Build rows for nuclides with available PKA files
    pka_rows = []
    for nuc in sorted(set(nuclides)):
        # Ensure format like 'Fe56'
        sym, A = _parse_nuclide_name(nuc)
        if not sym or not A:
            continue
        pka_file = _pka_filepath_for_nuclide(nuc, pka_base_dir)
        if not os.path.exists(pka_file):
            print(f"Warning: PKA file not found for {nuc}: {pka_file} (skipping)")
            continue
        # Masses for (n,gamma) estimate
        parent_mass = _atomic_mass_amu(nuc)
        daughter = f"{sym}{A+1}"
        daughter_mass = _atomic_mass_amu(daughter)
        # Parent element symbol and mass number
        parent_ele = sym
        parent_num = A
        pka_rows.append((pka_file, 1.0, parent_ele, parent_num, parent_mass, daughter_mass))

    with open(input_filename, 'w') as f:
        f.write(f'flux_filename="{flux_filename}"\n')
        f.write(f'results_stub="{results_stub}"\n')
        f.write('num_columns=6\n')
        f.write('columns= pka_filename pka_ratios parent_ele parent_num ngamma_parent_mass ngamma_daughter_mass\n')
        # One line per nuclide present
        for (pka_file, ratio, ele, anum, m_parent, m_daughter) in pka_rows:
            f.write(f'"{pka_file}" {ratio:.1f} {ele} {anum} {m_parent} {m_daughter}\n')
        f.write('flux_norm_type=2\n')
        f.write('pka_filetype=2\n')
        f.write('do_mtd_sums=.true.\n')
        f.write('do_ngamma_estimate=.t.\n')
        f.write('do_global_sums=.t.\n')
        f.write('do_exclude_light_from_total=.t.\n')
        f.write(f'number_pka_files={len(pka_rows)}\n')
        f.write('energies_once_perfile=.t.\n')
        f.write('do_tdam=.t.\n')
        f.write('assumed_ed=40.0\n')

    return input_filename


def run_spectra_pka(input_filename):
    """Run SPECTRA-PKA with the given input file"""
    print(f"Running SPECTRA-PKA with input file: {input_filename}")

    # Path to SPECTRA-PKA executable
    spectra_pka_exe = "damage/SPECTRA-PKA/spectra-pka"

    try:
        # Run SPECTRA-PKA
        result = subprocess.run(
            [spectra_pka_exe, input_filename],
            cwd=".",
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print("SPECTRA-PKA completed successfully")
            print("STDOUT:", result.stdout)
        else:
            print("SPECTRA-PKA failed with return code:", result.returncode)
            print("STDERR:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("SPECTRA-PKA timed out")
        return False
    except FileNotFoundError:
        print(f"SPECTRA-PKA executable not found at {spectra_pka_exe}")
        return False

    return True


def read_spectra_pka_results(results_stub="Fe56_OpenMC"):
    """Read SPECTRA-PKA results files and extract all reaction channels"""
    print(f"Reading SPECTRA-PKA results for {results_stub}")

    # Files created by SPECTRA-PKA
    out_file = f"{results_stub}.out"
    index_file = f"{results_stub}.indexes"

    if not os.path.exists(out_file):
        print(f"Results file {out_file} not found")
        return None, None

    if not os.path.exists(index_file):
        print(f"Index file {index_file} not found")
        return None, None

    # Read index file to understand the structure
    with open(index_file, 'r') as f:
        index_lines = f.readlines()

    # Parse index file to get reaction information
    reaction_channels = {}
    for line in index_lines:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split(None, 1)  # Split into at most 2 parts: index and rest
            if len(parts) >= 2:
                try:
                    index = int(parts[0])
                    description = parts[1]  # Full description including reaction type
                    reaction_channels[index] = description
                except ValueError:
                    continue

    print(f"Found {len(reaction_channels)} reaction channels")

    # Read the PKA results file
    with open(out_file, 'r') as f:
        lines = f.readlines()

    # Extract data for each reaction channel
    all_reactions = {}
    current_index = None
    current_reaction = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for section markers
        if line.startswith("### index") and "#####" in line:
            # Extract index number
            try:
                parts = line.split()
                current_index = int(parts[2])
                if current_index in reaction_channels:
                    current_reaction = reaction_channels[current_index]
                    print(f"Processing index {current_index}: {current_reaction}")

                    # Initialize storage for this reaction
                    energies_low = []
                    energies_high = []
                    pka_rates = []

                    # Read data lines for this section
                    j = i + 1
                    while j < len(lines):
                        data_line = lines[j].strip()

                        # Stop if we hit another section or empty lines indicating end
                        if data_line.startswith("### index") or (not data_line and j > i + 5):
                            break

                        # Parse energy and PKA rate data (skip comments and headers)
                        if data_line and not data_line.startswith('#') and len(data_line.split()) >= 3:
                            try:
                                parts = data_line.split()
                                energy_low = float(parts[0])  # MeV
                                energy_high = float(parts[1])  # MeV
                                pka_rate = float(parts[2])   # PKAs/s

                                energies_low.append(energy_low)
                                energies_high.append(energy_high)
                                pka_rates.append(pka_rate)
                            except (ValueError, IndexError):
                                pass
                        j += 1

                    # Store data if we found any
                    if energies_low and pka_rates:
                        # Convert to numpy arrays and eV
                        energies_low_ev = np.array(energies_low) * 1e6  # MeV to eV
                        energies_high_ev = np.array(energies_high) * 1e6  # MeV to eV
                        energy_centers = (energies_low_ev + energies_high_ev) / 2

                        all_reactions[current_index] = {
                            'name': current_reaction,
                            'energies': energy_centers,
                            'pka_rates': np.array(pka_rates)
                        }
                        print(f"  - Stored {len(energy_centers)} data points")

                    i = j - 1  # Continue from where we left off
            except (ValueError, IndexError):
                pass
        i += 1

    print(f"\nSuccessfully extracted {len(all_reactions)} reaction channels with data")

    # Create summary of key reactions
    key_reactions = {}
    for idx, data in all_reactions.items():
        reaction_name = data['name']
        # Categorize reactions based on exact descriptions from index file
        if reaction_name.startswith('(n,elastic) recoil matrix'):
            key_reactions['elastic'] = data
        elif reaction_name.startswith('(n,inelastic) recoil matrix'):
            key_reactions['inelastic'] = data
        elif reaction_name.startswith('scatter recoil matrix'):
            key_reactions['scatter'] = data
        elif reaction_name.startswith('(n,2n) recoil matrix'):
            key_reactions['n_2n'] = data
        elif reaction_name.startswith('total (z,p) recoil matrix'):
            key_reactions['n_p'] = data
        elif reaction_name.startswith('total (z,a) recoil matrix'):
            key_reactions['n_alpha'] = data
        elif reaction_name.startswith('estimated (n,g) recoil matrix'):
            key_reactions['n_gamma'] = data
        elif 'recoil matrix (He+H+unknowns excluded)' in reaction_name:
            key_reactions['total'] = data

    return all_reactions, key_reactions


def plot_results(flux, flux_energies, all_reactions, key_reactions):
    """Plot OpenMC flux and SPECTRA-PKA results (2x2 layout)."""
    print("Creating plots...")

    # Small performance tweaks for faster rendering
    try:
        import matplotlib as mpl
        mpl.rcParams['path.simplify'] = True
        mpl.rcParams['path.simplify_threshold'] = 0.5
        mpl.rcParams['agg.path.chunksize'] = 10000
        mpl.rcParams['lines.antialiased'] = False
    except Exception:
        pass

    def _format_reaction_label(desc: str) -> str:
        """Format channel description into 'Target(n,xxx)' like 'Fe56(n,gamma)'."""
        # Find target after 'from [ Fe56 ]' or in 'recoil matrix of Fe56'
        target = None
        m = re.search(r"from\s*\[\s*([A-Za-z]{1,2}\d{2,3})\s*\]", desc)
        if m:
            target = m.group(1)
        if not target:
            m2 = re.search(r"recoil matrix of\s+([A-Za-z]{1,2}\d{0,3})", desc)
            if m2:
                target = m2.group(1)
        # Extract reaction in parentheses containing 'n,' if present
        rx = re.search(r"\((n,[^)]+)\)", desc)
        reaction = None
        if rx:
            reaction = rx.group(1)
            # Expand short forms
            reaction = reaction.replace(',g', ',gamma').replace(',a', ',alpha')
        elif 'scatter recoil matrix' in desc:
            reaction = 'n,scatter'
        elif re.search(r'total\s*\(z,p\)', desc):
            reaction = 'n,p'
        elif re.search(r'total\s*\(z,a\)', desc):
            reaction = 'n,alpha'
        elif 'estimated (n,g)' in desc:
            reaction = 'n,gamma'
        # Handle totals without explicit (z,*) tokens
        if reaction is None:
            if re.search(r'\btotal\s+proton\s+matrix\b', desc, re.IGNORECASE) or re.search(r'\bproton\s+matrix\b', desc, re.IGNORECASE):
                reaction = 'n,p'
            elif re.search(r'\btotal\s+alpha\s+matrix\b', desc, re.IGNORECASE) or re.search(r'\balpha\s+matrix\b', desc, re.IGNORECASE):
                reaction = 'n,alpha'
        # Fallbacks
        if target and reaction:
            return f"{target}({reaction})"
        if target and 'recoil matrix' in desc:
            return f"{target}(recoil)"
        return desc

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # Plot 1: OpenMC neutron flux
    ax1.stairs(flux, flux_energies)
    ax1.set_xlabel('Energy [eV]')
    ax1.set_ylabel('Neutron Flux [n/cm²/s]')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('OpenMC Neutron Flux')
    ax1.grid(True, alpha=0.3)

    # Add flux statistics
    total_flux = np.trapezoid(flux, flux_energies[:-1])
    ax1.text(0.02, 0.98, f'Total: {total_flux:.2e} n/cm²/s',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Colors dict removed; plots use default color cycle for simplicity

    # Plot 2: Top-10 channels by integrated PKAs/s (standard reactions only)
    top10 = []
    if all_reactions is not None:
        def _is_standard_channel(name: str) -> bool:
            # Exclude aggregates and totals
            lname = name.lower()
            banned = [
                'input_spectrum',
                'total recoil matrix (he+h+unknowns excluded)'.lower(),
                'scatter recoil matrix',
                'total proton matrix',
                'total alpha matrix',
                'recoil matrix of ',
                'elemental recoil matrix of ',
                'total (z,p)',
                'total (z,a)'
            ]
            if any(b in lname for b in banned):
                return False
            # Include standard (n,*) channels and the (n,gamma) estimate
            if 'estimated (n,g)' in name:
                return True
            return re.search(r"\(n,[^)]+\)", name) is not None

        for idx, data in all_reactions.items():
            name = data['name']
            # Only consider standard reaction channels, not aggregates
            if not _is_standard_channel(name):
                continue
            nonzero_mask = data['pka_rates'] > 0
            if not np.any(nonzero_mask):
                continue
            energies = data['energies'][nonzero_mask]
            rates = data['pka_rates'][nonzero_mask]
            rate_int = np.trapezoid(rates, energies)
            top10.append((rate_int, name, energies, rates))
        top10.sort(reverse=True, key=lambda x: x[0])
        top10 = top10[:10]

        # Plot them
        color_cycle = plt.cm.tab10.colors
        for i, (rate_int, name, energies, rates) in enumerate(top10):
            label = _format_reaction_label(name)
            ax2.loglog(energies, rates, label=f"{i+1}. {label}", color=color_cycle[i % len(color_cycle)], linewidth=1.8)

        ax2.set_xlabel('PKA Energy [eV]')
        ax2.set_ylabel('PKA Rate [PKAs/s]')
        ax2.set_title('Top-10 Channels by Integrated PKAs/s')
        ax2.grid(True, alpha=0.3)
        if top10:
            ax2.legend(fontsize=7)

    # Plot 3: Total PKA spectrum
    if key_reactions is not None and 'total' in key_reactions:
        data = key_reactions['total']
        nonzero_mask = data['pka_rates'] > 0
        if np.any(nonzero_mask):
            energies = data['energies'][nonzero_mask]
            rates = data['pka_rates'][nonzero_mask]
            ax3.loglog(energies, rates, 'k-', linewidth=3, label='Total PKA Rate')

            # Add damage threshold
            damage_threshold = 40  # eV
            ax3.axvline(damage_threshold, color='red', linestyle='--', alpha=0.7,
                       label=f'Damage Threshold ({damage_threshold} eV)')

            # Add statistics
            total_pka_rate = np.trapezoid(rates, energies)
            above_threshold = energies >= damage_threshold
            if np.any(above_threshold):
                damage_rate = np.trapezoid(rates[above_threshold], energies[above_threshold])
                ax3.text(0.02, 0.98, f'Total: {total_pka_rate:.2e} PKAs/s\n'
                                     f'Above {damage_threshold} eV: {damage_rate:.2e} PKAs/s\n'
                                     f'Fraction: {damage_rate/total_pka_rate:.1%}',
                         transform=ax3.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    ax3.set_xlabel('PKA Energy [eV]')
    ax3.set_ylabel('PKA Rate [PKAs/s]')
    ax3.set_title('Total PKA Spectrum')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Top-10 contributions bar chart (bottom-right)
    if top10:
        names = [_format_reaction_label(name) for _, name, _, _ in top10]
        rates = [rate for rate, _, _, _ in top10]
        colors_list = [plt.cm.tab10.colors[i % 10] for i in range(len(top10))]
        bars = ax4.bar(range(len(names)), rates, color=colors_list)
        ax4.set_xlabel('Reaction Channel (Top-10)')
        ax4.set_ylabel('Integrated PKA Rate [PKAs/s]')
        ax4.set_title('Top-10 Channel Contributions')
        ax4.set_yscale('log')
        ax4.set_xticks(range(len(names)))
        # Truncate long names for readability
        tick_labels = [n if len(n) <= 22 else (n[:21] + '…') for n in names]
        ax4.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        ax4.grid(True, alpha=0.3)

        # Add values on bars
        for bar, rate in zip(bars, rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.05,
                     f'{rate:.1e}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    # Intentionally skip saving to PNG by default to avoid memory blow-ups during compression.
    # To enable saving, set environment variable SAVE_PNG=1.
    if os.environ.get('SAVE_PNG', '').lower() in ('1', 'true', 'yes'):
        plt.savefig('openmc_spectra_pka_results.png', dpi=150)
    plt.show()

    # Print detailed summary
    print("\n" + "="*80)
    print("DETAILED ANALYSIS SUMMARY")
    print("="*80)
    print("OpenMC Simulation:")
    print(f"  - Energy groups: {len(flux)}")
    print(f"  - Energy range: {flux_energies[0]:.1e} - {flux_energies[-1]:.1e} eV")
    print(f"  - Total neutron flux: {np.trapezoid(flux, flux_energies[:-1]):.2e} n/cm²/s")

    if all_reactions:
        print("\nSPECTRA-PKA Results:")
        print(f"  - Total reaction channels found: {len(all_reactions)}")

        # Summary of key reactions
        if key_reactions:
            print("\nKey Reaction Channels:")
            total_all_rates = 0
            for reaction_type, data in key_reactions.items():
                nonzero_mask = data['pka_rates'] > 0
                if np.any(nonzero_mask):
                    energies = data['energies'][nonzero_mask]
                    rates = data['pka_rates'][nonzero_mask]
                    integrated_rate = np.trapezoid(rates, energies)
                    max_rate = np.max(rates)
                    energy_range = f"{energies.min():.1e} - {energies.max():.1e}"

                    print(f"  {reaction_type:12s}: {integrated_rate:.2e} PKAs/s "
                          f"(max: {max_rate:.2e}, range: {energy_range} eV)")

                    if reaction_type != 'total':
                        total_all_rates += integrated_rate

            print(f"\nSum of individual reactions: {total_all_rates:.2e} PKAs/s")

            # Damage analysis
            if 'total' in key_reactions:
                data = key_reactions['total']
                nonzero_mask = data['pka_rates'] > 0
                if np.any(nonzero_mask):
                    energies = data['energies'][nonzero_mask]
                    rates = data['pka_rates'][nonzero_mask]

                    print("\nDamage Analysis (40 eV threshold):")
                    for threshold in [10, 40, 100, 1000]:
                        above_threshold = energies >= threshold
                        if np.any(above_threshold):
                            damage_rate = np.trapezoid(rates[above_threshold], energies[above_threshold])
                            total_rate = np.trapezoid(rates, energies)
                            fraction = damage_rate / total_rate if total_rate > 0 else 0
                            print(f"  Above {threshold:4d} eV: {damage_rate:.2e} PKAs/s ({fraction:.1%})")

    print("="*80)


def main():
    """Main function to run the complete analysis"""
    print("=" * 60)
    print("OpenMC + SPECTRA-PKA Integration for Fe56 PKA Analysis")
    print("=" * 60)

    # Step 1: Run OpenMC simulation
    flux, energies, model = run_openmc_simulation()

    # Step 2: Create SPECTRA-PKA flux file
    flux_filename = create_spectra_pka_flux_file(flux, energies)

    # Step 3: Create SPECTRA-PKA input file
    results_stub = "Fe_OpenMC"
    input_filename = create_spectra_pka_input_file(
        flux_filename,
        results_stub,
        model=model,
        pka_base_dir="/opt/data/spectra-pka/tendl2019/pka",
    )

    # Step 4: Run SPECTRA-PKA
    success = run_spectra_pka(input_filename)

    # Step 5: Read and analyze results
    all_reactions = None
    key_reactions = None
    if success:
        all_reactions, key_reactions = read_spectra_pka_results(results_stub)

    # Step 6: Plot results
    plot_results(flux, energies, all_reactions, key_reactions)

    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
