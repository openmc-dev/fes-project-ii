import os
import subprocess
from math import pi

import openmc
import matplotlib.pyplot as plt
import numpy as np


def run_openmc_simulation():
    """Run OpenMC simulation and extract neutron flux"""
    print("Running OpenMC simulation...")

    # Create Fe56 material
    mat = openmc.Material()
    mat.add_nuclide('Fe56', 1.0)
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

    return flux, energies


def create_spectra_pka_flux_file(flux, energies, filename="openmc_flux.dat"):
    """Create SPECTRA-PKA compatible flux file from OpenMC results"""
    print(f"Creating SPECTRA-PKA flux file: {filename}")

    # Convert energies from eV to MeV for SPECTRA-PKA
    energies_mev = energies / 1e6

    # Number of energy groups
    num_groups = len(flux)

    with open(filename, 'w') as f:
        # Header line (description)
        f.write("OpenMC-generated neutron flux for Fe56 target\n")

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


def create_spectra_pka_input_file(flux_filename, results_stub="Fe56_OpenMC"):
    """Create SPECTRA-PKA input file for Fe56"""
    input_filename = f"{results_stub}.in"
    print(f"Creating SPECTRA-PKA input file: {input_filename}")

    # Path to Fe56 PKA data file
    pka_data_file = "SPECTRA-PKA/manual/usergrid_test/Fe056s.asc"

    with open(input_filename, 'w') as f:
        f.write(f'flux_filename="{flux_filename}"\n')
        f.write(f'results_stub="{results_stub}"\n')
        f.write('num_columns=6\n')
        f.write('columns= pka_filename pka_ratios parent_ele parent_num ngamma_parent_mass ngamma_daughter_mass\n')
        f.write(f'"{pka_data_file}" 1.0 Fe 56 55.934936326 56.935392841\n')
        f.write('flux_norm_type=2\n')
        f.write('pka_filetype=2\n')
        f.write('do_mtd_sums=.true.\n')
        f.write('do_ngamma_estimate=.t.\n')
        f.write('do_global_sums=.t.\n')
        f.write('do_exclude_light_from_total=.t.\n')
        f.write('number_pka_files=1\n')
        f.write('energies_once_perfile=.t.\n')
        f.write('do_tdam=.t.\n')
        f.write('assumed_ed=40.0\n')

    return input_filename


def run_spectra_pka(input_filename):
    """Run SPECTRA-PKA with the given input file"""
    print(f"Running SPECTRA-PKA with input file: {input_filename}")

    # Path to SPECTRA-PKA executable
    spectra_pka_exe = "SPECTRA-PKA/spectra-pka"

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
    """Plot OpenMC flux and SPECTRA-PKA results for all reaction channels"""
    print("Creating plots...")

    plt.figure(figsize=(16, 12))

    # Create a 2x3 subplot layout
    ax1 = plt.subplot(2, 3, 1)  # Neutron flux
    ax2 = plt.subplot(2, 3, 2)  # Key reactions
    ax3 = plt.subplot(2, 3, 3)  # Total PKA spectrum
    ax4 = plt.subplot(2, 3, 4)  # Elastic vs inelastic
    ax5 = plt.subplot(2, 3, 5)  # Nuclear reactions
    ax6 = plt.subplot(2, 3, 6)  # Reaction contributions

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

    # Define colors for different reaction types
    colors = {
        'elastic': 'blue',
        'inelastic': 'green',
        'scatter': 'cyan',
        'n_2n': 'red',
        'n_p': 'orange',
        'n_alpha': 'purple',
        'n_gamma': 'brown',
        'total': 'black'
    }

    # Plot 2: Key reactions overview
    if key_reactions:
        for reaction_type, data in key_reactions.items():
            if reaction_type != 'total':  # Plot total separately
                nonzero_mask = data['pka_rates'] > 0
                if np.any(nonzero_mask):
                    energies = data['energies'][nonzero_mask]
                    rates = data['pka_rates'][nonzero_mask]
                    ax2.loglog(energies, rates, label=reaction_type.replace('_', ','),
                              color=colors.get(reaction_type, 'gray'), linewidth=2)

        ax2.set_xlabel('PKA Energy [eV]')
        ax2.set_ylabel('PKA Rate [PKAs/s]')
        ax2.set_title('Key Reaction Channels')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

    # Plot 3: Total PKA spectrum
    if 'total' in key_reactions:
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

    # Plot 4: Elastic vs Inelastic comparison
    if 'elastic' in key_reactions and 'inelastic' in key_reactions:
        for reaction_type in ['elastic', 'inelastic']:
            data = key_reactions[reaction_type]
            nonzero_mask = data['pka_rates'] > 0
            if np.any(nonzero_mask):
                energies = data['energies'][nonzero_mask]
                rates = data['pka_rates'][nonzero_mask]
                ax4.loglog(energies, rates, label=reaction_type,
                          color=colors[reaction_type], linewidth=2)

        ax4.set_xlabel('PKA Energy [eV]')
        ax4.set_ylabel('PKA Rate [PKAs/s]')
        ax4.set_title('Elastic vs Inelastic Scattering')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

    # Plot 5: Nuclear reactions (n,2n), (n,p), (n,α), (n,γ)
    nuclear_reactions = ['n_2n', 'n_p', 'n_alpha', 'n_gamma']
    for reaction_type in nuclear_reactions:
        if reaction_type in key_reactions:
            data = key_reactions[reaction_type]
            nonzero_mask = data['pka_rates'] > 0
            if np.any(nonzero_mask):
                energies = data['energies'][nonzero_mask]
                rates = data['pka_rates'][nonzero_mask]
                ax5.loglog(energies, rates, label=reaction_type.replace('_', ','),
                          color=colors[reaction_type], linewidth=2)

    ax5.set_xlabel('PKA Energy [eV]')
    ax5.set_ylabel('PKA Rate [PKAs/s]')
    ax5.set_title('Nuclear Reactions')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Plot 6: Reaction contributions (integrated rates)
    if key_reactions:
        reaction_names = []
        integrated_rates = []
        colors_list = []

        for reaction_type, data in key_reactions.items():
            if reaction_type != 'total':
                nonzero_mask = data['pka_rates'] > 0
                if np.any(nonzero_mask):
                    energies = data['energies'][nonzero_mask]
                    rates = data['pka_rates'][nonzero_mask]
                    integrated_rate = np.trapezoid(rates, energies)
                    if integrated_rate > 0:
                        reaction_names.append(reaction_type.replace('_', ','))
                        integrated_rates.append(integrated_rate)
                        colors_list.append(colors.get(reaction_type, 'gray'))

        if reaction_names:
            bars = ax6.bar(range(len(reaction_names)), integrated_rates, color=colors_list)
            ax6.set_xlabel('Reaction Channel')
            ax6.set_ylabel('Integrated PKA Rate [PKAs/s]')
            ax6.set_title('Reaction Channel Contributions')
            ax6.set_yscale('log')
            ax6.set_xticks(range(len(reaction_names)))
            ax6.set_xticklabels(reaction_names, rotation=45, ha='right')
            ax6.grid(True, alpha=0.3)

            # Add values on bars
            for i, (bar, rate) in enumerate(zip(bars, integrated_rates)):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1,
                        f'{rate:.1e}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('openmc_spectra_pka_results.png', dpi=300, bbox_inches='tight')
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
    flux, energies = run_openmc_simulation()

    # Step 2: Create SPECTRA-PKA flux file
    flux_filename = create_spectra_pka_flux_file(flux, energies)

    # Step 3: Create SPECTRA-PKA input file
    results_stub = "Fe56_OpenMC"
    input_filename = create_spectra_pka_input_file(flux_filename, results_stub)

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
