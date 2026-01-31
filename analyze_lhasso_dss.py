"""
Pipeline d'analyse DSS pour données LHASSO + GRB réelles
=========================================================

Ce script peut être appliqué directement aux données LHASSO réelles
une fois que vous y avez accès.

Usage:
------
python analyze_lhasso_dss.py --grb-data <path> --output <path>

Format des données d'entrée:
----------------------------
- FITS ou HDF5 avec colonnes: energy, time, instrument
- CSV avec colonnes: energy_GeV, arrival_time_s, detector

Auteur: Pipeline développé avec Claude
Date: 2026-01-30
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import argparse
import json
import sys

try:
    from astropy.io import fits
    FITS_AVAILABLE = True
except ImportError:
    FITS_AVAILABLE = False
    print("Warning: astropy not available, FITS support disabled")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available, CSV support limited")


class DSSAnalysisPipeline:
    """
    Pipeline complet pour l'analyse DSS de données GRB multi-énergies
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.photons = None
        self.results = {}
        
    def load_data(self, filepath, format='auto'):
        """
        Charge les données de photons depuis différents formats
        
        Parameters:
        -----------
        filepath : str
            Chemin vers le fichier de données
        format : str
            'auto', 'fits', 'hdf5', 'csv', 'npy'
        """
        if format == 'auto':
            if filepath.endswith('.fits'):
                format = 'fits'
            elif filepath.endswith(('.h5', '.hdf5')):
                format = 'hdf5'
            elif filepath.endswith('.csv'):
                format = 'csv'
            elif filepath.endswith('.npy'):
                format = 'npy'
            else:
                raise ValueError(f"Cannot determine format for {filepath}")
        
        if self.verbose:
            print(f"Loading data from {filepath} (format: {format})...")
        
        if format == 'fits':
            if not FITS_AVAILABLE:
                raise ImportError("astropy required for FITS format")
            with fits.open(filepath) as hdul:
                data = hdul[1].data
                self.photons = {
                    'energy_GeV': np.array(data['ENERGY']),
                    'time_s': np.array(data['TIME']),
                    'detector': np.array(data['DETECTOR']) if 'DETECTOR' in data.names else None
                }
        
        elif format == 'csv':
            if PANDAS_AVAILABLE:
                df = pd.read_csv(filepath)
                self.photons = {
                    'energy_GeV': df['energy_GeV'].values,
                    'time_s': df['time_s'].values,
                    'detector': df['detector'].values if 'detector' in df.columns else None
                }
            else:
                data = np.genfromtxt(filepath, names=True, delimiter=',')
                self.photons = {
                    'energy_GeV': data['energy_GeV'],
                    'time_s': data['time_s'],
                    'detector': None
                }
        
        elif format == 'npy':
            data = np.load(filepath, allow_pickle=True).item()
            self.photons = data
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        n_photons = len(self.photons['energy_GeV'])
        if self.verbose:
            print(f"  Loaded {n_photons} photons")
            print(f"  Energy range: {self.photons['energy_GeV'].min():.2e} - {self.photons['energy_GeV'].max():.2e} GeV")
            print(f"  Time range: {self.photons['time_s'].min():.3f} - {self.photons['time_s'].max():.3f} s")
        
        return self.photons
    
    def preprocess(self, energy_min=None, energy_max=None, time_min=None, time_max=None):
        """
        Filtre et nettoie les données
        """
        if self.photons is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        mask = np.ones(len(self.photons['energy_GeV']), dtype=bool)
        
        if energy_min is not None:
            mask &= self.photons['energy_GeV'] >= energy_min
        if energy_max is not None:
            mask &= self.photons['energy_GeV'] <= energy_max
        if time_min is not None:
            mask &= self.photons['time_s'] >= time_min
        if time_max is not None:
            mask &= self.photons['time_s'] <= time_max
        
        n_before = len(self.photons['energy_GeV'])
        
        for key in self.photons:
            if self.photons[key] is not None:
                self.photons[key] = self.photons[key][mask]
        
        n_after = len(self.photons['energy_GeV'])
        
        if self.verbose:
            print(f"Preprocessing: {n_before} → {n_after} photons ({n_before - n_after} removed)")
        
        return self.photons
    
    def compute_time_energy_correlation(self, n_bins=10, method='median'):
        """
        Calcule la corrélation temps-énergie par binning
        
        Parameters:
        -----------
        n_bins : int
            Nombre de bins en énergie
        method : str
            'median' ou 'mean'
        """
        energies = self.photons['energy_GeV']
        times = self.photons['time_s']
        
        log_E = np.log10(energies)
        bins = np.linspace(log_E.min(), log_E.max(), n_bins + 1)
        
        bin_energies = []
        bin_times = []
        bin_errors = []
        bin_n_photons = []
        
        for i in range(n_bins):
            mask = (log_E >= bins[i]) & (log_E < bins[i+1])
            n_in_bin = np.sum(mask)
            
            if n_in_bin >= 3:  # au moins 3 photons
                if method == 'median':
                    central_time = np.median(times[mask])
                    central_energy = np.median(energies[mask])
                else:
                    central_time = np.mean(times[mask])
                    central_energy = np.mean(energies[mask])
                
                # Erreur = écart-type / sqrt(n)
                time_error = np.std(times[mask]) / np.sqrt(n_in_bin)
                
                bin_energies.append(central_energy)
                bin_times.append(central_time)
                bin_errors.append(time_error)
                bin_n_photons.append(n_in_bin)
        
        self.results['binned_data'] = {
            'energies': np.array(bin_energies),
            'times': np.array(bin_times),
            'errors': np.array(bin_errors),
            'n_photons': np.array(bin_n_photons)
        }
        
        if self.verbose:
            print(f"Computed {len(bin_energies)} energy bins for correlation analysis")
        
        return self.results['binned_data']
    
    def fit_dss_model(self):
        """
        Ajuste le modèle DSS: t(E) = t0 + A/E
        
        Returns:
        --------
        fit_results : dict
            Paramètres ajustés et statistiques
        """
        if 'binned_data' not in self.results:
            raise ValueError("Must call compute_time_energy_correlation() first")
        
        E = self.results['binned_data']['energies']
        t = self.results['binned_data']['times']
        t_err = self.results['binned_data']['errors']
        
        # Modèle: t = t0 + A/E
        def model(E, t0, A):
            return t0 + A / E
        
        try:
            # Ajustement avec poids
            popt, pcov = curve_fit(model, E, t, sigma=t_err, absolute_sigma=True)
            t0, A = popt
            t0_err, A_err = np.sqrt(np.diag(pcov))
            
            # Chi2
            t_pred = model(E, *popt)
            chi2_val = np.sum(((t - t_pred) / t_err)**2)
            ndof = len(E) - 2
            p_value = 1 - chi2.cdf(chi2_val, ndof)
            
            # Signification statistique
            if A_err > 0:
                significance = abs(A) / A_err
            else:
                significance = 0
            
            fit_results = {
                't0': t0,
                't0_err': t0_err,
                'A': A,  # s·GeV
                'A_err': A_err,
                'chi2': chi2_val,
                'ndof': ndof,
                'chi2_reduced': chi2_val / ndof if ndof > 0 else np.inf,
                'p_value': p_value,
                'significance_sigma': significance,
                'fit_success': True
            }
            
            # Conversion en paramètres DSS
            # A = gamma_zeta / lambda_DSS × facteur_géométrique × D/c
            # Pour estimer gamma_zeta/lambda_DSS, il faut connaître D
            
        except Exception as e:
            if self.verbose:
                print(f"Fit failed: {e}")
            fit_results = {
                't0': np.nan,
                't0_err': np.nan,
                'A': np.nan,
                'A_err': np.nan,
                'chi2': np.nan,
                'ndof': len(E) - 2,
                'chi2_reduced': np.nan,
                'p_value': np.nan,
                'significance_sigma': 0,
                'fit_success': False
            }
        
        self.results['fit'] = fit_results
        
        if self.verbose and fit_results['fit_success']:
            print("\n=== DSS Fit Results ===")
            print(f"  t0 = {t0:.6f} ± {t0_err:.6f} s")
            print(f"  A = {A:.6e} ± {A_err:.6e} s·GeV")
            print(f"  χ²/ndof = {chi2_val:.2f}/{ndof} = {chi2_val/ndof:.2f}")
            print(f"  p-value = {p_value:.4f}")
            print(f"  Significance = {significance:.1f}σ")
            
            if significance > 3:
                print(f"  *** DETECTION SIGNIFICATIVE (>{significance:.1f}σ) ***")
            elif significance > 2:
                print(f"  ** Indice possible ({significance:.1f}σ) **")
            else:
                print(f"  No significant DSS signal detected")
        
        return fit_results
    
    def plot_results(self, output_path=None, show=True):
        """
        Génère les graphiques de l'analyse
        """
        if 'binned_data' not in self.results or 'fit' not in self.results:
            raise ValueError("Must run analysis first")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Panel 1: Time vs Energy (log scale)
        ax1 = axes[0]
        E = self.results['binned_data']['energies']
        t = self.results['binned_data']['times']
        t_err = self.results['binned_data']['errors']
        
        ax1.errorbar(E, t, yerr=t_err, fmt='o', color='blue', 
                    markersize=8, capsize=5, label='Données')
        
        if self.results['fit']['fit_success']:
            E_model = np.logspace(np.log10(E.min()), np.log10(E.max()), 100)
            t0 = self.results['fit']['t0']
            A = self.results['fit']['A']
            t_model = t0 + A / E_model
            
            ax1.plot(E_model, t_model, 'r-', linewidth=2,
                    label=f"DSS fit: A={A*1e3:.2f}±{self.results['fit']['A_err']*1e3:.2f} ms·GeV")
        
        ax1.set_xscale('log')
        ax1.set_xlabel('Énergie (GeV)', fontsize=12)
        ax1.set_ylabel('Temps d\'arrivée (s)', fontsize=12)
        ax1.set_title('Corrélation Temps-Énergie', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Résidus
        ax2 = axes[1]
        if self.results['fit']['fit_success']:
            t_pred = t0 + A / E
            residuals = (t - t_pred) / t_err
            
            ax2.errorbar(E, residuals, yerr=1, fmt='o', color='green',
                        markersize=8, capsize=5)
            ax2.axhline(0, color='red', linestyle='--', linewidth=2)
            ax2.axhline(3, color='orange', linestyle=':', linewidth=1, alpha=0.5)
            ax2.axhline(-3, color='orange', linestyle=':', linewidth=1, alpha=0.5)
            
            ax2.set_xscale('log')
            ax2.set_xlabel('Énergie (GeV)', fontsize=12)
            ax2.set_ylabel('Résidus normalisés (σ)', fontsize=12)
            ax2.set_title('Résidus du fit DSS', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Stats
            textstr = f"χ²/ndof = {self.results['fit']['chi2_reduced']:.2f}\n"
            textstr += f"p-value = {self.results['fit']['p_value']:.3f}\n"
            textstr += f"Signif. = {self.results['fit']['significance_sigma']:.1f}σ"
            ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Figure saved to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def save_results(self, output_path):
        """
        Sauvegarde les résultats au format JSON
        """
        # Convertir numpy arrays en listes pour JSON
        results_serializable = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        results_serializable[key][k] = v.tolist()
                    elif isinstance(v, (np.integer, np.floating)):
                        results_serializable[key][k] = float(v)
                    else:
                        results_serializable[key][k] = v
            else:
                results_serializable[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyse DSS pour données LHASSO + GRB'
    )
    parser.add_argument('--input', required=True, help='Fichier de données d\'entrée')
    parser.add_argument('--format', default='auto', help='Format: auto, fits, csv, npy')
    parser.add_argument('--output', default='dss_analysis', help='Préfixe pour fichiers de sortie')
    parser.add_argument('--n-bins', type=int, default=10, help='Nombre de bins en énergie')
    parser.add_argument('--energy-min', type=float, help='Énergie minimum (GeV)')
    parser.add_argument('--energy-max', type=float, help='Énergie maximum (GeV)')
    parser.add_argument('--time-min', type=float, help='Temps minimum (s)')
    parser.add_argument('--time-max', type=float, help='Temps maximum (s)')
    parser.add_argument('--no-plot', action='store_true', help='Désactiver les graphiques')
    
    args = parser.parse_args()
    
    # Initialisation du pipeline
    pipeline = DSSAnalysisPipeline(verbose=True)
    
    # Chargement des données
    try:
        pipeline.load_data(args.input, format=args.format)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Prétraitement
    pipeline.preprocess(
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        time_min=args.time_min,
        time_max=args.time_max
    )
    
    # Analyse
    pipeline.compute_time_energy_correlation(n_bins=args.n_bins)
    pipeline.fit_dss_model()
    
    # Visualisation
    if not args.no_plot:
        pipeline.plot_results(
            output_path=f"{args.output}_plot.png",
            show=False
        )
    
    # Sauvegarde
    pipeline.save_results(f"{args.output}_results.json")
    
    print("\n=== Analyse terminée avec succès ===")


if __name__ == '__main__':
    main()
