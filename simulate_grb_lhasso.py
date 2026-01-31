"""
Simulation réaliste de données GRB multi-longueurs d'onde
pour tester le pipeline d'analyse DSS
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.optimize import curve_fit
import json

# Paramètres physiques
c = 2.998e8  # m/s
h_planck = 6.626e-34  # J·s
eV_to_J = 1.602e-19

class GRBSimulator:
    def __init__(self, redshift, grb_duration, dss_params=None):
        """
        Simulateur de GRB avec ou sans effet DSS
        
        Parameters:
        -----------
        redshift : float
            Redshift cosmologique du GRB
        grb_duration : float
            Durée du GRB en secondes
        dss_params : dict ou None
            Si dict: {'gamma_zeta': float, 'lambda_DSS': float}
            Si None: simulation sans DSS (GR pure)
        """
        self.z = redshift
        self.duration = grb_duration
        self.dss_params = dss_params
        
        # Distance lumineuse (approximation)
        self.D_L = self._luminosity_distance(redshift)  # en mètres
        
    def _luminosity_distance(self, z):
        """Distance lumineuse simplifiée (flat ΛCDM)"""
        H0 = 70  # km/s/Mpc
        H0_SI = H0 * 1e3 / (3.086e22)  # conversion en SI
        # Approximation premier ordre
        D_L = (c / H0_SI) * z * (1 + z/2)
        return D_L
    
    def _intrinsic_lightcurve(self, times, peak_time=1.0):
        """
        Courbe de lumière intrinsèque du GRB (modèle FRED)
        Fast Rise Exponential Decay
        """
        tau_rise = 0.1
        tau_decay = 2.0
        
        rise = np.exp(-(peak_time - times) / tau_rise)
        decay = np.exp(-(times - peak_time) / tau_decay)
        
        flux = np.where(times < peak_time, rise, decay)
        flux[times < 0] = 0
        
        return flux / np.max(flux)
    
    def _dss_time_delay(self, energy_eV):
        """
        Calcul du délai DSS en fonction de l'énergie
        
        Returns: délai en secondes (positif = arrivée retardée)
        """
        if self.dss_params is None:
            return 0.0
        
        gamma_zeta = self.dss_params['gamma_zeta']
        lambda_DSS = self.dss_params['lambda_DSS']
        
        # Longueur d'onde observée
        wavelength = (h_planck * c) / (energy_eV * eV_to_J) * (1 + self.z)
        
        # Facteur géométrique (intégrale du potentiel gravitationnel)
        # Pour simplification: ~10^-4 pour ligne de visée cosmologique
        geom_factor = 1e-4
        
        # Délai DSS
        delay = gamma_zeta * (wavelength / lambda_DSS) * geom_factor * (self.D_L / c)
        
        return delay
    
    def generate_photons(self, energy_bands, n_photons_per_band):
        """
        Génère des photons détectés dans différentes bandes d'énergie
        
        Parameters:
        -----------
        energy_bands : list of tuples
            [(E_min, E_max, name), ...] en eV
        n_photons_per_band : list of int
            Nombre de photons à générer par bande
        
        Returns:
        --------
        photons : list of dicts
            [{'energy': E, 'time': t, 'band': name}, ...]
        """
        photons = []
        
        for (E_min, E_max, band_name), n_photons in zip(energy_bands, n_photons_per_band):
            # Énergies uniformément distribuées dans la bande (log-scale)
            log_E = np.random.uniform(np.log10(E_min), np.log10(E_max), n_photons)
            energies = 10**log_E
            
            # Temps d'émission intrinsèque
            t_emission = np.random.exponential(scale=self.duration/3, size=n_photons)
            t_emission = t_emission[t_emission < self.duration * 3]  # coupure
            n_actual = len(t_emission)
            energies = energies[:n_actual]
            
            # Application courbe de lumière
            lc_weights = self._intrinsic_lightcurve(t_emission)
            # Échantillonnage selon courbe de lumière
            keep = np.random.random(n_actual) < lc_weights
            t_emission = t_emission[keep]
            energies = energies[keep]
            
            # Délai DSS
            for E, t_em in zip(energies, t_emission):
                delay = self._dss_time_delay(E)
                t_arrival = t_em + delay
                
                # Incertitude de mesure temporelle (dépend de l'instrument)
                if E > 1e11:  # > 100 GeV (LHASSO)
                    t_error = np.random.normal(0, 1e-9)  # 1 ns
                elif E > 1e9:  # > 1 GeV (Fermi-LAT)
                    t_error = np.random.normal(0, 1e-6)  # 1 μs
                else:  # keV-MeV (Fermi-GBM, Swift)
                    t_error = np.random.normal(0, 1e-3)  # 1 ms
                
                photons.append({
                    'energy_eV': E,
                    'energy_GeV': E / 1e9,
                    'time': t_arrival + t_error,
                    'time_emission': t_em,
                    'dss_delay': delay,
                    'band': band_name
                })
        
        return photons


def analyze_time_delays(photons):
    """
    Analyse les délais temporels pour extraire les paramètres DSS
    """
    photons = sorted(photons, key=lambda p: p['energy_eV'])
    
    energies = np.array([p['energy_GeV'] for p in photons])
    times = np.array([p['time'] for p in photons])
    
    # Binning en énergie
    n_bins = 10
    log_E = np.log10(energies)
    bins = np.linspace(log_E.min(), log_E.max(), n_bins + 1)
    
    median_times = []
    median_energies = []
    time_errors = []
    
    for i in range(n_bins):
        mask = (log_E >= bins[i]) & (log_E < bins[i+1])
        if np.sum(mask) > 5:  # au moins 5 photons
            median_times.append(np.median(times[mask]))
            median_energies.append(np.median(energies[mask]))
            # Erreur = écart-type de la médiane
            time_errors.append(np.std(times[mask]) / np.sqrt(np.sum(mask)))
    
    median_times = np.array(median_times)
    median_energies = np.array(median_energies)
    time_errors = np.array(time_errors)
    
    # Ajustement linéaire: t = t0 + A/E
    def model(E, t0, A):
        return t0 + A / E
    
    try:
        popt, pcov = curve_fit(model, median_energies, median_times, 
                              sigma=time_errors, absolute_sigma=True)
        t0_fit, A_fit = popt
        t0_err, A_err = np.sqrt(np.diag(pcov))
        
        # Calcul du chi2
        chi2 = np.sum(((median_times - model(median_energies, *popt)) / time_errors)**2)
        ndof = len(median_times) - 2
        
        fit_success = True
    except:
        t0_fit, A_fit = np.nan, np.nan
        t0_err, A_err = np.nan, np.nan
        chi2, ndof = np.nan, np.nan
        fit_success = False
    
    return {
        'median_energies': median_energies,
        'median_times': median_times,
        'time_errors': time_errors,
        't0': t0_fit,
        't0_err': t0_err,
        'A': A_fit,  # coefficient de 1/E
        'A_err': A_err,
        'chi2': chi2,
        'ndof': ndof,
        'fit_success': fit_success
    }


# Simulation de 3 GRB
print("=== Simulation de GRB pour test pipeline DSS ===\n")

# Définition des bandes d'énergie
energy_bands = [
    (1e4, 1e6, 'Swift-BAT'),       # 10 keV - 1 MeV
    (1e6, 1e8, 'Fermi-GBM'),       # 1 MeV - 100 MeV
    (1e8, 3e11, 'Fermi-LAT'),      # 100 MeV - 300 GeV
    (1e11, 1e12, 'LHASSO')         # 100 GeV - 1 TeV
]

results = []

# GRB 1: Sans DSS (contrôle)
print("GRB 1: Simulation GR pure (pas de DSS)")
grb1 = GRBSimulator(redshift=1.5, grb_duration=2.0, dss_params=None)
photons1 = grb1.generate_photons(energy_bands, [500, 300, 100, 30])
analysis1 = analyze_time_delays(photons1)
print(f"  Photons générés: {len(photons1)}")
print(f"  Coefficient A (1/E): {analysis1['A']:.6e} ± {analysis1['A_err']:.6e} s·GeV")
print(f"  Chi2/ndof: {analysis1['chi2']:.2f}/{analysis1['ndof']}")
print()

# GRB 2: Avec DSS (signal modéré)
print("GRB 2: Simulation avec DSS (gamma_zeta=1e-4, lambda_DSS=1mm)")
grb2 = GRBSimulator(
    redshift=2.0, 
    grb_duration=1.5,
    dss_params={'gamma_zeta': 1e-4, 'lambda_DSS': 1e-3}
)
photons2 = grb2.generate_photons(energy_bands, [600, 400, 150, 40])
analysis2 = analyze_time_delays(photons2)
print(f"  Photons générés: {len(photons2)}")
print(f"  Coefficient A (1/E): {analysis2['A']:.6e} ± {analysis2['A_err']:.6e} s·GeV")
print(f"  Chi2/ndof: {analysis2['chi2']:.2f}/{analysis2['ndof']}")
print()

# GRB 3: Avec DSS fort (pour validation)
print("GRB 3: Simulation avec DSS fort (gamma_zeta=5e-4, lambda_DSS=1mm)")
grb3 = GRBSimulator(
    redshift=1.8,
    grb_duration=3.0,
    dss_params={'gamma_zeta': 5e-4, 'lambda_DSS': 1e-3}
)
photons3 = grb3.generate_photons(energy_bands, [800, 500, 200, 50])
analysis3 = analyze_time_delays(photons3)
print(f"  Photons générés: {len(photons3)}")
print(f"  Coefficient A (1/E): {analysis3['A']:.6e} ± {analysis3['A_err']:.6e} s·GeV")
print(f"  Chi2/ndof: {analysis3['chi2']:.2f}/{analysis3['ndof']}")
print()

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (photons, analysis, title) in enumerate([
    (photons1, analysis1, 'GRB 1: GR (pas de DSS)'),
    (photons2, analysis2, 'GRB 2: DSS modéré'),
    (photons3, analysis3, 'GRB 3: DSS fort')
]):
    ax = axes[idx]
    
    # Plot des données
    ax.errorbar(analysis['median_energies'], analysis['median_times'], 
                yerr=analysis['time_errors'], fmt='o', color='blue', 
                label='Données', markersize=8, capsize=5)
    
    # Plot du fit
    if analysis['fit_success']:
        E_model = np.logspace(np.log10(analysis['median_energies'].min()),
                             np.log10(analysis['median_energies'].max()), 100)
        t_model = analysis['t0'] + analysis['A'] / E_model
        ax.plot(E_model, t_model, 'r-', linewidth=2, 
               label=f"Fit: A={(analysis['A']*1e3):.2f}±{(analysis['A_err']*1e3):.2f} ms·GeV")
    
    ax.set_xscale('log')
    ax.set_xlabel('Énergie (GeV)', fontsize=11)
    ax.set_ylabel('Temps d\'arrivée (s)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/grb_dss_simulation.png', dpi=300, bbox_inches='tight')
print("Figure sauvegardée: grb_dss_simulation.png\n")

# Sauvegarde des résultats
results_summary = {
    'GRB1_no_DSS': {
        'n_photons': len(photons1),
        'A_coefficient': float(analysis1['A']),
        'A_error': float(analysis1['A_err']),
        'chi2_ndof': float(analysis1['chi2']) / analysis1['ndof'] if analysis1['ndof'] > 0 else np.nan
    },
    'GRB2_DSS_moderate': {
        'n_photons': len(photons2),
        'A_coefficient': float(analysis2['A']),
        'A_error': float(analysis2['A_err']),
        'chi2_ndof': float(analysis2['chi2']) / analysis2['ndof'] if analysis2['ndof'] > 0 else np.nan,
        'input_gamma_zeta': 1e-4,
        'input_lambda_DSS': 1e-3
    },
    'GRB3_DSS_strong': {
        'n_photons': len(photons3),
        'A_coefficient': float(analysis3['A']),
        'A_error': float(analysis3['A_err']),
        'chi2_ndof': float(analysis3['chi2']) / analysis3['ndof'] if analysis3['ndof'] > 0 else np.nan,
        'input_gamma_zeta': 5e-4,
        'input_lambda_DSS': 1e-3
    }
}

with open('/home/claude/grb_simulation_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("=== Analyse terminée ===")
print("Résultats sauvegardés dans grb_simulation_results.json")
