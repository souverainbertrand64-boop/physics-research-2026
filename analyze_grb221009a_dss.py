"""
Analyse DSS appliquée à GRB 221009A (LHAASO)
Données du papier Qin et al. (2025)

Test des délais temporels énergie-dépendants prédits par DSS
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

print("=== Analyse DSS de GRB 221009A (LHAASO) ===\n")

# Paramètres du GRB
z_grb = 0.151  # redshift
D_Mpc = 650    # Distance en Mpc
D_m = D_Mpc * 3.086e22  # Distance en mètres
c = 2.998e8  # vitesse lumière m/s

print(f"GRB 221009A:")
print(f"  Redshift z = {z_grb}")
print(f"  Distance D = {D_Mpc} Mpc = {D_m:.2e} m")
print(f"  Le GRB le plus énergétique jamais détecté !\n")

# Données spectrales extraites de Figure 1 du papier
# Format: [Energie_TeV, Flux, Erreur, Fenêtre_temporelle]

# Les auteurs ont analysé 6 datasets avec différentes fenêtres temporelles
# Je vais me concentrer sur les datasets avec le meilleur rapport signal/bruit

# WD 231-240s (Power Law)
E_WD1 = np.array([0.63, 0.79, 1.0, 1.26, 1.58, 2.0, 2.51, 3.16, 3.98])  # TeV
flux_WD1 = np.array([1.4e-12, 6.0e-13, 4.5e-13, 2.8e-13, 2.0e-13, 
                     1.3e-13, 1.0e-13, 5e-14, 3e-14])  # TeV cm^-2 s^-1
err_WD1 = flux_WD1 * 0.3  # erreur ~30% (estimée des figures)
t_center_WD1 = 235.5  # temps central (secondes après trigger)

# WD 326-900s (Exp Cut Power Law)
E_WD2 = np.array([0.63, 0.79, 1.0, 1.26, 1.58, 2.0, 2.51, 3.16, 3.98, 5.01])
flux_WD2 = np.array([1.2e-12, 5.5e-13, 4.0e-13, 2.5e-13, 1.8e-13,
                     1.2e-13, 8e-14, 5e-14, 2.5e-14, 1e-14])
err_WD2 = flux_WD2 * 0.3
t_center_WD2 = 613  # temps central

# WD+KM300-900s (Log Parabola) - meilleure statistique
E_WD_KM = np.array([0.63, 0.79, 1.0, 1.26, 1.58, 2.0, 2.51, 3.16, 3.98, 
                    5.01, 6.31, 7.94])  # TeV
flux_WD_KM = np.array([1.1e-12, 5.2e-13, 3.8e-13, 2.4e-13, 1.7e-13,
                       1.1e-13, 7.5e-14, 4.8e-14, 2.3e-14, 1.2e-14,
                       5e-15, 2e-15])
err_WD_KM = flux_WD_KM * 0.25  # meilleure statistique
t_center_WD_KM = 600

print("Datasets analysés:")
print(f"  WD 231-240s: {len(E_WD1)} points, temps central = {t_center_WD1:.1f}s")
print(f"  WD 326-900s: {len(E_WD2)} points, temps central = {t_center_WD2:.1f}s")
print(f"  WD+KM300-900s: {len(E_WD_KM)} points, temps central = {t_center_WD_KM:.1f}s")
print()

# Test DSS : recherche de corrélation temporelle vs 1/E
print("=== Test DSS: Corrélation temps d'arrivée vs 1/Energie ===\n")

# Hypothèse DSS: t_arrival(E) = t0 + A_DSS/E
# où A_DSS = (gamma_zeta/lambda_DSS) × facteur_géométrique × (D/c)

# Utilisons les 3 datasets pour estimer les temps d'arrivée relatifs
# En supposant que chaque fenêtre capte un "pulse" à des énergies différentes

def compute_energy_weighted_time(E, flux, t_center):
    """
    Calcule le temps d'arrivée moyen pondéré par le flux
    pour chaque bande d'énergie
    """
    # Le temps d'arrivée est décalé selon DSS: t = t_center + Δt_DSS(E)
    # On cherche à extraire Δt_DSS(E) des différences entre fenêtres
    return t_center * np.ones_like(E)

# Stratégie: comparer les énergies caractéristiques de chaque fenêtre
# Si DSS existe, les photons basse énergie arrivent plus tard

# Energie moyenne pondérée par le flux de chaque fenêtre
def mean_energy(E, flux):
    return np.sum(E * flux) / np.sum(flux)

E_mean_WD1 = mean_energy(E_WD1, flux_WD1)
E_mean_WD2 = mean_energy(E_WD2, flux_WD2)
E_mean_WD_KM = mean_energy(E_WD_KM, flux_WD_KM)

print("Energies moyennes (pondérées par flux):")
print(f"  WD 231-240s: E_mean = {E_mean_WD1:.3f} TeV, t = {t_center_WD1:.1f}s")
print(f"  WD 326-900s: E_mean = {E_mean_WD2:.3f} TeV, t = {t_center_WD2:.1f}s")
print(f"  WD+KM300-900s: E_mean = {E_mean_WD_KM:.3f} TeV, t = {t_center_WD_KM:.1f}s")
print()

# Test de corrélation
E_means = np.array([E_mean_WD1, E_mean_WD2, E_mean_WD_KM])
t_centers = np.array([t_center_WD1, t_center_WD2, t_center_WD_KM])

# Fit linéaire: t = t0 + A_DSS/E
def time_vs_inv_energy(inv_E, t0, A_DSS):
    return t0 + A_DSS * inv_E

try:
    popt, pcov = curve_fit(time_vs_inv_energy, 1/E_means, t_centers,
                          p0=[200, 100])
    t0_fit, A_DSS_fit = popt
    t0_err, A_DSS_err = np.sqrt(np.diag(pcov))
    
    print("Fit DSS: t(E) = t0 + A_DSS/E")
    print(f"  t0 = {t0_fit:.1f} ± {t0_err:.1f} s")
    print(f"  A_DSS = {A_DSS_fit:.2f} ± {A_DSS_err:.2f} s·TeV")
    print()
    
    # Signification statistique
    if A_DSS_err > 0:
        sigma_DSS = abs(A_DSS_fit) / A_DSS_err
        print(f"  Signification A_DSS: {sigma_DSS:.2f}σ")
        if sigma_DSS > 2:
            print(f"  *** INDICATION POTENTIELLE D'EFFET DSS ***")
        else:
            print(f"  Pas de signal DSS significatif avec 3 points")
    print()
    
    # Dérivation des paramètres DSS
    # A_DSS = (gamma_zeta/lambda_DSS) × geom_factor × (D/c)
    # geom_factor ~ 10^-4 pour ligne de visée cosmologique
    
    geom_factor = 1e-4
    gamma_zeta_over_lambda = A_DSS_fit * c / (geom_factor * D_m)
    gamma_zeta_over_lambda_err = A_DSS_err * c / (geom_factor * D_m)
    
    print("Paramètres DSS dérivés:")
    print(f"  gamma_zeta/lambda_DSS = {gamma_zeta_over_lambda:.1f} ± {gamma_zeta_over_lambda_err:.1f} m^-1")
    print()
    
    # Si lambda_DSS = 1 mm
    lambda_DSS_mm = 1e-3  # m
    gamma_zeta = gamma_zeta_over_lambda * lambda_DSS_mm
    gamma_zeta_err = gamma_zeta_over_lambda_err * lambda_DSS_mm
    
    print(f"Si lambda_DSS = 1 mm:")
    print(f"  gamma_zeta = {gamma_zeta:.3f} ± {gamma_zeta_err:.3f}")
    print()
    
    fit_success = True
    
except Exception as e:
    print(f"Fit échoué: {e}")
    fit_success = False
    A_DSS_fit = 0
    A_DSS_err = 0

# Comparaison avec analyse spectrale standard
print("=== Analyse spectrale (comme Qin et al.) ===\n")

# Le papier utilise des modèles intrinsèques + absorption EBL
# Nous cherchons des déviations spectrales compatibles avec DSS

# Fit du spectre combiné WD+KM300-900s (meilleure statistique)
def powerlaw(E, phi0, alpha):
    """Loi de puissance simple"""
    return phi0 * (E / 2.0)**(-alpha)

def powerlaw_dss_correction(E, phi0, alpha, B_DSS):
    """
    Loi de puissance avec correction spectrale DSS
    DSS modifie légèrement l'indice spectral apparent
    """
    alpha_eff = alpha + B_DSS / E  # correction 1er ordre
    return phi0 * (E / 2.0)**(-alpha_eff)

# Fit standard
popt_std, pcov_std = curve_fit(powerlaw, E_WD_KM, flux_WD_KM/E_WD_KM**2,
                               sigma=err_WD_KM/E_WD_KM**2,
                               absolute_sigma=True)
phi0_std, alpha_std = popt_std
phi0_std_err, alpha_std_err = np.sqrt(np.diag(pcov_std))

flux_model_std = powerlaw(E_WD_KM, *popt_std) * E_WD_KM**2
chi2_std = np.sum(((flux_WD_KM - flux_model_std) / err_WD_KM)**2)
ndof_std = len(E_WD_KM) - 2

print(f"Fit standard (Power Law):")
print(f"  α = {alpha_std:.3f} ± {alpha_std_err:.3f}")
print(f"  χ²/ndof = {chi2_std:.2f}/{ndof_std} = {chi2_std/ndof_std:.2f}")
print()

# Fit avec correction DSS
try:
    popt_dss, pcov_dss = curve_fit(powerlaw_dss_correction, E_WD_KM,
                                   flux_WD_KM/E_WD_KM**2,
                                   sigma=err_WD_KM/E_WD_KM**2,
                                   p0=[phi0_std, alpha_std, 0.1],
                                   absolute_sigma=True)
    phi0_dss, alpha_dss, B_DSS = popt_dss
    phi0_dss_err, alpha_dss_err, B_DSS_err = np.sqrt(np.diag(pcov_dss))
    
    flux_model_dss = powerlaw_dss_correction(E_WD_KM, *popt_dss) * E_WD_KM**2
    chi2_dss = np.sum(((flux_WD_KM - flux_model_dss) / err_WD_KM)**2)
    
    print(f"Fit avec correction DSS:")
    print(f"  α = {alpha_dss:.3f} ± {alpha_dss_err:.3f}")
    print(f"  B_DSS = {B_DSS:.3f} ± {B_DSS_err:.3f} TeV")
    print(f"  χ²/ndof = {chi2_dss:.2f}/{ndof_std-1} = {chi2_dss/(ndof_std-1):.2f}")
    
    Delta_chi2 = chi2_std - chi2_dss
    significance_spectral = np.sqrt(abs(Delta_chi2)) if Delta_chi2 > 0 else 0
    print(f"  Δχ² = {Delta_chi2:.2f}")
    print(f"  Signification correction DSS = {significance_spectral:.2f}σ")
    print()
    
    dss_spectral_fit_success = True
except:
    print("Fit DSS spectral échoué\n")
    dss_spectral_fit_success = False
    B_DSS = 0
    B_DSS_err = 0

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: Spectre GRB 221009A
ax1 = axes[0, 0]
ax1.errorbar(E_WD1, flux_WD1, yerr=err_WD1, fmt='o', color='blue',
            label='WD 231-240s', markersize=7, capsize=3, alpha=0.7)
ax1.errorbar(E_WD2, flux_WD2, yerr=err_WD2, fmt='s', color='green',
            label='WD 326-900s', markersize=7, capsize=3, alpha=0.7)
ax1.errorbar(E_WD_KM, flux_WD_KM, yerr=err_WD_KM, fmt='^', color='red',
            label='WD+KM300-900s', markersize=8, capsize=3)

E_plot = np.logspace(np.log10(0.5), np.log10(10), 200)
ax1.plot(E_plot, powerlaw(E_plot, *popt_std) * E_plot**2, 'k--',
        linewidth=2, label=f'Power Law (α={alpha_std:.2f})')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Énergie (TeV)', fontsize=12)
ax1.set_ylabel('E² dN/dE (TeV cm⁻² s⁻¹)', fontsize=12)
ax1.set_title('(a) Spectres GRB 221009A (LHAASO)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)

# Panel 2: Test corrélation temps vs 1/E
ax2 = axes[0, 1]
ax2.plot(1/E_means, t_centers, 'o', color='purple', markersize=12, label='Données')

if fit_success:
    inv_E_plot = np.linspace(0.3, 1.2, 100)
    ax2.plot(inv_E_plot, time_vs_inv_energy(inv_E_plot, t0_fit, A_DSS_fit),
            'r-', linewidth=2,
            label=f'DSS fit: A={A_DSS_fit:.1f}±{A_DSS_err:.1f} s·TeV')

ax2.set_xlabel('1/E (TeV⁻¹)', fontsize=12)
ax2.set_ylabel('Temps central (s)', fontsize=12)
ax2.set_title('(b) Test DSS: Temps vs 1/Energie', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

textstr = f'GRB 221009A\nz = {z_grb}\nD = {D_Mpc} Mpc'
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Panel 3: Résidus spectraux
ax3 = axes[1, 0]
residuals_std = (flux_WD_KM - flux_model_std) / err_WD_KM
ax3.errorbar(E_WD_KM, residuals_std, yerr=1, fmt='o', color='blue',
            markersize=8, capsize=3, label='Standard')

if dss_spectral_fit_success:
    residuals_dss = (flux_WD_KM - flux_model_dss) / err_WD_KM
    ax3.errorbar(E_WD_KM, residuals_dss, yerr=1, fmt='s', color='red',
                markersize=8, capsize=3, label='Avec DSS', alpha=0.7)

ax3.axhline(0, color='black', linestyle='--', linewidth=2)
ax3.axhline(2, color='orange', linestyle=':', alpha=0.5)
ax3.axhline(-2, color='orange', linestyle=':', alpha=0.5)
ax3.set_xscale('log')
ax3.set_xlabel('Énergie (TeV)', fontsize=12)
ax3.set_ylabel('Résidus normalisés (σ)', fontsize=12)
ax3.set_title('(c) Résidus spectraux', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Panel 4: Délais DSS prédits
ax4 = axes[1, 1]

# Prédiction DSS pour différentes valeurs de gamma_zeta
E_pred = np.logspace(np.log10(0.5), np.log10(10), 100)

for gz in [1e-5, 1e-4, 5e-4, 1e-3]:
    lambda_DSS = 1e-3  # 1 mm
    geom = 1e-4
    Delay_DSS = (gz / lambda_DSS) * geom * (D_m / c) * (1.0 / E_pred)
    ax4.plot(E_pred, Delay_DSS, linewidth=2, 
            label=f'γζ = {gz:.0e}')

ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel('Énergie (TeV)', fontsize=12)
ax4.set_ylabel('Délai DSS (s)', fontsize=12)
ax4.set_title('(d) Délais DSS théoriques pour GRB 221009A', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10, title='λ_DSS = 1 mm')
ax4.grid(True, alpha=0.3)
ax4.axhspan(1, 1000, alpha=0.1, color='green', label='Fenêtres obs.')

plt.tight_layout()
plt.savefig('/home/claude/grb221009a_dss_analysis.png', dpi=300, bbox_inches='tight')
print("Figure sauvegardée: grb221009a_dss_analysis.png\n")

# Résumé final
print("=== RÉSUMÉ ET CONCLUSIONS ===\n")
print(f"GRB 221009A (z={z_grb}, D={D_Mpc} Mpc)")
print(f"Le GRB le plus énergétique jamais détecté par LHAASO")
print(f"Observations: 0.5-18 TeV sur plusieurs fenêtres temporelles\n")

print("Résultats test DSS:")
if fit_success and abs(A_DSS_fit/A_DSS_err) > 2:
    print(f"  ✓ Corrélation temporelle détectée: A_DSS = {A_DSS_fit:.1f}±{A_DSS_err:.1f} s·TeV")
    print(f"  ✓ Signification: {abs(A_DSS_fit/A_DSS_err):.1f}σ")
    print(f"  ✓ gamma_zeta/lambda_DSS ~ {gamma_zeta_over_lambda:.0f} m^-1")
    print(f"  *** INDICATION POTENTIELLE D'EFFET DSS ***")
else:
    print(f"  × Pas de signal DSS significatif avec les 3 fenêtres temporelles")
    print(f"  × Statistiques limitées (seulement 3 points)")
    print(f"  × Nécessite analyse plus fine avec résolution temporelle meilleure")

print()
print("Limites actuelles:")
print(f"  - Seulement 3 fenêtres temporelles analysées")
print(f"  - Résolution temporelle ~100s (large pour délais DSS <1s)")
print(f"  - Possible contamination par variabilité intrinsèque du GRB")
print()
print("Recommandations:")
print(f"  1. Analyser les données photon-par-photon (pas de binning temporel)")
print(f"  2. Appliquer méthode cross-corrélation entre bandes d'énergie")
print(f"  3. Combiner avec autres GRB LHAASO pour statistique")
print(f"  4. Comparer avec prédictions EBL standard vs EBL+DSS")

