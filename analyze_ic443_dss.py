"""
Analyse DSS appliquée aux données IC 443 de LHAASO
Extrait les données spectrales du papier et teste les déviations DSS
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Données extraites de la Table S2 du papier LHAASO
# Source C0 (compacte)
E_C0 = np.array([0.62, 1.14, 2.14, 3.65, 7.14, 15.15, 13.40, 33.04])  # TeV
flux_C0 = np.array([1.42e-12, 6.40e-13, 5.51e-13, 2.26e-13, 1.57e-13, 
                    1.15e-13, 1.12e-13, 2.29e-14])  # TeV cm^-2 s^-1
flux_C0_err = np.array([0.45e-12, 1.57e-13, 0.96e-13, 0.53e-13, 0.37e-13,
                        0.30e-13, 0.91e-13, 1.64e-14])

# Source C1 (étendue)
E_C1 = np.array([0.79, 1.36, 2.36, 3.84, 7.11, 13.63, 12.78, 28.33])  # TeV
flux_C1 = np.array([2.51e-12, 2.52e-12, 1.74e-12, 1.02e-12, 5.74e-13,
                    2.39e-13, 2.95e-13, 1.80e-13])  # TeV cm^-2 s^-1
flux_C1_err = np.array([0.72e-12, 0.32e-12, 0.24e-12, 0.16e-12, 1.31e-13,
                        1.59e-13, 1.77e-13, 0.55e-13])

# Données Fermi-LAT (extraites de Table S3)
# Pour C0 (4FGL J0617.2+2234e)
E_fermi_C0 = np.array([0.069, 0.134, 0.253, 0.477, 0.900, 1.698, 3.201, 
                       6.034, 11.376, 21.441])  # GeV convertis de log(E/MeV)
flux_fermi_C0 = np.array([1.42e-9, 1.16e-9, 7.28e-10, 4.03e-10, 2.15e-10,
                          1.04e-10, 5.17e-11, 2.28e-11, 9.92e-12, 4.33e-12])  # MeV^-1 cm^-2 s^-1
# Conversion en E^2 dN/dE
E_fermi_C0_TeV = E_fermi_C0 / 1000  # TeV
flux_fermi_C0_converted = flux_fermi_C0 * (E_fermi_C0 * 1e3)**2 * 1e6  # TeV cm^-2 s^-1

print("=== Analyse DSS des données IC 443 ===\n")

# Test 1: Fit standard (loi de puissance)
def powerlaw(E, phi0, alpha):
    return phi0 * (E / 3.0)**(-alpha)

# Fit C0
popt_C0, pcov_C0 = curve_fit(powerlaw, E_C0, flux_C0/E_C0**2, 
                              sigma=flux_C0_err/E_C0**2, absolute_sigma=True)
phi0_C0, alpha_C0 = popt_C0
phi0_C0_err, alpha_C0_err = np.sqrt(np.diag(pcov_C0))

print("Source C0 (LHAASO) - Fit loi de puissance standard:")
print(f"  φ₀ = {phi0_C0:.2e} ± {phi0_C0_err:.2e} TeV⁻¹ cm⁻² s⁻¹")
print(f"  α = {alpha_C0:.3f} ± {alpha_C0_err:.3f}")
print(f"  (Papier: α = 2.95 ± 0.07)")

# Calcul du chi2
flux_model_C0 = powerlaw(E_C0, *popt_C0) * E_C0**2
chi2_C0 = np.sum(((flux_C0 - flux_model_C0) / flux_C0_err)**2)
ndof_C0 = len(E_C0) - 2
print(f"  χ²/ndof = {chi2_C0:.2f}/{ndof_C0} = {chi2_C0/ndof_C0:.2f}\n")

# Test 2: Fit avec correction DSS
# Modèle: flux(E) = flux_std(E) × [1 + A_DSS/E]
def powerlaw_dss(E, phi0, alpha, A_DSS):
    flux_std = phi0 * (E / 3.0)**(-alpha)
    correction_DSS = 1 + A_DSS / E  # correction DSS: Δt ∝ 1/E
    return flux_std * correction_DSS

try:
    popt_C0_dss, pcov_C0_dss = curve_fit(powerlaw_dss, E_C0, flux_C0/E_C0**2,
                                         sigma=flux_C0_err/E_C0**2, 
                                         p0=[phi0_C0, alpha_C0, 0.01],
                                         absolute_sigma=True)
    phi0_dss, alpha_dss, A_DSS = popt_C0_dss
    phi0_dss_err, alpha_dss_err, A_DSS_err = np.sqrt(np.diag(pcov_C0_dss))
    
    flux_model_C0_dss = powerlaw_dss(E_C0, *popt_C0_dss) * E_C0**2
    chi2_C0_dss = np.sum(((flux_C0 - flux_model_C0_dss) / flux_C0_err)**2)
    
    print("Source C0 - Fit avec correction DSS:")
    print(f"  φ₀ = {phi0_dss:.2e} ± {phi0_dss_err:.2e} TeV⁻¹ cm⁻² s⁻¹")
    print(f"  α = {alpha_dss:.3f} ± {alpha_dss_err:.3f}")
    print(f"  A_DSS = {A_DSS:.4f} ± {A_DSS_err:.4f} TeV")
    print(f"  χ²/ndof = {chi2_C0_dss:.2f}/{ndof_C0-1} = {chi2_C0_dss/(ndof_C0-1):.2f}")
    
    # Test statistique
    Delta_chi2 = chi2_C0 - chi2_C0_dss
    significance = np.sqrt(Delta_chi2)
    print(f"  Δχ² = {Delta_chi2:.2f}")
    print(f"  Signification correction DSS = {significance:.2f}σ")
    
    if A_DSS_err > 0:
        A_DSS_sigma = abs(A_DSS) / A_DSS_err
        print(f"  A_DSS/σ(A_DSS) = {A_DSS_sigma:.2f}σ")
        if A_DSS_sigma > 2:
            print(f"  *** DEVIATION DSS POTENTIELLE (>{A_DSS_sigma:.1f}σ) ***")
    
    dss_fit_success = True
except:
    print("  Fit DSS échoué (pas assez de contrainte)")
    dss_fit_success = False

print()

# Test 3: Analyse des résidus
residuals_C0 = (flux_C0 - flux_model_C0) / flux_C0_err
print("Analyse des résidus (standard):")
print(f"  Résidu moyen = {np.mean(residuals_C0):.3f} σ")
print(f"  Écart-type résidus = {np.std(residuals_C0):.3f}")

# Chercher corrélation résidus vs 1/E (signature DSS)
corr_coeff = np.corrcoef(1/E_C0, residuals_C0)[0,1]
print(f"  Corrélation résidus vs 1/E = {corr_coeff:.3f}")
if abs(corr_coeff) > 0.5:
    print(f"  *** Corrélation significative détectée ! ***")

print()

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: SED complet (Fermi + LHAASO)
ax1 = axes[0, 0]
E_plot = np.logspace(np.log10(0.01), np.log10(100), 200)

# Données
ax1.errorbar(E_fermi_C0_TeV, flux_fermi_C0_converted, fmt='o', 
            color='purple', label='Fermi-LAT C0', markersize=6, alpha=0.7)
ax1.errorbar(E_C0, flux_C0, yerr=flux_C0_err, fmt='s', 
            color='red', label='LHAASO C0', markersize=8, capsize=3)

# Modèles
ax1.plot(E_plot, powerlaw(E_plot, *popt_C0) * E_plot**2, 'r--', 
        linewidth=2, label=f'Loi puissance (α={alpha_C0:.2f})')
if dss_fit_success:
    ax1.plot(E_plot, powerlaw_dss(E_plot, *popt_C0_dss) * E_plot**2, 'b-', 
            linewidth=2, label=f'Avec DSS (A={A_DSS:.3f} TeV)')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Énergie (TeV)', fontsize=12)
ax1.set_ylabel('E² dN/dE (TeV cm⁻² s⁻¹)', fontsize=12)
ax1.set_title('(a) Spectre IC 443 C0: Fermi + LHAASO', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Panel 2: Résidus vs Énergie
ax2 = axes[0, 1]
ax2.errorbar(E_C0, residuals_C0, yerr=1, fmt='o', color='red', 
            markersize=8, capsize=3)
ax2.axhline(0, color='black', linestyle='--', linewidth=2)
ax2.axhline(2, color='orange', linestyle=':', linewidth=1, alpha=0.5)
ax2.axhline(-2, color='orange', linestyle=':', linewidth=1, alpha=0.5)
ax2.set_xscale('log')
ax2.set_xlabel('Énergie (TeV)', fontsize=12)
ax2.set_ylabel('Résidus normalisés (σ)', fontsize=12)
ax2.set_title('(b) Résidus du fit standard', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Panel 3: Résidus vs 1/E (test DSS)
ax3 = axes[1, 0]
ax3.errorbar(1/E_C0, residuals_C0, yerr=1, fmt='o', color='blue',
            markersize=8, capsize=3, label='Données')
# Fit linéaire des résidus
if len(E_C0) > 2:
    p_resid = np.polyfit(1/E_C0, residuals_C0, 1)
    x_resid = np.linspace((1/E_C0).min(), (1/E_C0).max(), 100)
    ax3.plot(x_resid, np.polyval(p_resid, x_resid), 'r--', linewidth=2,
            label=f'Tendance (pente={p_resid[0]:.2f})')
ax3.axhline(0, color='black', linestyle='--', linewidth=2)
ax3.set_xlabel('1/E (TeV⁻¹)', fontsize=12)
ax3.set_ylabel('Résidus normalisés (σ)', fontsize=12)
ax3.set_title('(c) Test signature DSS (résidus vs 1/E)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
textstr = f'Corrélation = {corr_coeff:.3f}'
ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Panel 4: Comparaison modèles
ax4 = axes[1, 1]
# Déviation fractionnelle
deviation_std = (flux_C0 - flux_model_C0) / flux_C0 * 100
deviation_std_err = flux_C0_err / flux_C0 * 100

ax4.errorbar(E_C0, deviation_std, yerr=deviation_std_err, fmt='o',
            color='red', markersize=8, capsize=3, label='Standard')
if dss_fit_success:
    deviation_dss = (flux_C0 - flux_model_C0_dss) / flux_C0 * 100
    ax4.errorbar(E_C0, deviation_dss, yerr=deviation_std_err, fmt='s',
                color='blue', markersize=8, capsize=3, label='Avec DSS', alpha=0.7)
ax4.axhline(0, color='black', linestyle='--', linewidth=2)
ax4.axhline(5, color='orange', linestyle=':', alpha=0.5)
ax4.axhline(-5, color='orange', linestyle=':', alpha=0.5)
ax4.set_xscale('log')
ax4.set_xlabel('Énergie (TeV)', fontsize=12)
ax4.set_ylabel('Déviation fractionnelle (%)', fontsize=12)
ax4.set_title('(d) Comparaison modèles', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/ic443_dss_analysis.png', dpi=300, bbox_inches='tight')
print("Figure sauvegardée: ic443_dss_analysis.png")

# Résumé et conclusion
print("\n=== RÉSUMÉ DE L'ANALYSE ===")
print(f"Source: IC 443 C0 (LHAASO + Fermi-LAT)")
print(f"Distance: ~1.5 kpc")
print(f"Gamme énergie: {E_fermi_C0_TeV.min()*1000:.1f} GeV - {E_C0.max():.1f} TeV")
print(f"\nRésultat fit standard: χ²/ndof = {chi2_C0/ndof_C0:.2f}")
if dss_fit_success:
    print(f"Résultat fit DSS: χ²/ndof = {chi2_C0_dss/(ndof_C0-1):.2f}")
    print(f"Amélioration: Δχ² = {Delta_chi2:.2f} ({significance:.2f}σ)")
    if significance > 2:
        print("\n*** INDICATION POSSIBLE D'EFFET DSS ***")
        print(f"Paramètre DSS: A = {A_DSS:.4f} ± {A_DSS_err:.4f} TeV")
    else:
        print("\nPas de déviation DSS significative détectée")
        print(f"Limite 95% sur A_DSS: < {A_DSS + 1.96*A_DSS_err:.4f} TeV")
else:
    print("\nFit DSS non convergent - données compatibles avec modèle standard")
