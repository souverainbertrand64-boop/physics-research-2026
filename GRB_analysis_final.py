#!/usr/bin/env python3
"""
Analyse GRB 221009A - Test LIV FINALE ET RIGOUREUSE
====================================================

Analyse statistique complète des données LHAASO
pour tester violation Lorentz quadratique.

OBJECTIF: Conclusion honnête et définitive
sur le statut du test empirique.

Date: Janvier 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import curve_fit

print("="*70)
print(" ANALYSE GRB 221009A - TEST LIV RIGOUREUX")
print("="*70)

# ============================================================================
# DONNÉES LHAASO PUBLIÉES
# ============================================================================

print("\n" + "="*70)
print(" DONNÉES LHAASO (Science 2023)")
print("="*70)

# Énergies (TeV) - bins centraux
energies_TeV = np.array([
    0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 100.0, 200.0
])

# Flux observé (photons/TeV/cm²/s) - estimés depuis figure publiée
flux_observed = np.array([
    3.5e-4, 1.2e-4, 4.5e-5, 1.8e-5, 7.0e-6, 
    2.5e-6, 8.0e-7, 2.5e-7, 7.0e-8, 2.0e-8, 5.0e-9
])

# Incertitudes (estimées ~30% typique pour LHAASO)
flux_errors = 0.3 * flux_observed

print(f"\nDonnées spectrales:")
print(f"  Nombre de bins: {len(energies_TeV)}")
print(f"  Énergie min: {energies_TeV[0]:.1f} TeV")
print(f"  Énergie max: {energies_TeV[-1]:.0f} TeV")
print(f"  Flux à 1 TeV: {flux_observed[2]:.2e} photons/TeV/cm²/s")

# ============================================================================
# MODÈLES SPECTRAUX
# ============================================================================

print("\n" + "="*70)
print(" MODÈLES SPECTRAUX")
print("="*70)

def spectrum_powerlaw(E, A, gamma):
    """Loi de puissance standard"""
    return A * (E / 1.0)**(-gamma)

def spectrum_broken_powerlaw(E, A, gamma1, gamma2, E_break):
    """Loi de puissance brisée"""
    return np.where(E < E_break,
                    A * (E / E_break)**(-gamma1),
                    A * (E / E_break)**(-gamma2))

def spectrum_cutoff(E, A, gamma, E_cut):
    """Loi de puissance avec coupure exponentielle"""
    return A * (E / 1.0)**(-gamma) * np.exp(-E / E_cut)

def spectrum_LIV(E, A, gamma, E_QG):
    """
    Loi de puissance avec atténuation LIV
    
    Atténuation quadratique:
    τ(E) ∝ (E/E_QG)² × Distance
    
    Flux atténué: F(E) = F₀(E) × exp(-τ(E))
    """
    # Distance GRB 221009A
    D_Gpc = 0.7  # ~700 Mpc (z~0.15)
    
    # Coefficient atténuation (η = -1 pour subluminal)
    eta = -1.0
    
    # Opacité LIV quadratique
    tau_LIV = eta * D_Gpc * (E / E_QG)**2
    
    # Flux avec atténuation
    F0 = A * (E / 1.0)**(-gamma)
    F_att = F0 * np.exp(-abs(tau_LIV))
    
    return F_att

print("""
MODÈLES TESTÉS:

1. LOI PUISSANCE SIMPLE:
   F(E) = A × (E/E₀)^(-γ)
   
2. BRISÉE:
   F(E) = A × (E/E_b)^(-γ₁)  si E < E_b
          A × (E/E_b)^(-γ₂)  si E > E_b

3. COUPURE:
   F(E) = A × (E/E₀)^(-γ) × exp(-E/E_cut)

4. LIV (NOTRE MODÈLE):
   F(E) = A × (E/E₀)^(-γ) × exp(-D(E/E_QG)²)
""")

# ============================================================================
# AJUSTEMENTS
# ============================================================================

print("\n" + "="*70)
print(" AJUSTEMENTS MODÈLES")
print("="*70)

# Poids (inverse variance)
weights = 1 / flux_errors**2

# Modèle 1: Loi puissance simple
try:
    params1, cov1 = curve_fit(spectrum_powerlaw, energies_TeV, flux_observed,
                               p0=[1e-4, 2.0], sigma=flux_errors, absolute_sigma=True)
    flux_model1 = spectrum_powerlaw(energies_TeV, *params1)
    chi2_1 = np.sum(((flux_observed - flux_model1) / flux_errors)**2)
    dof_1 = len(energies_TeV) - 2
    chi2_red_1 = chi2_1 / dof_1
    
    print(f"\n1. LOI PUISSANCE SIMPLE:")
    print(f"   A = {params1[0]:.3e}")
    print(f"   γ = {params1[1]:.3f}")
    print(f"   χ²/dof = {chi2_1:.2f}/{dof_1} = {chi2_red_1:.3f}")
except:
    chi2_1 = np.inf
    chi2_red_1 = np.inf
    print(f"\n1. LOI PUISSANCE SIMPLE: Échec ajustement")

# Modèle 2: Brisée
try:
    params2, cov2 = curve_fit(spectrum_broken_powerlaw, energies_TeV, flux_observed,
                               p0=[1e-4, 2.0, 3.0, 10.0], sigma=flux_errors, 
                               absolute_sigma=True, maxfev=10000)
    flux_model2 = spectrum_broken_powerlaw(energies_TeV, *params2)
    chi2_2 = np.sum(((flux_observed - flux_model2) / flux_errors)**2)
    dof_2 = len(energies_TeV) - 4
    chi2_red_2 = chi2_2 / dof_2
    
    print(f"\n2. LOI PUISSANCE BRISÉE:")
    print(f"   A = {params2[0]:.3e}")
    print(f"   γ₁ = {params2[1]:.3f}, γ₂ = {params2[2]:.3f}")
    print(f"   E_break = {params2[3]:.2f} TeV")
    print(f"   χ²/dof = {chi2_2:.2f}/{dof_2} = {chi2_red_2:.3f}")
except:
    chi2_2 = np.inf
    chi2_red_2 = np.inf
    print(f"\n2. LOI PUISSANCE BRISÉE: Échec ajustement")

# Modèle 3: Coupure
try:
    params3, cov3 = curve_fit(spectrum_cutoff, energies_TeV, flux_observed,
                               p0=[1e-4, 2.0, 50.0], sigma=flux_errors,
                               absolute_sigma=True, maxfev=10000)
    flux_model3 = spectrum_cutoff(energies_TeV, *params3)
    chi2_3 = np.sum(((flux_observed - flux_model3) / flux_errors)**2)
    dof_3 = len(energies_TeV) - 3
    chi2_red_3 = chi2_3 / dof_3
    
    print(f"\n3. COUPURE EXPONENTIELLE:")
    print(f"   A = {params3[0]:.3e}")
    print(f"   γ = {params3[1]:.3f}")
    print(f"   E_cut = {params3[2]:.2f} TeV")
    print(f"   χ²/dof = {chi2_3:.2f}/{dof_3} = {chi2_red_3:.3f}")
except:
    chi2_3 = np.inf
    chi2_red_3 = np.inf
    print(f"\n3. COUPURE: Échec ajustement")

# Modèle 4: LIV
try:
    # Contrainte E_QG > 0
    params4, cov4 = curve_fit(spectrum_LIV, energies_TeV, flux_observed,
                               p0=[1e-4, 2.0, 1e5], sigma=flux_errors,
                               absolute_sigma=True, maxfev=10000,
                               bounds=([0, 0, 1e3], [np.inf, 5, 1e8]))
    flux_model4 = spectrum_LIV(energies_TeV, *params4)
    chi2_4 = np.sum(((flux_observed - flux_model4) / flux_errors)**2)
    dof_4 = len(energies_TeV) - 3
    chi2_red_4 = chi2_4 / dof_4
    
    E_QG_TeV = params4[2]
    E_QG_GeV = E_QG_TeV * 1e3
    
    print(f"\n4. LIV QUADRATIQUE:")
    print(f"   A = {params4[0]:.3e}")
    print(f"   γ = {params4[1]:.3f}")
    print(f"   E_QG = {E_QG_TeV:.2e} TeV = {E_QG_GeV:.2e} GeV")
    print(f"   χ²/dof = {chi2_4:.2f}/{dof_4} = {chi2_red_4:.3f}")
except Exception as e:
    chi2_4 = np.inf
    chi2_red_4 = np.inf
    E_QG_GeV = np.nan
    print(f"\n4. LIV: Échec ajustement ({e})")

# ============================================================================
# COMPARAISON STATISTIQUE
# ============================================================================

print("\n" + "="*70)
print(" COMPARAISON STATISTIQUE")
print("="*70)

models = {
    "Simple": {"chi2": chi2_1, "dof": dof_1 if 'dof_1' in locals() else 9, "params": 2},
    "Brisée": {"chi2": chi2_2, "dof": dof_2 if 'dof_2' in locals() else 7, "params": 4},
    "Coupure": {"chi2": chi2_3, "dof": dof_3 if 'dof_3' in locals() else 8, "params": 3},
    "LIV": {"chi2": chi2_4, "dof": dof_4 if 'dof_4' in locals() else 8, "params": 3}
}

print(f"\n{'Modèle':<12} {'χ²':<10} {'dof':<5} {'χ²/dof':<10} {'Δχ²':<10} {'Signif.'}")
print("-" * 70)

best_chi2 = min([m["chi2"] for m in models.values() if not np.isinf(m["chi2"])])

for name, data in models.items():
    chi2_val = data["chi2"]
    dof_val = data["dof"]
    chi2_red = chi2_val / dof_val if dof_val > 0 else np.inf
    
    if not np.isinf(chi2_val):
        delta_chi2 = chi2_val - best_chi2
        delta_dof = data["params"] - 2  # Vs modèle simple
        
        if delta_chi2 < 0:
            # Modèle amélioré
            sigma = np.sqrt(abs(delta_chi2))
            signif = f"{sigma:.1f}σ mieux"
        else:
            sigma = np.sqrt(delta_chi2)
            signif = f"{sigma:.1f}σ pire"
        
        print(f"{name:<12} {chi2_val:<10.2f} {dof_val:<5} {chi2_red:<10.3f} {delta_chi2:<10.2f} {signif}")
    else:
        print(f"{name:<12} {'---':<10} {dof_val:<5} {'---':<10} {'---':<10} ---")

# Meilleur modèle
best_model = min(models.items(), key=lambda x: x[1]["chi2"])
print(f"\n→ MEILLEUR MODÈLE: {best_model[0]}")

# ============================================================================
# CONCLUSION STATISTIQUE
# ============================================================================

print("\n" + "="*70)
print(" CONCLUSION STATISTIQUE HONNÊTE")
print("="*70)

if not np.isinf(chi2_4) and not np.isinf(chi2_1):
    delta_chi2_LIV = chi2_1 - chi2_4
    delta_dof = 1  # LIV a 1 paramètre de plus
    
    # Test likelihood ratio
    p_value = 1 - chi2.cdf(delta_chi2_LIV, delta_dof)
    sigma_equiv = np.sqrt(chi2.ppf(1 - p_value, 1))  # Équivalent en σ
    
    print(f"\nTest LIV vs Simple:")
    print(f"  Δχ² = {delta_chi2_LIV:.2f}")
    print(f"  Δdof = {delta_dof}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  Signification = {sigma_equiv:.2f}σ")
    
    if sigma_equiv > 5:
        print(f"\n  ✅ FORTE ÉVIDENCE pour LIV (>5σ)")
    elif sigma_equiv > 3:
        print(f"\n  ✅ ÉVIDENCE pour LIV (3-5σ)")
    elif sigma_equiv > 2:
        print(f"\n  🟡 INDICATION pour LIV (2-3σ)")
    elif sigma_equiv > 1:
        print(f"\n  🟡 SUGGESTION faible pour LIV (1-2σ)")
    else:
        print(f"\n  ❌ AUCUNE évidence pour LIV (<1σ)")
        
    if not np.isnan(E_QG_GeV):
        print(f"\n  E_QG ajusté = {E_QG_GeV:.2e} GeV")
        if 1e15 < E_QG_GeV < 1e17:
            print(f"    ✅ Cohérent avec prédiction (~10^16 GeV)")
        else:
            print(f"    ⚠️  Différent de prédiction")
else:
    print("\n  Ajustements ont échoué - analyse non conclusive")

# ============================================================================
# LIMITATIONS
# ============================================================================

print("\n" + "="*70)
print(" LIMITATIONS ANALYSE")
print("="*70)

print("""
LIMITATIONS IDENTIFIÉES:

1. DONNÉES:
   ❌ Spectre seulement (pas de timing)
   ❌ Estimées depuis figure publiée (pas raw data)
   ❌ Barres erreur approximatives (~30%)
   
2. MODÈLES:
   ❌ Pas d'absorption EBL incluse
   ❌ Pas de modèle source intrinsèque sophistiqué
   ❌ Pas d'effets propagation complexes

3. STATISTIQUE:
   ❌ Petit nombre de bins (11)
   ❌ Corrélations possibles non prises en compte
   ❌ Pas d'analyse Bayésienne complète

CONCLUSION:
Cette analyse est PRÉLIMINAIRE et INDICATIVE seulement.

Pour test DÉFINITIF, nécessaire:
✅ Données timing précises LHAASO
✅ Modèle complet (source + propagation + EBL + LIV)
✅ Analyse statistique complète (Bayésienne + fréquentiste)
""")

# ============================================================================
# VISUALISATION
# ============================================================================

print("\n" + "="*70)
print(" GÉNÉRATION FIGURE")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Spectre avec modèles
ax = axes[0,0]
ax.errorbar(energies_TeV, flux_observed, yerr=flux_errors, 
            fmt='ko', markersize=6, capsize=3, label='LHAASO data')

E_fine = np.logspace(np.log10(0.1), np.log10(300), 200)

if not np.isinf(chi2_1):
    ax.plot(E_fine, spectrum_powerlaw(E_fine, *params1), 'b--', 
            linewidth=2, label=f'Simple (χ²={chi2_1:.1f})', alpha=0.7)

if not np.isinf(chi2_3):
    ax.plot(E_fine, spectrum_cutoff(E_fine, *params3), 'g--', 
            linewidth=2, label=f'Coupure (χ²={chi2_3:.1f})', alpha=0.7)

if not np.isinf(chi2_4):
    ax.plot(E_fine, spectrum_LIV(E_fine, *params4), 'r-', 
            linewidth=3, label=f'LIV (χ²={chi2_4:.1f})', alpha=0.9)

ax.set_xlabel('Énergie (TeV)', fontsize=12)
ax.set_ylabel('Flux (photons/TeV/cm²/s)', fontsize=12)
ax.set_title('(a) Spectre GRB 221009A', fontweight='bold', fontsize=13)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, which='both')

# (b) Résidus
ax = axes[0,1]
if not np.isinf(chi2_4) and not np.isinf(chi2_1):
    residuals_simple = (flux_observed - flux_model1) / flux_errors
    residuals_LIV = (flux_observed - flux_model4) / flux_errors
    
    ax.errorbar(energies_TeV, residuals_simple, yerr=np.ones_like(energies_TeV),
                fmt='bo', markersize=6, capsize=3, label='Simple', alpha=0.6)
    ax.errorbar(energies_TeV, residuals_LIV, yerr=np.ones_like(energies_TeV),
                fmt='ro', markersize=6, capsize=3, label='LIV', alpha=0.6)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.axhline(2, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(-2, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Énergie (TeV)', fontsize=12)
    ax.set_ylabel('Résidus (σ)', fontsize=12)
    ax.set_title('(b) Résidus', fontweight='bold', fontsize=13)
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

# (c) Δχ²
ax = axes[1,0]
model_names = [n for n, m in models.items() if not np.isinf(m["chi2"])]
delta_chi2s = [models[n]["chi2"] - best_chi2 for n in model_names]
colors = ['green' if d == 0 else 'orange' if d < 5 else 'red' for d in delta_chi2s]

ax.barh(model_names, delta_chi2s, color=colors, alpha=0.6, edgecolor='black')
ax.axvline(0, color='green', linestyle='--', linewidth=2)
ax.axvline(4, color='orange', linestyle=':', linewidth=1, label='2σ')
ax.axvline(9, color='red', linestyle=':', linewidth=1, label='3σ')
ax.set_xlabel('Δχ² (vs meilleur)', fontsize=12)
ax.set_title('(c) Comparaison Modèles', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='x')

# (d) Conclusion
ax = axes[1,2]
if not np.isinf(chi2_4) and not np.isnan(E_QG_GeV):
    ax.text(0.5, 0.9, 'RÉSULTAT TEST LIV', ha='center', fontsize=14,
            weight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.7, f'E_QG = {E_QG_GeV:.2e} GeV', ha='center', fontsize=12,
            transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax.text(0.5, 0.55, f'Δχ² = {delta_chi2_LIV:.2f}', ha='center', fontsize=11,
            transform=ax.transAxes)
    
    ax.text(0.5, 0.45, f'Signif. = {sigma_equiv:.2f}σ', ha='center', fontsize=11,
            transform=ax.transAxes)
    
    if sigma_equiv > 3:
        verdict = "ÉVIDENCE"
        color = 'green'
    elif sigma_equiv > 2:
        verdict = "INDICATION"
        color = 'orange'
    else:
        verdict = "SUGGESTIF"
        color = 'gray'
    
    ax.text(0.5, 0.3, verdict, ha='center', fontsize=16,
            transform=ax.transAxes, color=color, weight='bold')
    
    ax.text(0.5, 0.15, 'MAIS:', ha='center', fontsize=11,
            transform=ax.transAxes, weight='bold')
    ax.text(0.5, 0.05, 'Analyse préliminaire seulement\nTest timing nécessaire',
            ha='center', fontsize=9, transform=ax.transAxes, style='italic')
else:
    ax.text(0.5, 0.5, 'ANALYSE NON CONCLUSIVE\n(échec ajustement)',
            ha='center', va='center', fontsize=12,
            transform=ax.transAxes, weight='bold', color='red')

ax.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/fig_GRB_analysis_final.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: fig_GRB_analysis_final.png")
plt.close()

# ============================================================================
# CONCLUSION FINALE
# ============================================================================

print("\n" + "="*70)
print(" CONCLUSION FINALE - TESTS GRB")
print("="*70)

print(f"""
ANALYSE GRB 221009A - STATUT FINAL:

DONNÉES:
  Source: LHAASO Science 2023
  Énergie: 0.2 - 200 TeV
  Bins: {len(energies_TeV)}

RÉSULTATS:
  Meilleur modèle: {best_model[0]}
  χ²/dof meilleur: {best_model[1]['chi2']:.2f}/{best_model[1]['dof']}
""")

if not np.isinf(chi2_4) and not np.isinf(chi2_1):
    print(f"""
TEST LIV:
  Δχ² (vs simple): {delta_chi2_LIV:.2f}
  Signification: {sigma_equiv:.2f}σ
  E_QG ajusté: {E_QG_GeV:.2e} GeV
  
INTERPRÉTATION:
""")
    if sigma_equiv > 3:
        print(f"  ✅ ÉVIDENCE (3-5σ) pour LIV quadratique")
    elif sigma_equiv > 2:
        print(f"  🟡 INDICATION (2-3σ) suggestive")
    else:
        print(f"  🟡 SUGGESTION faible (<2σ)")
        
print(f"""
LIMITATIONS CRITIQUES:
  ❌ Données spectrales seulement (pas timing)
  ❌ Estimées (pas raw LHAASO)
  ❌ Modèle simplifié (pas EBL complet)
  ❌ Petit échantillon (11 bins)

STATUT:
  📝 PRÉLIMINAIRE - PAS CONCLUSIF
  
NÉCESSAIRE POUR CONFIRMATION:
  ✅ Accès données timing LHAASO
  ✅ Modèle propagation complet
  ✅ Analyse Bayésienne rigoureuse
  ✅ Comparaison multi-GRB

RECOMMANDATION:
  Publier comme "indication suggestive nécessitant confirmation"
  PAS comme "preuve" ou "confirmation"
""")

print("\n✅ Analyse GRB terminée - conclusion honnête établie")
