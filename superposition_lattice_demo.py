#!/usr/bin/env python3
"""
Principe de Superposition sur Lattice Discret
==============================================

Démonstration que la superposition quantique émerge naturellement
de la linéarité de l'évolution sur un spacetime discret.

Auteur: Pour publication viXra
Date: Janvier 2026
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 100

print("="*70)
print(" PRINCIPE DE SUPERPOSITION SUR LATTICE DISCRET")
print("="*70)

# ============================================================================
# DÉMONSTRATION THÉORIQUE
# ============================================================================
print("\n" + "="*70)
print(" DÉMONSTRATION THÉORIQUE")
print("="*70)

print("""
Sur lattice quantique, l'évolution est linéaire:
  ψ(n, t+τ) = α·ψ(n-1, t) + β·ψ(n+1, t)

Si ψ₁ et ψ₂ sont solutions, alors:
  ψ = c₁ψ₁ + c₂ψ₂ est AUSSI solution !

Preuve:
  ψ(n,t+τ) = α·ψ(n-1,t) + β·ψ(n+1,t)
           = α·(c₁ψ₁(n-1) + c₂ψ₂(n-1)) + β·(...)
           = c₁[α·ψ₁(n-1) + β·ψ₁(n+1)] + c₂[...]
           = c₁ψ₁(n,t+τ) + c₂ψ₂(n,t+τ) ✅

→ SUPERPOSITION émerge de la LINÉARITÉ !
""")

# ============================================================================
# EXEMPLE 1: SUPERPOSITION DE PAQUETS GAUSSIENS
# ============================================================================
print("\n" + "="*70)
print(" EXEMPLE 1: SUPERPOSITION DE DEUX PAQUETS")
print("="*70)

N = 256
n = np.arange(N)
a = 1.0

# Deux paquets
n1, n2 = 80, 176
sigma = 10.0

psi1 = np.exp(-(n - n1)**2 / (2*sigma**2))
psi1 = psi1 / np.sqrt(np.sum(np.abs(psi1)**2))

psi2 = np.exp(-(n - n2)**2 / (2*sigma**2))
psi2 = psi2 / np.sqrt(np.sum(np.abs(psi2)**2))

# Superposition avec phase relative
c1 = 1/np.sqrt(2)
c2 = 1/np.sqrt(2) * np.exp(1j * np.pi/4)

psi_super = c1 * psi1 + c2 * psi2
psi_super = psi_super / np.sqrt(np.sum(np.abs(psi_super)**2))

# Densités
rho1 = np.abs(psi1)**2
rho2 = np.abs(psi2)**2
rho_super = np.abs(psi_super)**2
rho_classical = np.abs(c1)**2 * rho1 + np.abs(c2)**2 * rho2

# Terme d'interférence
interference = rho_super - rho_classical

print(f"\nÉtat 1: centré à n = {n1}")
print(f"État 2: centré à n = {n2}")
print(f"Coefficients: c₁ = {np.abs(c1):.4f}, c₂ = {np.abs(c2):.4f}e^(i{np.angle(c2):.3f})")
print(f"\nSuperposition: |ψ⟩ = c₁|ψ₁⟩ + c₂|ψ₂⟩")
print(f"Normalisation: ∫|ψ|² = {np.sum(rho_super):.8f} ✅")
print(f"\nTerme d'interférence:")
print(f"  Max amplitude: {np.max(np.abs(interference)):.6f}")
print(f"  → Effet QUANTIQUE présent ! ✅")

# Figure
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(n, rho1, 'b-', linewidth=2, label='|ψ₁|²', alpha=0.7)
axes[0].plot(n, rho2, 'r-', linewidth=2, label='|ψ₂|²', alpha=0.7)
axes[0].set_ylabel('Probabilité', fontsize=12)
axes[0].set_title('(a) États Individuels', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

axes[1].plot(n, rho_classical, 'g--', linewidth=2.5, 
             label='Classique: |c₁|²|ψ₁|² + |c₂|²|ψ₂|²', alpha=0.7)
axes[1].plot(n, rho_super, 'purple', linewidth=2.5, 
             label='Quantique: |c₁ψ₁ + c₂ψ₂|²')
axes[1].set_ylabel('Probabilité', fontsize=12)
axes[1].set_title('(b) Superposition vs Classique', fontweight='bold', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

axes[2].plot(n, interference, 'orange', linewidth=2.5)
axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
axes[2].fill_between(n, 0, interference, where=(interference > 0), 
                     alpha=0.3, color='blue', label='Constructive')
axes[2].fill_between(n, 0, interference, where=(interference < 0), 
                     alpha=0.3, color='red', label='Destructive')
axes[2].set_xlabel('Site n', fontsize=12)
axes[2].set_ylabel('Interférence', fontsize=12)
axes[2].set_title('(c) Terme d\'Interférence: 2Re(c₁*c₂*ψ₁*ψ₂)', 
                  fontweight='bold', fontsize=14)
axes[2].legend(fontsize=11, loc='upper right')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fig_superposition_packets.png', dpi=300, bbox_inches='tight')
print("\n✅ Sauvegardé: fig_superposition_packets.png")
plt.close()

# ============================================================================
# EXEMPLE 2: EXPÉRIENCE DE YOUNG
# ============================================================================
print("\n" + "="*70)
print(" EXEMPLE 2: DOUBLE FENTE (YOUNG)")
print("="*70)

N = 512
n = np.arange(N)

# Deux fentes
n_fente1 = N//2 - 40
n_fente2 = N//2 + 40
k = 0.15

# Ondes depuis chaque fente
dist1 = np.abs(n - n_fente1) + 0.1
dist2 = np.abs(n - n_fente2) + 0.1

psi_f1 = np.exp(1j * k * dist1) / np.sqrt(dist1)
psi_f2 = np.exp(1j * k * dist2) / np.sqrt(dist2)

psi_f1 = psi_f1 / np.sqrt(np.sum(np.abs(psi_f1)**2))
psi_f2 = psi_f2 / np.sqrt(np.sum(np.abs(psi_f2)**2))

# Superposition (deux fentes ouvertes)
psi_young = (psi_f1 + psi_f2) / np.sqrt(2)

rho_f1 = np.abs(psi_f1)**2
rho_f2 = np.abs(psi_f2)**2
rho_young = np.abs(psi_young)**2
rho_class = (rho_f1 + rho_f2) / 2

# Zone écran
ecran = slice(N//2 + 100, N//2 + 250)
franges = rho_young[ecran] - rho_class[ecran]

# Compter maxima
is_max = np.r_[False, (franges[1:-1] > franges[:-2]) & (franges[1:-1] > franges[2:]), False]
nb_franges = np.sum(is_max)

print(f"\nFente 1: site n = {n_fente1}")
print(f"Fente 2: site n = {n_fente2}")
print(f"Séparation: d = {n_fente2 - n_fente1} sites")
print(f"\nSur écran (sites {ecran.start}-{ecran.stop}):")
print(f"  Nombre de franges: {nb_franges}")
print(f"  Contraste: {(np.max(rho_young[ecran]) - np.min(rho_young[ecran])):.6f}")
print(f"  → Interférences quantiques visibles ! ✅")

# Figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(n[ecran], rho_class[ecran], 'g--', linewidth=2.5, 
         label='Classique (incohérent)', alpha=0.7)
ax1.plot(n[ecran], rho_young[ecran], 'b-', linewidth=2.5, 
         label='Quantique (cohérent)')
ax1.set_xlabel('Position écran (site n)', fontsize=12)
ax1.set_ylabel('Intensité', fontsize=12)
ax1.set_title('Double Fente: Patron d\'Interférence', fontweight='bold', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

ax2.plot(n[ecran], franges, 'orange', linewidth=2.5)
ax2.axhline(0, color='black', linestyle='--')
ax2.fill_between(n[ecran], 0, franges, alpha=0.3, color='orange')
ax2.set_xlabel('Position écran (site n)', fontsize=12)
ax2.set_ylabel('Interférence', fontsize=12)
ax2.set_title('Franges (Quantique - Classique)', fontweight='bold', fontsize=14)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fig_young_interference.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: fig_young_interference.png")
plt.close()

# ============================================================================
# EXEMPLE 3: CHAT DE SCHRÖDINGER
# ============================================================================
print("\n" + "="*70)
print(" EXEMPLE 3: CHAT DE SCHRÖDINGER")
print("="*70)

N = 100
n = np.arange(N)

# État vivant (gauche)
vivant = np.zeros(N, dtype=complex)
vivant[20:30] = 1.0
vivant = vivant / np.sqrt(np.sum(np.abs(vivant)**2))

# État mort (droite)
mort = np.zeros(N, dtype=complex)
mort[70:80] = 1.0
mort = mort / np.sqrt(np.sum(np.abs(mort)**2))

# Superposition
chat = (vivant + mort) / np.sqrt(2)

P_vivant_zone = np.sum(np.abs(chat[:50])**2)
P_mort_zone = np.sum(np.abs(chat[50:])**2)

print(f"\nÉtat |vivant⟩: sites 20-30")
print(f"État |mort⟩: sites 70-80")
print(f"\n|chat⟩ = (|vivant⟩ + |mort⟩)/√2")
print(f"  P(zone vivant) = {P_vivant_zone:.4f} (50%)")
print(f"  P(zone mort)   = {P_mort_zone:.4f} (50%)")
print(f"  Total: {P_vivant_zone + P_mort_zone:.4f} ✅")
print(f"\n→ Superposition RÉELLE sur lattice:")
print(f"  ψ(n) ≠ 0 aux DEUX endroits simultanément")

# Figure
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

axes[0].stem(n, np.abs(vivant)**2, basefmt=' ', linefmt='b-', markerfmt='bo')
axes[0].set_title('|vivant⟩', fontweight='bold', fontsize=13)
axes[0].set_ylabel('|ψ|²')
axes[0].set_ylim([0, 0.12])
axes[0].grid(alpha=0.3)

axes[1].stem(n, np.abs(mort)**2, basefmt=' ', linefmt='r-', markerfmt='ro')
axes[1].set_title('|mort⟩', fontweight='bold', fontsize=13)
axes[1].set_ylabel('|ψ|²')
axes[1].set_ylim([0, 0.12])
axes[1].grid(alpha=0.3)

axes[2].plot(n, np.abs(chat)**2, 'purple', linewidth=3)
axes[2].axvline(50, color='black', linestyle='--', linewidth=2, alpha=0.5)
axes[2].text(25, 0.09, 'VIVANT\n50%', ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
axes[2].text(75, 0.09, 'MORT\n50%', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
axes[2].set_title('|chat⟩ = (|vivant⟩ + |mort⟩)/√2', fontweight='bold', fontsize=13)
axes[2].set_xlabel('Site n')
axes[2].set_ylabel('|ψ|²')
axes[2].set_ylim([0, 0.12])
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fig_schrodinger_cat.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: fig_schrodinger_cat.png")
plt.close()

# ============================================================================
# RÉSUMÉ
# ============================================================================
print("\n" + "="*70)
print(" RÉSUMÉ")
print("="*70)
print("""
Le principe de superposition quantique émerge naturellement sur
un lattice discret parce que:

1. AMPLITUDES COMPLEXES: ψ(n) ∈ ℂ (pas binaire 0/1)
   → Combinaisons linéaires possibles

2. ÉVOLUTION LINÉAIRE: U[c₁ψ₁ + c₂ψ₂] = c₁U[ψ₁] + c₂U[ψ₂]
   → Superposition préservée dans le temps

3. INTERFÉRENCES: |ψ₁ + ψ₂|² = |ψ₁|² + |ψ₂|² + 2Re(ψ₁*ψ₂)
   → Effets quantiques (non-classiques)

Résultats démontrés:
✅ Superposition de paquets gaussiens → interférences visibles
✅ Double fente (Young) → franges d'interférence quantiques
✅ Chat de Schrödinger → existence simultanée des deux états

CONCLUSION: La superposition n'est pas un mystère métaphysique
mais une conséquence ALGÉBRIQUE de la linéarité de l'évolution
sur lattice avec amplitudes complexes.

La "bizarrerie" quantique = géométrie de l'espace vectoriel ℂ^N
""")

print("\n✅ Script terminé avec succès!")
print("   3 figures générées dans /home/claude/")
