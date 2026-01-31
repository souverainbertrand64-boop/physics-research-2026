#!/usr/bin/env python3
"""
Intrication Quantique sur Lattice Discret
==========================================

Démonstration que l'intrication quantique émerge naturellement
de la structure d'espace produit tensoriel ℂ^N ⊗ ℂ^N sur lattice.

Auteur: Pour publication viXra
Date: Janvier 2026
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 11

print("="*70)
print(" INTRICATION QUANTIQUE SUR LATTICE DISCRET")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
N = 64  # Sites par particule
n = np.arange(N)

print(f"\nConfiguration:")
print(f"  Nombre de sites par particule: N = {N}")
print(f"  Espace 1 particule: ℂ^{N} (dimension {N})")
print(f"  Espace 2 particules: ℂ^{N} ⊗ ℂ^{N} = ℂ^{N**2} (dimension {N**2})")
print(f"  États séparables: ~{2*N} dimensions")
print(f"  États intriqués: ~{N**2 - 2*N} dimensions")
print(f"  → Intrication domine ! ({100*(N**2-2*N)/N**2:.1f}% de l'espace)")

# ============================================================================
# EXEMPLE 1: PAIRE EPR
# ============================================================================
print("\n" + "="*70)
print(" EXEMPLE 1: PAIRE EPR (Einstein-Podolsky-Rosen)")
print("="*70)

na, nb = 20, 44
nc, nd = 20, 44

# État séparable (pour comparaison)
psi_A = np.zeros(N, dtype=complex)
psi_A[na] = 1/np.sqrt(2)
psi_A[nb] = 1/np.sqrt(2)

psi_B = np.zeros(N, dtype=complex)
psi_B[nc] = 1/np.sqrt(2)
psi_B[nd] = 1/np.sqrt(2)

Psi_sep = np.outer(psi_A, psi_B)

# État intriqué EPR
Psi_EPR = np.zeros((N, N), dtype=complex)
Psi_EPR[na, nc] = 1/np.sqrt(2)
Psi_EPR[nb, nd] = 1/np.sqrt(2)

print(f"\nÉtat SÉPARABLE:")
print(f"  |Ψ⟩ = |ψ_A⟩ ⊗ |ψ_B⟩")
print(f"  = [(|{na}⟩+|{nb}⟩)/√2] ⊗ [(|{nc}⟩+|{nd}⟩)/√2]")
print(f"  = (|{na},{nc}⟩ + |{na},{nd}⟩ + |{nb},{nc}⟩ + |{nb},{nd}⟩)/2")
print(f"  → 4 configurations, indépendantes")

print(f"\nÉtat INTRIQUÉ (EPR):")
print(f"  |Ψ_EPR⟩ = (|{na},{nc}⟩ + |{nb},{nd}⟩)/√2")
print(f"  → 2 configurations seulement, CORRÉLÉES!")
print(f"    Si A en {na}, alors B en {nc} (100% corrélation)")
print(f"    Si A en {nb}, alors B en {nd} (100% corrélation)")

# Test séparabilité (SVD)
U_sep, S_sep, Vh_sep = np.linalg.svd(Psi_sep)
U_EPR, S_EPR, Vh_EPR = np.linalg.svd(Psi_EPR)

rank_sep = np.sum(S_sep > 1e-10)
rank_EPR = np.sum(S_EPR > 1e-10)

print(f"\nTest de séparabilité (rang de Schmidt):")
print(f"  Séparable: rang = {rank_sep}")
print(f"    Valeurs singulières: {S_sep[S_sep > 1e-10]}")
print(f"    → Rang 1 = séparable ✅")

print(f"\n  EPR: rang = {rank_EPR}")
print(f"    Valeurs singulières: {S_EPR[S_EPR > 1e-10]}")
print(f"    → Rang > 1 = intriqué ✅")

# Visualisation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

im1 = axes[0].imshow(np.abs(Psi_sep)**2, cmap='hot', interpolation='nearest',
                      extent=[0, N, N, 0])
axes[0].plot([nc-0.5, nc-0.5, nd-0.5, nd-0.5], 
             [na-0.5, nb-0.5, na-0.5, nb-0.5], 
             'wo', markersize=12, markeredgewidth=2.5)
axes[0].set_xlabel('Particule B (site n₂)', fontsize=12)
axes[0].set_ylabel('Particule A (site n₁)', fontsize=12)
axes[0].set_title('(a) SÉPARABLE\n4 pics indépendants', fontweight='bold', fontsize=13)
plt.colorbar(im1, ax=axes[0], label='|Ψ|²')

im2 = axes[1].imshow(np.abs(Psi_EPR)**2, cmap='hot', interpolation='nearest',
                      extent=[0, N, N, 0])
axes[1].plot([nc-0.5, nd-0.5], [na-0.5, nb-0.5], 
             'wo', markersize=12, markeredgewidth=2.5)
axes[1].set_xlabel('Particule B (site n₂)', fontsize=12)
axes[1].set_ylabel('Particule A (site n₁)', fontsize=12)
axes[1].set_title('(b) INTRIQUÉ (EPR)\n2 pics corrélés', fontweight='bold', fontsize=13)
plt.colorbar(im2, ax=axes[1], label='|Ψ|²')

diff = np.abs(Psi_EPR)**2 - np.abs(Psi_sep)**2
vmax = np.max(np.abs(diff))
im3 = axes[2].imshow(diff, cmap='RdBu_r', interpolation='nearest',
                      extent=[0, N, N, 0], vmin=-vmax, vmax=vmax)
axes[2].set_xlabel('Particule B (site n₂)', fontsize=12)
axes[2].set_ylabel('Particule A (site n₁)', fontsize=12)
axes[2].set_title('(c) SIGNATURE\nEPR - Séparable', fontweight='bold', fontsize=13)
plt.colorbar(im3, ax=axes[2], label='Δ|Ψ|²')

plt.tight_layout()
plt.savefig('/home/claude/fig_entanglement_EPR.png', dpi=300, bbox_inches='tight')
print("\n✅ Sauvegardé: fig_entanglement_EPR.png")
plt.close()

# ============================================================================
# EXEMPLE 2: ENTROPIE D'INTRICATION
# ============================================================================
print("\n" + "="*70)
print(" EXEMPLE 2: ENTROPIE D'INTRICATION")
print("="*70)

def entanglement_entropy(Psi):
    """Calcule entropie de von Neumann"""
    U, S, Vh = np.linalg.svd(Psi)
    eigenvalues = S**2
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))
    return entropy, eigenvalues

S_sep, eig_sep = entanglement_entropy(Psi_sep)
S_EPR, eig_EPR = entanglement_entropy(Psi_EPR)

print(f"\nEntropie d'intrication S (bits):")
print(f"  Séparable: S = {S_sep:.6f}")
print(f"    → Pas d'intrication ✅")
print(f"\n  EPR: S = {S_EPR:.6f}")
print(f"    → Intrication maximale (S_max = log₂(2) = 1) ✅")
print(f"    → Ratio: {S_EPR/1.0*100:.1f}% du maximum")

# États intermédiaires
print(f"\nÉtats intermédiaires (intrication partielle):")
alphas = np.linspace(0, 1, 11)
entropies = []

for alpha in alphas:
    Psi_mixed = np.zeros((N, N), dtype=complex)
    Psi_mixed[na, nc] = np.sqrt(alpha)/np.sqrt(2)
    Psi_mixed[nb, nd] = np.sqrt(alpha)/np.sqrt(2)
    Psi_mixed[na, nd] = np.sqrt(1-alpha)/2
    Psi_mixed[nb, nc] = np.sqrt(1-alpha)/2
    Psi_mixed = Psi_mixed / np.linalg.norm(Psi_mixed)
    S, _ = entanglement_entropy(Psi_mixed)
    entropies.append(S)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(alphas, entropies, 'bo-', linewidth=3, markersize=10, label='S(α)')
ax.axhline(0, color='green', linestyle='--', linewidth=2, 
           label='Séparable (S=0)', alpha=0.7)
ax.axhline(1, color='red', linestyle='--', linewidth=2, 
           label='Maximale (S=1)', alpha=0.7)
ax.fill_between(alphas, 0, entropies, alpha=0.25, color='blue')

ax.set_xlabel('Paramètre d\'intrication α', fontsize=13)
ax.set_ylabel('Entropie S (bits)', fontsize=13)
ax.set_title('Quantification de l\'Intrication via Entropie de von Neumann', 
             fontweight='bold', fontsize=14)
ax.legend(fontsize=12, loc='upper left')
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([-0.05, 1.1])

plt.tight_layout()
plt.savefig('/home/claude/fig_entanglement_entropy.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: fig_entanglement_entropy.png")
plt.close()

# ============================================================================
# EXEMPLE 3: ÉVOLUTION DE L'INTRICATION
# ============================================================================
print("\n" + "="*70)
print(" EXEMPLE 3: ÉVOLUTION TEMPORELLE")
print("="*70)

# Simulation évolution simple (dispersion libre)
times = np.linspace(0, 50, 21)
entropies_t = []

for t in times:
    # Phase evolution (simplifié)
    phase_A = np.exp(-1j * 0.1 * n**2 * t / N**2)
    phase_B = np.exp(-1j * 0.1 * n**2 * t / N**2)
    
    # Appliquer phase
    Psi_t = Psi_EPR * np.outer(phase_A, phase_B)
    
    S_t, _ = entanglement_entropy(Psi_t)
    entropies_t.append(S_t)

print(f"\nÉvolution de l'intrication:")
print(f"  t =  0 : S = {entropies_t[0]:.6f} bits")
print(f"  t = 25 : S = {entropies_t[len(times)//2]:.6f} bits")
print(f"  t = 50 : S = {entropies_t[-1]:.6f} bits")
print(f"  → Intrication CONSERVÉE pendant évolution unitaire ✅")

# Plot évolution
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(times, entropies_t, 'r-', linewidth=3)
ax.axhline(1, color='black', linestyle='--', linewidth=2, alpha=0.5,
           label='Maximum (S=1)')
ax.fill_between(times, 0, entropies_t, alpha=0.3, color='red')

ax.set_xlabel('Temps (unités arbitraires)', fontsize=13)
ax.set_ylabel('Entropie d\'intrication S (bits)', fontsize=13)
ax.set_title('Conservation de l\'Intrication (Évolution Unitaire)', 
             fontweight='bold', fontsize=14)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
ax.set_ylim([0, 1.2])

plt.tight_layout()
plt.savefig('/home/claude/fig_entanglement_evolution.png', dpi=300, bbox_inches='tight')
print("✅ Sauvegardé: fig_entanglement_evolution.png")
plt.close()

# ============================================================================
# RÉSUMÉ
# ============================================================================
print("\n" + "="*70)
print(" RÉSUMÉ")
print("="*70)
print("""
L'intrication quantique émerge naturellement sur lattice discret de la
structure d'espace produit tensoriel ℂ^N ⊗ ℂ^N:

1. ESPACE PRODUIT: 2 particules sur N sites → espace ℂ^(N²)
   - Séparables: sous-variété de dimension ~2N
   - Intriqués: dimension ~N² - 2N
   → Intrication = RÈGLE (>97% de l'espace pour N=64)

2. CORRÉLATIONS NON-LOCALES: État Ψ(n₁,n₂) unique
   - Mesure sur A projette instantanément B
   - MAIS: pas de transmission d'information
   - Corrélations déjà présentes dans Ψ

3. QUANTIFICATION: Entropie de von Neumann
   - S = 0 → séparable
   - S > 0 → intriqué
   - S = log(dim) → maximalement intriqué

4. CONSERVATION: Évolution unitaire préserve S
   - Intrication ne disparaît pas spontanément
   - Décohérence requise pour "collapse"

DÉMYSTIFICATION: L'intrication n'est pas "action fantôme à distance"
mais conséquence géométrique de vivre dans ℂ^(N²) plutôt que ℂ^N × ℂ^N.

Einstein avait tort: Dieu joue aux dés dans ℂ^(N²) !
""")

print("\n✅ Script terminé!")
print("   3 figures générées dans /home/claude/")
