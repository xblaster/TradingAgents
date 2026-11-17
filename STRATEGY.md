# Stratégie Revolut X : protocole pré-enregistré

Ce document est **commité avant** toute évaluation hors échantillon. La spécification, les paramètres et les critères de viabilité ci-dessous ne doivent pas être modifiés après coup pour faire passer un test. Si une modification s'impose, elle est datée, justifiée dans ce fichier, et l'évaluation est relancée en entier.

## 1. Hypothèse

Les cryptomonnaies présentent un **effet de tendance (time-series momentum)**, documenté avant 2023 dans la littérature académique : Moskowitz, Ooi & Pedersen (2012) sur les futures, Liu & Tsyvinski (2021) puis Liu, Tsyvinski & Wu (2022) sur les cryptos. Une stratégie long-only qui est investie quand la tendance est haussière et en cash (EUR) sinon devrait :

- capter une partie des hausses ;
- éviter une grande partie des krachs (-75 % en 2018, -65 % en 2022) ;
- et donc offrir un meilleur rendement ajusté du risque, avec un drawdown nettement plus faible, que l'achat-conservation.

Le but n'est **pas** de battre le buy & hold en rendement brut sur un marché haussier, mais d'obtenir une exposition crypto **survivable**.

## 2. Spécification (paramètres fixés a priori, aucune optimisation)

**Univers** (paires EUR cotées sur Revolut X, vérifiées le 2026-11-17) :
- *core* : BTC-EUR, ETH-EUR ;
- *étendu* : core + SOL, XRP, ADA, DOGE, LINK, DOT, AVAX, LTC (en EUR).

Un actif n'est éligible qu'avec au moins 250 jours d'historique.

**Signal de tendance** de l'actif *i* au jour *t*, calculé à la clôture avec les seules données jusqu'à *t* :
`s_i = moyenne sur L ∈ {20, 60, 120, 250} de 1[close_t > close_{t-L}]`.
C'est un ensemble d'horizons standards, sans choix de « meilleur » horizon. Le signal vaut 0, 0,25, 0,5, 0,75 ou 1.

**Pondération** en parité de risque inverse, puis modulée par la tendance :
- `v_i` = volatilité réalisée sur 60 jours ;
- `b_i = (1/v_i) / Σ_j (1/v_j)` sur les actifs éligibles ;
- `w_i = b_i × s_i`.

L'exposition totale est `Σ w_i ≤ 100 %`. Le reste est en EUR. Pas de levier, pas de vente à découvert.

**Exécution** :
- rebalancement tous les 7 jours ;
- ordres exécutés à l'open du lendemain ;
- un ordre n'est passé que si l'écart de poids dépasse 2 % du portefeuille (sortie complète toujours exécutée) ;
- coût de 15 pb par unité de turnover : frais taker de 0,09 % + spread et glissement.

**Benchmarks** :
- *EW B&H* : équipondéré sur le même univers, rebalancé tous les 30 jours, mêmes coûts ;
- *BTC B&H* : 100 % BTC-EUR.

## 3. Périodes

| | Période | Rôle |
|---|---|---|
| In-sample | 2018-01-01 → 2022-12-31 | Contrôle de cohérence uniquement : aucun paramètre n'y est ajusté. |
| **Hors échantillon** | **2023-01-01 → 2026-11-17** | **Seule période qui décide de la viabilité.** |

Limite assumée : l'hypothèse est publique depuis 2021, et les grandes lignes du marché 2023-2026 sont connues de l'auteur. Aucun paramètre n'est choisi en fonction de ces années, mais le hors-échantillon n'est pas « aveugle » au sens strict. La validation définitive est le **paper trading prospectif** (section 5).

## 4. Critères de viabilité (tous requis, hors échantillon)

| # | Critère |
|---|---|
| C1 | CAGR > 0 après coûts, sur l'univers core **et** sur l'univers étendu |
| C2 | Sharpe ≥ Sharpe de l'EW B&H du même univers (core et étendu) |
| C3 | Max drawdown ≤ 0,6 × max drawdown de l'EW B&H (au moins 40 % moins profond), core et étendu |
| C4 | Robustesse des paramètres : ≥ 75 % des variantes satisfont C2 **et** C3 sur l'univers core. Les variantes sont 4 échelles d'horizons (×0,5, ×0,75, ×1,25, ×1,5) × 3 fréquences de rebalancement (3, 7, 14 jours), soit 12 variantes. |
| C5 | Coûts doublés (30 pb) : C1 tient toujours, core et étendu |
| C6 | Cohérence in-sample : sur 2018-2022, Sharpe ≥ EW B&H et max drawdown < EW B&H (univers core) |

Verdict : **VIABLE** si C1 à C6 passent. Sinon **NON VIABLE**, et le rapport dit lesquels échouent. Il n'y a pas de « presque ».

## 5. Après le verdict

Même si la stratégie est VIABLE en backtest, le passage en réel suit cet ordre :

1. **Paper trading prospectif pendant au moins 8 semaines** sur les prix live de Revolut X, avec un journal horodaté des signaux.
2. **Capital réel plafonné**, ordres limit post-only (0 % de frais maker) passés manuellement à partir de `desk portfolio`.
3. Automatisation par API seulement ensuite, avec plafonds durs.

Le desk LLM n'entre pas dans cette stratégie. Il pourra être évalué plus tard comme filtre (veto) en paper trading, puisqu'aucun backtest du LLM n'est fiable avant sa date de coupure.
