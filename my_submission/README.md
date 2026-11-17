# Era Nova — Soumission « Possession de balle » (More Than A Hack 2027, Phase 1)

Ce dossier contient **tout ce qui est nécessaire pour reproduire la soumission** à partir du
bundle de données fourni par les organisateurs et de notre code.

| Fichier | Rôle |
|---|---|
| `Era_Nova.csv` | **la soumission** — CSV `frame,label`, 199 319 lignes |
| `possession_inference.ipynb` | **le document technique** — méthode documentée, exécutable de bout en bout |
| `common.py` | le module de la méthode (features, décodage, règles métier, écriture) |
| `requirements.txt` | dépendances Python, **versions épinglées** |
| `README.md` | ce fichier — installation, exécution, vérification |

---

## 1. Résultat

| | |
|---|---|
| Métrique de classement | `spellF1` (appariement des possessions en IoU temporel) |
| Score (fenêtre publique 5 min) | `spellF1` = **0,5889** · `macroF1` = **0,7302** |
| Plancher à battre | `spellF1` = 0,2433 |
| Diagnostic (match complet) | **322** possessions · durée moyenne **12,6 s** · part de `D` **35,9 %** |

La méthode est **déterministe** : à entrées identiques, elle produit un fichier **byte-à-byte
identique**. À titre de contrôle d'intégrité, le fichier de référence a pour empreinte :

```
lignes : 199319
md5    : 68a4308e09b540a269661a5dc24e4bbe
```

---

## 2. Prérequis

- **Python 3.9 ou plus récent** (testé sous 3.12.9).
- Les dépendances :

  ```bash
  pip install -r my_submission/requirements.txt
  ```

  Le scoreur officiel n'a besoin que de `numpy` et `pandas` ; la méthode ajoute `scipy`
  (lissage/hystérésis) et `matplotlib` (figure de diagnostic).

---

## 3. Arborescence attendue

Le notebook suppose qu'il est exécuté **depuis ce dossier**, lui-même placé à la racine du dépôt
More Than A Hack (celui qui contient `games.py` et `evaluation/`). Les données de match (gated) vont
dans `data/tactical/`.

```
<depot>/
├── games.py                         # constantes de match fournies
├── evaluation/                      # scoreur + validation fournis
│   ├── validate_submission.py
│   └── possession_metrics.py
├── data/
│   └── tactical/                    # bundle de données (gated, non redistribué)
│       ├── BAR-ATH_tactical_atd.txt          # tracking (joueurs + ballon)
│       ├── BAR-ATH_tactical_atd.xml          # métadonnées (identité d'équipe)
│       ├── BAR-ATH_tactical_smart_tagging.xml
│       └── BAR-ATH_tactical_homography.csv
└── my_submission/                   # CE DOSSIER
    ├── Era_Nova.csv
    ├── possession_inference.ipynb
    ├── common.py
    ├── requirements.txt
    └── README.md
```

> Les fichiers de données ne sont **pas** inclus (le bundle est sous licence, non redistribuable).
> Déposez-le dans `data/tactical/`, ou pointez la variable d'environnement `MTAH_DATA_DIR`
> vers son emplacement.

---

## 4. Reproduire la soumission

### Option A — Notebook (recommandé)

```bash
pip install -r my_submission/requirements.txt
jupyter notebook my_submission/possession_inference.ipynb
```

Puis **Run All** (exécuter toutes les cellules). La cellule §7 réécrit `my_submission/Era_Nova.csv`
et lance le validateur officiel. La première lecture du tracking prend 1 à 2 minutes, puis un cache
local `cache_tracking.npz` accélère les exécutions suivantes.

### Option B — Ligne de commande (sans notebook)

Depuis la racine du dépôt :

```bash
python -c "import sys; sys.path.insert(0,'my_submission'); import common as C; tr=C.load_tracking(); feat=C.build_features(tr); P=dict(hys_win=50,hys_theta=0.4,hys_conf_factor=1.0,d_min_run=40); lab=C.run_method('hysteresis',feat,P); C.write_submission('my_submission/Era_Nova.csv',lab); print(C.diagnostics(lab))"
```

Sortie attendue :

```
{'possessions': 322, 'mean_poss_s': 12.6, 'share_H': 0.329, 'share_A': 0.312, 'share_D': 0.359}
```

---

## 5. Vérifier la soumission

```bash
python evaluation/validate_submission.py my_submission/Era_Nova.csv
```

Le validateur applique **exactement** les contrôles du scoreur officiel (en-tête `frame,label`,
199 319 lignes, trame `0 … 199318`, labels dans `{H, A, D, X}`, aucune cellule vide).

---

## 6. La méthode en bref

**Lissage temporel de la preuve de possession + décodage par hystérésis** (déclencheur de Schmitt),
complété par les règles métier des Lois du Jeu. Aucune probabilité apprise, aucun paramètre ajusté
sur la vérité terrain.

1. **Preuve de possession** `e` (signée `+H` / `−A`) à partir de la distance et du **nombre** de
   joueurs de chaque camp près du ballon (`proxy de contrôle`), pondérée par la **confiance** du
   tracking (le ballon est interpolé sur les trous courts, ignoré sur les longs).
2. **Preuve de ballon mort** `eD` : ballon hors terrain, immobile, trou prolongé près d'une ligne,
   gardien balle en main, jeu collectivement arrêté…
3. **Décodage** : moyenne de la preuve sur ≈ 4 s, `D` par seuil (`eD > |e|`), bascule `H`/`A` par
   hystérésis à seuil **adaptatif à la confiance**. Chaque mi-temps est décodée indépendamment.
4. **Nettoyage** : suppression des segments trop courts, fusion des `H–D–H` (tacle manqué).
5. **Règles métier** : ancrage de `D` sur les redémarrages officiels du Smart Tagging (Lois 8, 13–17)
   et sur les sorties de jeu (Loi 9), bornés par le retour du ballon à l'état jouable.

Les détails, justifications et variantes écartées sont dans `possession_inference.ipynb`.