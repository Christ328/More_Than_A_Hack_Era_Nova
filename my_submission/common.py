"""
Pipeline commun :
  1. lecture du tracking brut (ATD) -> positions en mètres, cache .npz
  2. features par frame : ballon interpolé + confiance, distances joueur-ballon, signaux "ballon mort"
  3. potentiels d'émission log E[t, {H, A, D}]
  4. un décodeur (viterbi / filtrage / lissage / hystérésis) par notebook
  5. écriture du CSV frame,label

Rien ici n'est calibré sur la vérité terrain du match : les paramètres de PARAMS sont des valeurs
de départ raisonnables, à juger par la cohérence (nombre de possessions, part de D), pas par les
5 minutes publiques.
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- constantes du match (DATA.md) -------------------------------------------------------------
N_FRAMES = 199319
PITCH = (105.3, 68.0)
FIRST_HALF = (11185, 90651)          # notée dès le coup d'envoi
SECOND_HALF_GRADED = (110808, 190197)
KICKOFF2 = 113742                    # 110808..113741 : toujours D (FAQ)
SEGMENTS = [FIRST_HALF, (KICKOFF2, SECOND_HALF_GRADED[1])]
FORCED_D = (SECOND_HALF_GRADED[0], KICKOFF2 - 1)
FPS = 25
LETTERS = np.array(["H", "A", "D"])

# Codes du Smart Tagging porteurs d'un camp. Les phases offensives appartiennent à l'équipe qui a
# le ballon ; les phases défensives à l'équipe qui défend, donc à l'adversaire du porteur.
_ST_OFF = {"build up", "progression", "creating chances", "ball in final third",
           "ball in the box", "attacking transition", "long ball"}
_ST_DEF = {"defending in defensive third", "defending in middle third",
           "defending in attacking third", "defensive transition"}

# Types de "SET PIECES" qui sont une remise en jeu (donc une fin de ballon mort). Les valeurs sont
# celles du groupe "Type" du Smart Tagging, en minuscules.
_RESTART_TYPES = {"throw-in", "corner kick", "goal kick", "free kick",
                  "kick off (start)", "kick off (after goal)"}

try:  # les constantes officielles priment si games.py est importable
    import games
    PITCH = tuple(games.GAMES[games.GAME]["pitch"])
    N_FRAMES = int(games.source_frames("tactical", games.GAME))
except Exception:
    games = None


def atd_paths():
    """Chemins ATD : variables d'environnement MM_ATD_META / MM_ATD_RAW, sinon games.asset_paths."""
    if os.environ.get("MM_ATD_META") and os.environ.get("MM_ATD_RAW"):
        return os.environ["MM_ATD_META"], os.environ["MM_ATD_RAW"]
    p = games.asset_paths(source="tactical")
    return p["atd_meta"], p["atd_raw"]


def smart_tagging_path():
    p = games.asset_paths(source="tactical")
    return p["smart_tagging"]


def load_smart_tagging():
    """[(code, start_frame, end_frame, team_code, type)] ou [] si indisponible.

    team_code : 0 = Barcelona (H), 1 = Atletic (A), -1 = N/A.
    type : valeur du groupe "Type" (ex. "throw-in", "corner kick", "kick off (start)") en minuscules,
    "" si absent. C'est ce qui distingue les redémarrages officiels des autres instances SET PIECES.
    """
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(smart_tagging_path()).getroot()
    except Exception:
        return []
    out = []
    for inst in root.iter("instance"):
        code = (inst.findtext("code") or "").strip().lower()
        try:
            s = int(round(float(inst.findtext("start")) * FPS))
            e = int(round(float(inst.findtext("end")) * FPS))
        except (TypeError, ValueError):
            continue
        team_str = ""
        typ_str = ""
        for lab in inst.findall("label"):
            grp = (lab.findtext("group") or "").strip().lower()
            txt = (lab.findtext("text") or "").strip().lower()
            if grp == "team":
                team_str = txt
            elif grp == "type":
                typ_str = txt
        if team_str.startswith("barcelona"):
            tc = 0
        elif team_str.startswith("atletic") or team_str.startswith("athletic"):
            tc = 1
        else:
            tc = -1
        out.append((code, s, e, tc, typ_str))
    return out


# --- 1. lecture du tracking --------------------------------------------------------------------
def team_of_track(meta_path):
    """0 = domicile, 1 = extérieur, pour chaque emplacement de piste (même logique que baseline.py)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], dtype=np.int8)


def load_tracking(use_cache=True, cache=None):
    """dict : ball (n,3) = x,y normalisés + vitesse ; players (n,P,2) normalisés ; team (P,)."""
    cache = cache or os.path.join(HERE, "cache_tracking.npz")
    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    meta, raw = atd_paths()
    team = team_of_track(meta)
    n, P = N_FRAMES, len(team)
    ball = np.full((n, 3), np.nan, np.float32)
    pl = np.full((n, P, 2), np.nan, np.float32)
    with open(raw) as fh:
        for line in fh:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n:
                continue
            b = parts[2].strip()
            if b and b[0] not in "Nn":
                v = b.split(",")
                for k in range(min(3, len(v))):
                    ball[i, k] = float(v[k])
            for s, g in enumerate(parts[1].split(";")):
                if s >= P:
                    break
                if not g or g[0] in "Nn":
                    continue
                v = g.split(",")
                pl[i, s, 0] = float(v[0])
                pl[i, s, 1] = float(v[1])
    out = {"ball": ball, "players": pl, "team": team}
    if use_cache:
        np.savez_compressed(cache, **out)   # dérivé des données : NE PAS versionner (voir .gitignore)
    return out


# --- 2/3. features et potentiels d'émission ----------------------------------------------------
PARAMS = dict(
    # ballon manquant
    G1=25, G2=75, c1=0.7, c2=0.35,       # trou <=1 s : confiance .7 ; <=3 s : .35 ; au-delà : 0
    # contrôle du ballon
    # near_w : poids du déséquilibre numérique (nH-nA) autour du ballon. Réglé à 1.0 (au lieu de
    # 0.3) : le nombre de joueurs proches du ballon est un signal de possession plus fiable que
    # le seul joueur le plus proche (le tracker perd le ballon et le duplique souvent).
    r0=4.0, zc=6.0, zs=3.0, a=1.0, near_r=3.0, near_w=1.0,
    # ballon mort
    b_out=8.0, still_v=0.5, still_win=37, b_still=4.5,
    gap_min=40, edge_m=2.5, gap_full=100, b_gap=4.0, b0=1.5,
    # ballon mort : signaux additionnels
    b_cluster=0.0, cluster_r=10.0, cluster_n=7,   # regroupement serré de joueurs
    b_veldrop=0.0, vdrop_thresh=4.0,             # chute brutale de vitesse du ballon
    b_longgap=0.0, long_gap=300,                 # trou tres long (desactive par defaut)
    # ballon mort : "le jeu est collectivement arrete". Loi du jeu / physique : quand le jeu est
    # arrete, aucun joueur ne court (on marche pour se replacer) ; quand il est en jeu, il y a
    # toujours des courses (porteur, pressers, appels). On mesure la fraction de joueurs se
    # deplacant au-dessus d'une allure de course, lissee dans le temps (un arret de jeu dure
    # plusieurs secondes ; un simple ralentissement non). act_thresh = part de joueurs en course
    # en dessous de laquelle on considere le jeu arrete (0.10 ~ un joueur sur dix).
    b_activity=0.0, act_jog=2.0, act_thresh=0.10, act_win=25,
    # ballon mort "profond" : pres d'une ligne de but, quasi immobile, un seul camp a proximite.
    # Actif : capte les arrets longs que les autres signaux ratent (ballon visible ET immobile),
    # typiquement un gardien balle en main. Ameliore la precision ET le rappel de D.
    b_keeper=8.0, goal_r=8.0, keep_v=2.0, keep_close=10.0, keep_far=12.0,
    # transitions (probabilité par frame) — asymétriques
    p_ha=1.5e-3, p_ah=2.5e-3, p_hd=2e-3, p_ad=2e-3, p_dh=3e-3, p_da=3e-3, temp=1.0,
    # lissage + hystérésis. Échelle de lissage = ~2 s : une possession dure plusieurs secondes,
    # donc lisser sur 2 s efface le scintillement frame à frame sans effacer les possessions.
    # hys_theta = preuve adimensionnée nécessaire à une bascule H/A (voir smooth_hysteresis).
    hys_win=50, hys_theta=0.4, hys_conf_factor=1.0, d_min_run=40,
    # nettoyage final — différent pour D et pour H/A
    min_run_ha=15, min_run_d=8,
    # Vote temporel : chaque frame prend l'étiquette majoritaire de la fenêtre
    # [t - tvote_left, t + tvote_right]. C'est un filtre modal sur la séquence décodée : il efface
    # les blips isolés (frontières ambiguës, confiance faible) qui fragmentent les possessions.
    # Corrige la position du centre vis-à-vis d'un éventuel retard de transition. 0 = désactivé.
    tvote_left=0, tvote_right=0,
    # fusion H-D-H / A-D-A : D plus court que d_fuse_run entouré de la même équipe -> retiré
    d_fuse_run=20,
    # biais d'équipe du Smart Tagging sur l'émission (0 = ignoré)
    st_weight=0.0,
    # Levier A — ancrage du ballon mort (D) sur les redémarrages officiels du Smart Tagging. On
    # force D jusqu'à ce que le ballon redevienne jouable (observé et roulant, de façon tenue),
    # détecté sur le tracking lui-même : la fin de la fenêtre n'est plus une constante arbitraire.
    st_restart_force=True, live_v=0.5, live_run=4,
    # Loi 9 — le ballon est hors du jeu dès qu'il franchit entièrement une ligne ; le jeu ne reprend
    # qu'à la remise en jeu officielle. On force D du premier frame observé hors du terrain jusqu'à
    # la prochaine remise en jeu Smart Tagging, borné à out_max_s (durée d'une fenêtre SET PIECES
    # fixe) : au-delà, l'observation "hors terrain" est un artefact du tracker, pas une sortie.
    st_out_force=True, out_min_run=5, out_max_s=40.0,
    # Levier A (but, Loi 8) — du franchissement de la ligne au coup d'envoi, le jeu est arrêté. La
    # fenêtre "goals" du Smart Tagging est exactement cet intervalle : on la force en D telle quelle.
    # Possession H/A — germe d'une règle métier : ne basculer de camp que si le joueur le plus
    # proche est nettement plus proche que celui de l'autre camp (|dH - dA| > ha_clean_m).
    # DÉSACTIVÉE : geler l'état en zone disputée sur-fusionne les possessions (341 -> 255 sur le
    # match) — une possession en zone disputée reste une possession. Conservée pour mémoire.
    ha_clean_m=5.0, ha_clean_gate=False,
    # Levier C — rejet des observations de ballon physiquement impossibles. 2.0 m/frame ≈ 50 m/s,
    # très au-dessus de la frappe la plus rapide : ne coupe que les artefacts du tracker.
    ball_max_step=2.0,
    # Levier D — modèle semi-Markov : durée minimale explicite par état. Un HMM à transitions
    # géométriques autorise des séjours d'une seule frame ; la physique du jeu non. semi_mode :
    #   "off"     : HMM géométrique, on garde les seuils min_run_*/d_min_run.
    #   "model"   : min = semi_dur_frac × E[dwell], E[dwell] = 1/p_sortie que trans_matrix implique.
    #               Entièrement déterminé par le modèle, donc indépendant de toute donnée.
    #   "physics" : bornes tirées du jeu (possession = contrôle soutenu ; D = redémarrage non
    #               instantané). Voir semi_min_dwell().
    semi_mode="off", semi_dur_frac=0.25, semi_min_ha_s=1.0, semi_min_d_s=2.0,
    # Levier B — élagage MDL du nombre de segments. On retire un segment intérieur quand le gain de
    # description ne justifie pas sa présence (ΔDL < mdl_penalty). Pénalité en nats, mêmes unités
    # que la vraisemblance du modèle.
    mdl_on=False, mdl_penalty=0.0, mdl_max_len=100,
    # --- leviers implémentés mais NON appliqués (valeurs 0/False) ---
    # Règle : aucune valeur n'est choisie en regardant la fenêtre publique. Ces leviers n'ont pas
    # de justification indépendante suffisante, donc ils restent désactivés.
    d_dilate=0,                          # élargissement du masque D (valeur publique révoquée)
    b_noplayer=0.0, noplayer_r=14.0,     # ballon visible mais aucun joueur proche
    latency_shift=0,                     # décalage global des frontières (valeur publique révoquée)
    ha_fuse_run=0,                       # fusion des blips H/A
    hold_at_line=False,                  # interpolation contrainte près de la touche
)


def build_features(tr, P=None, impute=None):
    P = {**PARAMS, **(P or {})}
    L, W = PITCH
    n = len(tr["ball"])
    bx, by, bs = tr["ball"][:, 0], tr["ball"][:, 1], tr["ball"][:, 2].copy()
    obs = ~np.isnan(bx)

    # Levier C — le tracker perd puis ré-acquiert le ballon, parfois très loin de sa dernière
    # position. Le ballon a une vitesse bornée (~50 m/s), donc toute observation impliquant une
    # vitesse supérieure depuis l'observation valide précédente est un artefact : on l'écarte
    # (NaN) avant interpolation.
    idx = np.arange(n)
    prev0 = np.maximum.accumulate(np.where(obs, idx, -1))
    k0 = np.clip(prev0, 0, n - 1)
    span = np.maximum(idx - prev0, 1)
    jump = np.hypot((bx - bx[k0]) * L, (by - by[k0]) * W)
    bad = obs & (prev0 >= 0) & (jump > P["ball_max_step"] * span)
    if bad.any():
        bx = bx.copy(); by = by.copy()
        bx[bad] = np.nan; by[bad] = np.nan
        obs = ~np.isnan(bx)

    out_pitch = obs & ((bx < 0) | (bx > 1) | (by < 0) | (by > 1))

    prev_i = np.maximum.accumulate(np.where(obs, idx, -1))
    next_i = np.minimum.accumulate(np.where(obs, idx, 10 ** 9)[::-1])[::-1]
    valid_gap = (prev_i >= 0) & (next_i < 10 ** 9)
    gap_len = np.where(obs, 0, np.where(valid_gap, next_i - prev_i, 10 ** 6))

    # interpolation linéaire à l'intérieur des trous, maintien aux extrémités.
    # Méthode 4 : un trou dont les deux extrémités sont près d'une ligne de touche ne se traverse
    # pas en ligne droite ; on fige le ballon à son dernier point connu (signature d'une touche).
    pos = pd.DataFrame({"x": bx * L, "y": by * W}).interpolate(limit_area="inside").ffill().bfill().values
    if P["hold_at_line"]:
        def _edge(xy):
            return np.minimum(np.minimum(xy[:, 0], L - xy[:, 0]), np.minimum(xy[:, 1], W - xy[:, 1]))
        ep = pos[np.clip(prev_i, 0, n - 1)]
        en = pos[np.clip(next_i, 0, n - 1)]
        near_line = (~obs) & (_edge(ep) < P["edge_m"]) & (_edge(en) < P["edge_m"])
        pos = np.where(near_line[:, None], ep, pos)

    if impute is not None:
        # Ballon comblé par un modèle auto-supervisé (option 2) : sur les trous du tracker (ballon
        # non observé), on remplace la position interpolée par une estimation apprise sur la
        # configuration des joueurs. Les positions réellement observées ne sont jamais touchées.
        impute = np.asarray(impute, np.float32)
        pos = np.where((~obs)[:, None], impute, pos)

    conf = np.where(obs, 1.0, np.where(gap_len <= P["G1"], P["c1"], np.where(gap_len <= P["G2"], P["c2"], 0.0)))

    def edge(p):
        return np.minimum(np.minimum(p[:, 0], L - p[:, 0]), np.minimum(p[:, 1], W - p[:, 1]))
    e_prev = edge(pos[np.clip(prev_i, 0, n - 1)])
    e_next = edge(pos[np.clip(next_i, 0, n - 1)])
    edge_gap = np.where(valid_gap, np.minimum(e_prev, e_next), 99.0)
    gap_out = ((~obs) & (gap_len > P["gap_min"]) & (edge_gap < P["edge_m"])) * np.minimum(1.0, gap_len / P["gap_full"])

    # ballon quasi immobile. La colonne vitesse du fichier n'est renseignée que sur ~5 % des
    # frames : l'ancien repli ne se déclenchait que si *toute* la colonne était NaN (jamais), donc
    # le signal "ballon immobile" était mort. On la complète frame par frame par la dérivée des
    # positions observées.
    sp_pos = np.r_[0, np.hypot(np.diff(pos[:, 0]), np.diff(pos[:, 1]))] * FPS
    bs = np.where(np.isnan(bs), np.where(obs, sp_pos, np.nan), bs)
    rmax = pd.Series(bs).rolling(P["still_win"], center=True, min_periods=5).max().values
    still = np.where(np.isnan(rmax), False, rmax < P["still_v"]) & (conf > 0)
    # "ballon jouable" : observé et roulant. Borne la fin des fenêtres de ballon mort (levier A).
    live = obs & np.where(np.isnan(bs), False, bs > P["live_v"])

    # distances joueurs-ballon, par équipe + comptage de cluster
    pl = tr["players"] * np.array([L, W], dtype=np.float32)
    is_h = tr["team"] == 0
    dH = np.full(n, np.inf, np.float32); dA = dH.copy()
    nH = np.zeros(n, np.int16); nA = nH.copy(); nClust = nH.copy()
    step = 20000
    for s in range(0, n, step):
        d = np.hypot(pl[s:s + step, :, 0] - pos[s:s + step, None, 0],
                     pl[s:s + step, :, 1] - pos[s:s + step, None, 1])
        d = np.where(np.isnan(d), np.inf, d)
        dH[s:s + step] = d[:, is_h].min(axis=1)
        dA[s:s + step] = d[:, ~is_h].min(axis=1)
        nH[s:s + step] = (d[:, is_h] < P["near_r"]).sum(axis=1)
        nA[s:s + step] = (d[:, ~is_h] < P["near_r"]).sum(axis=1)
        nClust[s:s + step] = (d < P["cluster_r"]).sum(axis=1)
    cluster = (nClust >= P["cluster_n"]) & (conf > 0)

    # "le jeu est collectivement arrete" : fraction de joueurs en course (deplacement > act_jog),
    # lissee sur act_win frames. NaN (joueur non suivi) exclu de la moyenne.
    pv = np.hypot(np.diff(pl[:, :, 0], axis=0), np.diff(pl[:, :, 1], axis=0)) * FPS
    pv = np.vstack([np.full((1, pv.shape[1]), np.nan, np.float32), pv])
    with np.errstate(invalid="ignore"):
        running = np.nanmean(pv > P["act_jog"], axis=1)
    running = np.where(np.isfinite(running), running, 0.0)
    act = pd.Series(running).rolling(int(P["act_win"]), center=True, min_periods=1).mean().values
    inactive = act < P["act_thresh"]

    # vitesse du ballon et chute brutale (tacle / interception) — bs est déjà complété plus haut
    bs_smooth = pd.Series(bs).rolling(5, center=True, min_periods=2).mean().values
    accel = np.r_[0, np.diff(bs_smooth)]
    veldrop = (accel < -P["vdrop_thresh"]) & (conf > 0)
    veldrop = pd.Series(veldrop.astype(np.float32)).rolling(7, center=True, min_periods=1).max().values > 0.5

    # trou très long du ballon -> probablement un arrêt de jeu
    longgap = (~obs) & (gap_len > P["long_gap"])

    # ballon mort "profond" : ballon près d'une ligne de but, quasi immobile, un seul camp à
    # proximité (l'autre loin). Signature d'un gardien balle en main ou d'un dégagement préparé :
    # un arrêt de jeu que les autres signaux ratent quand le ballon reste visible et immobile.
    near_goal = obs & ((pos[:, 0] < P["goal_r"]) | (pos[:, 0] > L - P["goal_r"]))
    close_side = np.minimum(dH, dA) < P["keep_close"]
    far_side = np.maximum(dH, dA) > P["keep_far"]
    slow = np.where(np.isnan(bs), False, bs < P["keep_v"])
    keeper = near_goal & close_side & far_side & slow & (conf > 0)

    # méthode 1 : ballon visible mais aucun joueur dans un rayon raisonnable -> ballon qui roule
    # hors du jeu ou arrêt que les autres signaux manquent.
    noplayer = obs & (np.minimum(dH, dA) > P["noplayer_r"])

    dmin = np.minimum(dH, dA)
    ok = np.isfinite(dmin) & (conf > 0)
    # Signaux de "qui a le ballon", utiles aux règles métier. On exige que le ballon soit OBSERVÉ :
    # sur un trou du tracker, la position est interpolée et "proche" d'un joueur par construction,
    # ce qui ferait croire à tort que le jeu continue.
    #   clean : le joueur le plus proche est nettement plus proche que celui de l'autre camp ;
    #           le camp du porteur est alors fiable, on peut basculer H/A sans ambiguïté.
    clean = obs & np.isfinite(dmin) & (np.abs(dH - dA) > P["ha_clean_m"])
    w = np.where(ok, np.exp(-np.where(ok, dmin, 0) / P["r0"]), 0.0)
    z = np.clip(np.where(ok, np.minimum(dA, 99) - np.minimum(dH, 99), 0), -P["zc"], P["zc"]) / P["zs"]
    z = z + P["near_w"] * np.clip(nH - nA, -3, 3)
    e = P["a"] * conf * w * z                       # > 0 : domicile, < 0 : extérieur

    # biais équipe du Smart Tagging. Les phases "offensives" désignent l'équipe qui a le ballon,
    # les phases "défensives" désignent l'équipe qui défend : c'est donc l'adversaire qui a le
    # ballon, d'où le signe opposé. Sans cette distinction, "Defending in Middle Third (Barça)"
    # pousserait à tort vers le domicile.
    st_e = np.zeros(n, np.float32)
    for code, s, e_f, tc, _typ in load_smart_tagging():
        if tc < 0:
            continue              # "N/A" : le code seul n'indique pas le camp
        if code in _ST_OFF:
            sign = 1.0
        elif code in _ST_DEF:
            sign = -1.0
        else:
            continue
        s = max(0, min(n, s)); e_f = max(0, min(n, e_f))
        if s >= e_f:
            continue
        st_e[s:e_f] += sign * P["st_weight"] * (1.0 if tc == 0 else -1.0)
    e = e + st_e

    eD = (P["b_out"] * out_pitch + P["b_still"] * still + P["b_gap"] * gap_out
          + P["b_cluster"] * cluster + P["b_veldrop"] * veldrop + P["b_longgap"] * longgap
          + P["b_activity"] * inactive + P["b_keeper"] * keeper + P["b_noplayer"] * noplayer)
    logE = np.stack([e, -e, eD - P["b0"]], axis=1) * P["temp"]
    return dict(logE=logE, e=e, eD=eD - P["b0"], conf=conf, obs=obs, still=still,
                out=out_pitch, gap_out=gap_out, pos=pos, cluster=cluster, live=live,
                veldrop=veldrop, longgap=longgap, activity=act, inactive=inactive,
                keeper=keeper, noplayer=noplayer, clean=clean)


# --- 4. décodeurs ------------------------------------------------------------------------------
def trans_matrix(P):
    """Matrice de transition 3x3 asymétrique : H=0, A=1, D=2.

    p_ha : H->A, p_ah : A->H (Barça a plus de possession, donc p_ah > p_ha),
    p_hd/p_ad : H/A -> D, p_dh/p_da : D -> H/A.
    """
    p_ha, p_ah = P["p_ha"], P["p_ah"]
    p_hd, p_ad = P["p_hd"], P["p_ad"]
    p_dh, p_da = P["p_dh"], P["p_da"]
    return np.array([[1 - p_ha - p_hd, p_ha, p_hd],
                     [p_ah, 1 - p_ah - p_ad, p_ad],
                     [p_dh, p_da, 1 - p_dh - p_da]])


def viterbi(logE, P):
    """Maximum de probabilité jointe (max-produit en log) sur les 3 états."""
    logT = np.log(trans_matrix(P))
    n = len(logE)
    delta = np.zeros((n, 3)); back = np.zeros((n, 3), np.int8)
    delta[0] = logE[0] - np.log(3)
    for t in range(1, n):
        cand = delta[t - 1][:, None] + logT
        back[t] = cand.argmax(axis=0)
        delta[t] = cand.max(axis=0) + logE[t]
    path = np.zeros(n, np.int8); path[-1] = delta[-1].argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _emis(logE):
    return np.exp(logE - logE.max(axis=1, keepdims=True))


def forward_filter(logE, P):
    """Filtrage (causal) : P(état_t | obs_1..t). Utile pour un usage en ligne ; argmax à chaque t."""
    T = trans_matrix(P); B = _emis(logE); n = len(B)
    alpha = np.zeros((n, 3)); a = B[0] / B[0].sum(); alpha[0] = a
    for t in range(1, n):
        a = (a @ T) * B[t]; a /= a.sum(); alpha[t] = a
    return alpha


def forward_backward(logE, P):
    """Lissage (non causal) : P(état_t | toutes les obs), puis argmax marginal."""
    T = trans_matrix(P); B = _emis(logE); n = len(B)
    alpha = forward_filter(logE, P)
    beta = np.ones((n, 3))
    for t in range(n - 2, -1, -1):
        b = T @ (B[t + 1] * beta[t + 1]); beta[t] = b / b.sum()
    post = alpha * beta
    return post / post.sum(axis=1, keepdims=True)


def _runs(c):
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    return starts, np.r_[starts[1:], len(c)]


def drop_short_runs(c, min_len, passes=3):
    """Remplace tout segment de moins de min_len frames par le label du segment précédent.

    min_len : scalaire (même seuil partout), tuple (min_ha, min_d), ou séquence de taille 3 indexée
    par état [H, A, D] — c'est cette dernière forme qu'utilise le modèle semi-Markov (levier D).
    """
    c = c.copy()
    if isinstance(min_len, (tuple, list, np.ndarray)) and len(min_len) == 3:
        thr_by_state = np.asarray(min_len, dtype=float)
    elif isinstance(min_len, tuple):
        min_ha, min_d = min_len
        thr_by_state = np.array([min_ha, min_ha, min_d], dtype=float)
    else:
        thr_by_state = np.full(3, float(min_len))
    if (thr_by_state <= 1).all():
        return c
    for _ in range(passes):
        starts, ends = _runs(c)
        for k, (s, e) in enumerate(zip(starts, ends)):
            label = c[s]
            if e - s < thr_by_state[label]:
                if k > 0:
                    c[s:e] = c[s - 1]
                elif len(starts) > 1:
                    c[s:e] = c[e]
    return c


def fuse_short_dead(c, d_run, ha_run=0, passes=2):
    """Fusionne un segment court encadré par la même équipe.

    H ... D(court) ... H  ->  H ... H   (arrêt mal détecté, probablement un tacle)
    H ... A(court) ... H  ->  H ... H   (bascule de possession manquée) — activé si ha_run > 0
    A ... D/A(court) ... A -> A ... A

    Un segment séparant deux équipes différentes est conservé (c'est un vrai arrêt / une vraie
    bascule). d_run s'applique aux segments D, ha_run aux segments H/A.
    """
    c = c.copy()
    for _ in range(passes):
        starts, ends = _runs(c)
        for k in range(1, len(starts) - 1):
            s, e = starts[k], ends[k]
            mid = c[s]
            thr = d_run if mid == 2 else ha_run
            if thr > 0 and e - s < thr:
                left = c[starts[k - 1]]
                right = c[ends[k]]
                if left == right and left != mid:
                    c[s:e] = left
    return c


def smooth_hysteresis(feat, P, reverse=False):
    """Lissage + décodage : moyenne glissante de la preuve, D par seuil, bascule H/A par hystérésis.

    Le seuil d'hystérésis est adaptatif : plus la confiance est haute, plus le seuil est bas
    (on réagit vite à un changement clair) ; plus la confiance est basse, plus le seuil est haut
    (on résiste au bruit).

    reverse=True parcourt les frames de droite à gauche : c'est une lecture indépendante du même
    modèle (l'hystérésis est directionnel), utilisée par le décodage par vote.
    """
    from scipy.ndimage import uniform_filter1d
    e, eD, conf = feat["e"], feat["eD"], feat["conf"]
    es = uniform_filter1d(e.astype(float), size=2 * P["hys_win"] + 1, mode="nearest")
    conf_s = uniform_filter1d(conf.astype(float), size=2 * P["hys_win"] + 1, mode="nearest")
    # Règle métier H/A : on n'autorise une bascule de camp que là où le joueur le plus proche est
    # nettement plus proche que l'autre camp (feat["clean"]). En zone disputée, on conserve l'état.
    gate = None
    if P.get("ha_clean_gate", True) and feat.get("clean") is not None:
        gate = feat["clean"]
    isD = drop_short_runs((eD > np.abs(e)).astype(np.int8), P["d_min_run"]).astype(bool)
    if P["d_dilate"] > 0:
        # méthode 1 : élargit le masque D de part et d'autre pour absorber les frames de frontière
        # où la preuve était limite (les vrais arrêts durent > 5 s, une poignée de frames est sûr).
        from scipy.ndimage import binary_dilation
        isD = binary_dilation(isD, iterations=int(P["d_dilate"]))
    codes = np.zeros(len(e), np.int8)
    state = 0 if es[np.argmax(np.abs(es))] >= 0 else 1
    th_base = P["hys_theta"]
    cf = P["hys_conf_factor"]
    order = range(len(e) - 1, -1, -1) if reverse else range(len(e))
    for t in order:
        th = th_base * (1.0 - cf * conf_s[t])
        th = max(th, th_base * 0.3)
        if not isD[t] and (gate is None or gate[t]):
            if es[t] > th:
                state = 0
            elif es[t] < -th:
                state = 1
        codes[t] = 2 if isD[t] else state
    return codes


def _vote(stack, ref=None):
    """Vote majoritaire d'une pile d'étiquettes (m, n). Égalité -> étiquette de `ref` si elle fait
    partie des mieux placées, sinon la plus petite."""
    cnts = np.stack([(stack == lab).sum(axis=0) for lab in (0, 1, 2)])
    mx = cnts.max(axis=0)
    out = cnts.argmax(axis=0).astype(np.int8)
    if ref is not None:
        tie = (cnts == mx).sum(axis=0) > 1
        ri = ref.astype(int)
        ref_is_max = cnts[ri, np.arange(stack.shape[1])] == mx
        out = np.where(tie & ref_is_max, ref, out)
    return out.astype(np.int8)


def _reader(name, feat, P):
    if name == "hysL":
        return smooth_hysteresis(feat, P, reverse=False)
    if name == "hysR":
        return smooth_hysteresis(feat, P, reverse=True)
    if name == "viterbi":
        return viterbi(feat["logE"], P)
    if name == "lissage":
        return forward_backward(feat["logE"], P).argmax(axis=1).astype(np.int8)
    if name == "filtrage":
        return forward_filter(feat["logE"], P).argmax(axis=1).astype(np.int8)
    raise ValueError(name)


def vote_decode(feat, P):
    """Décodage par vote de plusieurs lectures du MÊME modèle de Markov.

    Chaque membre est une lecture du modèle ; elles votent frame par frame. Les lectures
    directionnelles ("hysL" gauche->droite, "hysR" droite->gauche) apportent le vote gauche/droite.
    Aucun paramètre n'est réglé sur les données : vote majoritaire, égalité -> premier membre.
    """
    members = P.get("vote_members", ("hysL", "hysR", "viterbi"))
    stack = []
    for m in members:
        s = _reader(m, feat, P)
        if P.get("vote_preclean", False):
            # Chaque votant est nettoyé AVANT le vote : un votant non lissé scintille frame à frame,
            # et un vote majoritaire de votants qui scintillent produit... du scintillement.
            s = drop_short_runs(s, (P["min_run_ha"], P["min_run_d"]))
        stack.append(s)
    stack = np.stack(stack)
    ref = stack[0] if members else None
    return _vote(stack, ref)


def temporal_vote(codes, left, right):
    """Vote temporel : chaque frame prend l'étiquette majoritaire de [t-left, t+right].

    Filtre modal (non causal) sur la séquence décodée. Il supprime les blips isolés plus courts que
    la fenêtre — ce sont eux qui fragmentent les possessions — en conservant les étiquettes d'accord
    majoritaire. En cas d'égalité (deux camps à égalité de voix dans la fenêtre), on garde l'étiquette
    d'origine : on ne bascule que si la fenêtre parle clairement.
    """
    if left <= 0 and right <= 0:
        return codes
    c = codes.astype(np.int8)
    n = len(c)
    idx = np.arange(n)
    lo = np.maximum(0, idx - left)
    hi = np.minimum(n, idx + right + 1)
    cum = np.zeros((3, n + 1), np.int32)
    for lab in range(3):
        cum[lab, 1:] = np.cumsum(c == lab)
    cnt = np.stack([cum[lab][hi] - cum[lab][lo] for lab in range(3)])
    mx = cnt.max(axis=0)
    best = cnt.argmax(axis=0).astype(np.int8)
    tie = (cnt == mx).sum(axis=0) > 1
    return np.where(tie, c, best).astype(np.int8)


def _slice(feat, a, b):
    return {k: (v[a:b] if isinstance(v, np.ndarray) and len(v) == N_FRAMES else v) for k, v in feat.items()}


def _restart_frame(feat, s, e, P):
    """Premier frame de [s, e] où le ballon redevient jouable (observé et roulant, de façon tenue).

    Renvoie e si aucune reprise de jeu n'est détectée dans la fenêtre officielle.
    """
    live = feat.get("live")
    if live is None:
        return e
    run = 0
    need = int(P.get("live_run", 4))
    for t in range(s, e):
        if live[t]:
            run += 1
            if run >= need:
                return t - need + 1
        else:
            run = 0
    return e


def _expected_dwell(P):
    """Durée de séjour moyenne (frames) par état, impliquée par la matrice de transition.

    Un HMM à transitions géométriques passe en moyenne 1/p_sortie frames dans un état avant d'en
    changer. C'est la seule échelle de durée que le modèle fournit lui-même : elle ne dépend
    d'aucune donnée, et en particulier pas des 5 minutes publiques.
    """
    T = trans_matrix(P)
    leave = np.clip(1.0 - np.diag(T), 1e-9, None)
    return 1.0 / leave


def semi_min_dwell(P):
    """Durées minimales (frames) par état [H, A, D] pour le modèle semi-Markov, ou None si inactif.

    "model"   : une fraction de la durée de séjour moyenne déjà impliquée par trans_matrix. Rien
                n'est choisi en observant la vérité terrain : la contrainte prolonge le modèle.
    "physics" : bornes tirées du jeu — une possession exige un contrôle soutenu (H/A), un
                redémarrage n'est jamais instantané (D).
    """
    mode = P.get("semi_mode", "off")
    if mode == "model":
        return _expected_dwell(P) * P["semi_dur_frac"]
    if mode == "physics":
        return np.array([P["semi_min_ha_s"], P["semi_min_ha_s"], P["semi_min_d_s"]]) * FPS
    return None


def _merge_delta(logE, logT, seg, s, e, cand):
    """Δ de la longueur de description (DL) si l'on absorbe [s, e) dans l'état cand.

    DL(seg) = -Σ_t log E_t[seg_t] - Σ_t log T[seg_t, seg_{t+1}] (émissions + coût des transitions).
    On ne recalcule que la fenêtre modifiée ; les frontières gauche (s-1→s) et droite (e-1→e) sont
    les seules transitions touchées, les transitions internes (toutes seg_s→seg_s) disparaissent.
    """
    old = seg[s]
    em = float(logE[s:e, cand].sum() - logE[s:e, old].sum())
    ts = -logT[old, old] * (e - s - 1)
    if s > 0:
        ts += logT[seg[s - 1], cand] - logT[seg[s - 1], old]
    if e < len(seg):
        ts += logT[cand, seg[e]] - logT[old, seg[e]]
    return -(em + ts)


def mdl_merge(feat, seg, P):
    """Levier B — élagage MDL du nombre de segments.

    On retire itérativement le segment intérieur dont le retrait augmente le moins la longueur de
    description, tant que cette augmentation reste sous la pénalité mdl_penalty (nats) : au-delà,
    garder le segment coûte moins cher que le décrire. Le segment est absorbé par le voisin gauche
    ou droit, au choix de celui qui augmente le moins DL — jamais d'après la vérité terrain.
    Les segments longs ne sont pas candidats (mdl_max_len) : un segment étayé ne peut pas disparaître.
    """
    logE = feat["logE"]
    logT = np.log(trans_matrix(P))
    seg = seg.copy()
    lam = P["mdl_penalty"]
    max_len = int(P["mdl_max_len"])
    while True:
        starts, ends = _runs(seg)
        if len(starts) < 3:
            break
        best = None
        for k in range(1, len(starts) - 1):
            s, e = int(starts[k]), int(ends[k])
            if e - s > max_len:
                continue
            for cand in {int(seg[s - 1]), int(seg[e])} - {int(seg[s])}:
                d = _merge_delta(logE, logT, seg, s, e, cand)
                if best is None or d < best[0]:
                    best = (d, s, e, cand)
        if best is None or best[0] >= lam:
            break
        _, s, e, cand = best
        seg[s:e] = cand
    return seg


METHODS = {
    "viterbi":   lambda f, P: viterbi(f["logE"], P),
    "filtrage":  lambda f, P: forward_filter(f["logE"], P).argmax(axis=1).astype(np.int8),
    "lissage":   lambda f, P: forward_backward(f["logE"], P).argmax(axis=1).astype(np.int8),
    "hysteresis": smooth_hysteresis,
    "vote":      vote_decode,
}


def apply_match_rules(codes, feat, P):
    """Règles métier des Lois du Jeu, appliquées à un décodage QUELCONQUE (chaîne de Markov ou non).

    - début de 2e mi-temps : toujours un arrêt de jeu ;
    - Levier A (Lois 8, 13-17) : les redémarrages officiels datés par le Smart Tagging (touche,
      corner, coup franc, six mètres, coup d'envoi, but) sont des arrêts de jeu ; on force D sur la
      fenêtre officielle, bornée au moment où le ballon redevient jouable d'après le tracking ;
    - Loi 9 : ballon sorti du jeu -> D jusqu'à la remise en jeu officielle (borné à out_max_s).

    Fonction pure (renvoie `codes`), réutilisable pour comparer des décodeurs alternatifs.
    """
    codes = codes.copy()
    codes[FORCED_D[0]:FORCED_D[1] + 1] = 2
    if P.get("st_restart_force", True):
        insts = load_smart_tagging()
        kick_starts = [s for code, s, e, tc, typ in insts
                       if code == "set pieces" and typ.startswith("kick off")]
        for code, s, e, tc, typ in insts:
            s = max(0, min(N_FRAMES, s))
            e = max(0, min(N_FRAMES, e))
            if e <= s:
                continue
            if code == "goals":
                nxt = [k for k in kick_starts if s <= k <= e + 500]
                end = nxt[0] if nxt else e
                if end > s:
                    codes[s:end] = 2
                continue
            if not (code == "set pieces" and typ in _RESTART_TYPES):
                continue
            stop = _restart_frame(feat, s, e, P)
            if stop > s:
                codes[s:stop] = 2
    if P.get("st_out_force", False):
        outm = feat.get("out")
        if outm is not None:
            outm = np.asarray(outm, dtype=bool)
            restart_starts = sorted(s for code, s, e, tc, typ in load_smart_tagging()
                                    if code == "set pieces" and typ in _RESTART_TYPES)
            max_len = int(P["out_max_s"] * FPS)
            starts, ends = _runs(outm)
            for s, e in zip(starts, ends):
                if not outm[s] or (e - s) < int(P["out_min_run"]):
                    continue
                nxt = [r for r in restart_starts if r >= s]
                if not nxt or (nxt[0] - s) > max_len:
                    continue
                codes[s:nxt[0]] = 2
    return codes


def run_method(name, feat, P=None):
    """Décode chaque mi-temps séparément et renvoie un tableau de lettres H/A/D/X (longueur N_FRAMES)."""
    P = {**PARAMS, **(P or {})}
    codes = np.full(N_FRAMES, -1, np.int8)
    for a, b in SEGMENTS:
        seg_feat = _slice(feat, a, b + 1)
        seg = METHODS[name](seg_feat, P)
        seg = drop_short_runs(seg, (P["min_run_ha"], P["min_run_d"]))
        seg = fuse_short_dead(seg, P["d_fuse_run"], P["ha_fuse_run"])
        # Levier D — semi-Markov : durée minimale explicite par état (model / physics / off).
        dwell = semi_min_dwell(P)
        if dwell is not None:
            seg = drop_short_runs(seg, dwell)
        # Levier B — élagage MDL du nombre de segments.
        if P["mdl_on"]:
            seg = mdl_merge(seg_feat, seg, P)
        # Vote temporel : étape finale du décodage, avant les ancrages officiels (SMART TAGGING),
        # qui restent prioritaires. Nettoyage modal des blips résiduels du modèle.
        if P["tvote_right"] > 0 or P["tvote_left"] > 0:
            seg = temporal_vote(seg, P["tvote_left"], P["tvote_right"])
        if P["latency_shift"] > 0:
            # méthode 2 : retard global des frontières (nos transitions partaient ~7 frames trop tôt)
            k = int(P["latency_shift"])
            seg = np.concatenate([np.full(k, seg[0], np.int8), seg[:-k]])
        codes[a:b + 1] = seg
    codes = apply_match_rules(codes, feat, P)
    labels = np.full(N_FRAMES, "X", dtype="<U1")
    for k, l in enumerate(LETTERS):
        labels[codes == k] = l
    return labels


# --- 5. sorties, diagnostics, scores -----------------------------------------------------------
def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path


def diagnostics(labels):
    """Nombre de possessions, durée moyenne, part de D sur les frames notées."""
    out = {}
    tot = 0; poss = []
    for a, b in SEGMENTS:
        seg = labels[a:b + 1]
        starts, ends = _runs(seg)
        for s, e in zip(starts, ends):
            if seg[s] in ("H", "A"):
                poss.append(e - s)
        tot += len(seg)
    graded = np.concatenate([labels[a:b + 1] for a, b in SEGMENTS] + [labels[FORCED_D[0]:FORCED_D[1] + 1]])
    out["possessions"] = len(poss)
    out["mean_poss_s"] = round(float(np.mean(poss)) / FPS, 1) if poss else 0
    out["share_H"] = round(float(np.mean(graded == "H")), 3)
    out["share_A"] = round(float(np.mean(graded == "A")), 3)
    out["share_D"] = round(float(np.mean(graded == "D")), 3)
    return out


def audit(labels):
    """Contrôle interne SANS vérité terrain : couverture des redémarrages officiels.

    Une remise en jeu datée par le Smart Tagging est, par définition, un arrêt de jeu : elle doit
    tomber dans une zone de ballon mort (D) prédite. Ce n'est pas une mesure de justesse (aucune
    ne l'est sans GT) mais une vérification de cohérence, comparable d'une config à l'autre.
    """
    lab = np.asarray(labels)
    tot = 0
    covered = 0
    for code, s, e, tc, typ in load_smart_tagging():
        if not ((code == "set pieces" and typ in _RESTART_TYPES) or code == "goals"):
            continue
        s = max(0, min(N_FRAMES, s)); e = max(0, min(N_FRAMES, e))
        if e <= s:
            continue
        tot += 1
        if np.any(lab[s:e] == "D"):
            covered += 1
    return {"restarts": tot, "restart_D_coverage": round(covered / tot, 3) if tot else 0.0}


def score_public(labels):
    """macroF1 / spellF1 sur les 5 minutes publiques. VÉRIFICATION DE PIPELINE, pas de réglage."""
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm
    gt = pm.label_codes(pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")))
    r = pm.evaluate(gt, pm.label_codes(labels))
    return {"macroF1": round(r["macro_f1"], 4), "spellF1": round(r["spell_f1_mean"], 4),
            "possessions_5min": r["n_pred_spells"], "mean_dur_s": round(r["mean_spell_duration_s"], 1)}


def sweep(name, feat, key, values, base=None):
    """Sensibilité d'un paramètre, SANS vérité terrain : on regarde possessions / durée / part de D."""
    rows = []
    for v in values:
        lab = run_method(name, feat, {**(base or {}), key: v})
        rows.append({key: v, **diagnostics(lab)})
    return pd.DataFrame(rows).set_index(key)
