# services/ml_engine/data/laf/preprocessor.py
import pandas as pd


class LAFPreprocessor:
    """
    Responsabilité unique : nettoyer les données LAF brutes et construire
    les features historiques nécessaires au modèle LightGBM.

    Deux opérations principales :
        1. Nettoyage des DataFrames bruts (CC, PV)
           - Suppression des lignes avec données manquantes critiques
           - Parsing des colonnes datetime
           - Construction du troncon_id (clé de jointure avec le GTFS)

        2. Agrégation par tronçon (build_troncon_stats)
           - nb_controles, nb_irregularites, taux_irregularite
           - Utilisées comme features par HistoricalFeatureTransformer

    Définition du troncon_id :
        Format : "{origin_uic}_{dest_uic}_{dep_hour}"
        Exemple : "87212027_87214007_8" = Strasbourg→Sélestat entre 8h et 9h

        Pourquoi inclure l'heure ?
            Le taux de fraude varie fortement selon le créneau horaire.
            Un train Strasbourg→Sélestat à 8h du matin (pendulaires)
            a un profil très différent du même tronçon à 14h.
            L'heure est déjà présente dans les données GTFS (dep_hour)
            et dans les données LAF (verifiedTickets_verificationDateTime).

    Usage :
        preprocessor = LAFPreprocessor()
        cc_clean = preprocessor.clean_cc(loader.load_cc())
        stats    = preprocessor.build_troncon_stats(cc_clean)
    """

    # Colonnes datetime à parser dans CC/SCAN
    _CC_DATETIME_COLS = [
        "verifiedTickets_verificationDateTime",
        "ticket_travelInformation_departureDateTime",
    ]

    # Colonnes UIC critiques dans CC — ligne supprimée si l'une est absente
    _CC_UIC_REQUIRED = [
        "ticket_travelInformation_origin_uicCode",
        "ticket_travelInformation_destination_uicCode",
        "ticket_travelInformation_departureDateTime",
    ]

    # Statuts d'irrégularité dans verifiedTickets_verificationStatus
    # À affiner avec l'équipe LAF quand les données réelles arrivent
    STATUTS_IRREGULIERS = {"IRREGULAR", "FRAUD", "EVADER"}

    def clean_cc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie le DataFrame CC/SCAN brut.

        Opérations :
            1. Copie défensive (ne modifie jamais l'entrée)
            2. Suppression des lignes avec UIC ou datetime manquants
            3. Parsing des colonnes datetime
            4. Construction du troncon_id

        Paramètres :
            df : DataFrame brut produit par LAFLoader.load_cc() ou load_scan()

        Retourne un DataFrame nettoyé avec la colonne troncon_id ajoutée.
        """
        result = df.copy()
        avant = len(result)

        # Supprimer les lignes avec données critiques manquantes
        result = result.dropna(subset=self._CC_UIC_REQUIRED)
        supprimees = avant - len(result)
        if supprimees > 0:
            print(
                f"[LAFPreprocessor] {supprimees:,} lignes CC supprimées "
                f"(UIC ou datetime manquant)."
            )

        # Parser les colonnes datetime
        for col in self._CC_DATETIME_COLS:
            if col in result.columns:
                result[col] = pd.to_datetime(result[col], errors="coerce")

        # Supprimer les lignes dont le parsing datetime a échoué
        result = result.dropna(
            subset=["ticket_travelInformation_departureDateTime"]
        )

        # Construire le troncon_id
        result["troncon_id"] = self._build_troncon_id(
            origin_uic=result["ticket_travelInformation_origin_uicCode"],
            dest_uic=result["ticket_travelInformation_destination_uicCode"],
            dep_datetime=result["ticket_travelInformation_departureDateTime"],
        )

        return result.reset_index(drop=True)

    def clean_pv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie le DataFrame PV brut.

        Le PV n'a pas de departure_time au niveau du titre de transport —
        on utilise l'heure d'émission du PV (penalties_issueDateTime)
        comme proxy pour le créneau horaire.

        Retourne un DataFrame nettoyé avec troncon_id ajouté.
        """
        result = df.copy()

        # Parser issueDateTime
        result["penalties_issueDateTime"] = pd.to_datetime(
            result["penalties_issueDateTime"], errors="coerce"
        )

        # Supprimer les lignes sans UIC ou sans datetime
        result = result.dropna(subset=[
            "penalties_origin_uicCode",
            "penalties_destination_uicCode",
            "penalties_issueDateTime",
        ])

        # troncon_id basé sur l'heure d'émission du PV
        result["troncon_id"] = self._build_troncon_id(
            origin_uic=result["penalties_origin_uicCode"],
            dest_uic=result["penalties_destination_uicCode"],
            dep_datetime=result["penalties_issueDateTime"],
        )

        return result.reset_index(drop=True)

    def build_troncon_stats(self, cc_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Agrège les données CC nettoyées par troncon_id.

        Produit les features historiques qui alimenteront
        HistoricalFeatureTransformer :
            - nb_controles         : nombre total de vérifications
            - nb_irregularites     : nombre d'irrégularités constatées
            - taux_irregularite    : nb_irregularites / nb_controles
            - derniere_date_controle : date du dernier contrôle sur ce tronçon

        Paramètres :
            cc_clean : DataFrame produit par clean_cc()

        Retourne une ligne par troncon_id.
        """
        # Créer un indicateur booléen d'irrégularité
        is_irreg = cc_clean["verifiedTickets_verificationStatus"].isin(
            self.STATUTS_IRREGULIERS
        )

        stats = (
            cc_clean
            .assign(is_irregulier=is_irreg)
            .groupby("troncon_id", as_index=False)
            .agg(
                nb_controles=(
                    "verifiedTickets_verificationStatus", "count"
                ),
                nb_irregularites=("is_irregulier", "sum"),
                derniere_date_controle=(
                    "verifiedTickets_verificationDateTime", "max"
                ),
            )
        )

        stats["taux_irregularite"] = (
            stats["nb_irregularites"] / stats["nb_controles"]
        ).round(4)

        stats["nb_irregularites"] = stats["nb_irregularites"].astype(int)

        print(
            f"[LAFPreprocessor] Stats construites : "
            f"{len(stats):,} tronçons uniques."
        )
        return stats

    # ── Méthode privée ────────────────────────────────────────────────────────

    @staticmethod
    def _build_troncon_id(
        origin_uic: pd.Series,
        dest_uic: pd.Series,
        dep_datetime: pd.Series,
    ) -> pd.Series:
        """
        Construit le troncon_id vectorisé : "{origin}_{dest}_{hour}".

        Vectorisé avec .str et .dt pour performance sur millions de lignes.
        dep_datetime doit déjà être de type datetime64 (après pd.to_datetime).
        """
        hour = dep_datetime.dt.hour.astype(str)
        return origin_uic.str.strip() + "_" + dest_uic.str.strip() + "_" + hour