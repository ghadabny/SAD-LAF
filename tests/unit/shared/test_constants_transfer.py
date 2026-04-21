# tests/unit/test_constants_transfer.py
"""
Tests unitaires de get_min_transfer() et MIN_TRANSFER_BY_STOP.
"""
from shared.constants import (
    get_min_transfer,
    MIN_TRANSFER_BY_STOP,
    MIN_TRANSFER_DEFAULT,
)


class TestGetMinTransfer:

    def test_strasbourg_8_minutes(self):
        """Grande gare — quais éloignés → 8 min."""
        assert get_min_transfer("87212027") == 8

    def test_selestat_5_minutes(self):
        assert get_min_transfer("87214007") == 5

    def test_colmar_5_minutes(self):
        assert get_min_transfer("87214080") == 5

    def test_mulhouse_6_minutes(self):
        assert get_min_transfer("87182063") == 6

    def test_saverne_5_minutes(self):
        assert get_min_transfer("87213132") == 5

    def test_gare_inconnue_retourne_defaut(self):
        """Code UIC inexistant → valeur par défaut."""
        assert get_min_transfer("00000000") == MIN_TRANSFER_DEFAULT

    def test_defaut_est_4_minutes(self):
        assert MIN_TRANSFER_DEFAULT == 4

    def test_toutes_les_gares_connues_superieur_au_defaut(self):
        """Toutes les gares référencées doivent avoir un temps ≥ MIN_TRANSFER_DEFAULT."""
        for stop_id, min_tr in MIN_TRANSFER_BY_STOP.items():
            assert min_tr >= MIN_TRANSFER_DEFAULT, (
                f"La gare {stop_id} a un temps de correspondance ({min_tr} min) "
                f"inférieur au défaut ({MIN_TRANSFER_DEFAULT} min)."
            )

    def test_get_min_transfer_retourne_int(self):
        assert isinstance(get_min_transfer("87212027"), int)
        assert isinstance(get_min_transfer("00000000"), int)