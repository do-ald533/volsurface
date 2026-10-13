from datetime import UTC, datetime, timedelta

from volsurface.analytics.alerts import (
    Alert,
    AlertEngine,
    AlertKind,
    GexSignChange,
    IvJump,
    MarketSummary,
    NearFlip,
)

T0 = datetime(2026, 10, 2, 14, 0, tzinfo=UTC)


def _summary(
    *,
    gex: float = 0.5,
    distance: float = 3.0,
    iv: float = 20.0,
    size: float = 1.0,
    minute: int = 0,
) -> MarketSummary:
    return MarketSummary(
        underlying="SYN",
        asof=T0 + timedelta(minutes=minute),
        spot=100.0,
        gex_at_spot=gex,
        flip=97.0,
        distance_to_flip_pct=distance,
        atm_iv_pct=iv,
        gex_size=size,
    )


def _kinds(alerts: tuple[Alert, ...]) -> list[AlertKind]:
    return [a.kind for a in alerts]


class TestRules:
    def test_sign_change_needs_a_previous_update_and_an_actual_flip(self) -> None:
        rule = GexSignChange()
        assert list(rule.evaluate(None, _summary(gex=-1.0))) == []
        assert list(rule.evaluate(_summary(gex=1.0), _summary(gex=2.0))) == []
        (alert,) = rule.evaluate(_summary(gex=1.0), _summary(gex=-0.5))
        assert alert.kind is AlertKind.GEX_SIGN_CHANGE
        assert "negative" in alert.message

    def test_near_flip_fires_on_entry_only(self) -> None:
        rule = NearFlip(0.5)
        far, near = _summary(distance=2.0), _summary(distance=0.2)
        assert [a.kind for a in rule.evaluate(far, near)] == [AlertKind.NEAR_FLIP]
        assert list(rule.evaluate(near, near)) == []
        assert list(rule.evaluate(near, far)) == []

    def test_near_flip_reports_first_sight_when_already_inside_the_band(self) -> None:
        assert len(list(NearFlip(0.5).evaluate(None, _summary(distance=-0.3)))) == 1

    def test_near_flip_ignores_nan_distance(self) -> None:
        assert list(NearFlip(0.5).evaluate(None, _summary(distance=float("nan")))) == []

    def test_iv_jump_is_symmetric_and_thresholded(self) -> None:
        rule = IvJump(2.0)
        base = _summary(iv=20.0)
        assert list(rule.evaluate(None, base)) == []
        assert list(rule.evaluate(base, _summary(iv=21.9))) == []
        assert len(list(rule.evaluate(base, _summary(iv=22.5)))) == 1
        assert len(list(rule.evaluate(base, _summary(iv=17.5)))) == 1
        assert list(rule.evaluate(base, _summary(iv=float("nan")))) == []


class TestEngine:
    def test_default_engine_combines_all_rules(self) -> None:
        engine = AlertEngine.default()
        before = _summary(gex=1.0, distance=2.0, iv=20.0)
        after = _summary(gex=-1.0, distance=0.1, iv=24.0)
        assert set(_kinds(engine.evaluate(before, after))) == {
            AlertKind.GEX_SIGN_CHANGE,
            AlertKind.NEAR_FLIP,
            AlertKind.IV_JUMP,
        }

    def test_underlyings_with_negligible_gamma_are_skipped(self) -> None:
        engine = AlertEngine.default(min_gex=0.05)
        tiny = _summary(gex=-0.001, size=0.01)
        assert engine.evaluate(_summary(gex=0.001, size=0.01), tiny) == ()
        assert engine.evaluate(None, _summary(size=float("nan"))) == ()

    def test_custom_rule_sets_are_supported(self) -> None:
        engine = AlertEngine([IvJump(1.0)], min_gex=0.0)
        assert _kinds(engine.evaluate(_summary(iv=20.0), _summary(iv=22.0))) == [AlertKind.IV_JUMP]
