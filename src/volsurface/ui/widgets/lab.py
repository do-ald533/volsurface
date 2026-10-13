"""Pricing Lab: Black-Scholes and tree pricing, Monte Carlo, and a Newton-Raphson explorer."""

import asyncio
from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from volsurface.app.lab import (
    VANILLA_METRICS,
    HoverInfo,
    McInputs,
    McOutcome,
    NewtonInputs,
    NewtonTrace,
    VanillaCurves,
    VanillaInputs,
    VanillaQuote,
    price_vanilla,
    run_monte_carlo,
    trace_newton,
    vanilla_curves,
)
from volsurface.pricing.backends import PricingBackend
from volsurface.pricing.montecarlo import PayoffKind
from volsurface.ui import theme
from volsurface.ui.widgets.hover import HoverProbe


def _spin(
    low: float, high: float, value: float, *, decimals: int = 2, step: float = 1.0, suffix: str = ""
) -> QtWidgets.QDoubleSpinBox:
    box = QtWidgets.QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    return box


def _scrolling(form: QtWidgets.QFormLayout, width: int = 330) -> QtWidgets.QScrollArea:
    """Put a long form in a scroll area so it does not force the dock to be tall."""
    holder = QtWidgets.QWidget()
    holder.setLayout(form)
    area = QtWidgets.QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(holder)
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setFixedWidth(width)
    return area


def _plot(x_label: str, y_label: str) -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.setLabel("bottom", x_label)
    plot.setLabel("left", y_label)
    theme.style_plot(plot)
    return plot


_SERIES_STYLES = (
    (theme.ACCENT, QtCore.Qt.PenStyle.SolidLine),
    (theme.ACCENT_2, QtCore.Qt.PenStyle.SolidLine),
    (theme.MUTED, QtCore.Qt.PenStyle.DashLine),
)


class VanillaTab(QtWidgets.QWidget):
    """European value and Greeks next to the American tree value, with a chart over spot."""

    def __init__(self, backend: PricingBackend, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._backend = backend
        self.spot = _spin(0.01, 1e6, 100.0)
        self.strike = _spin(0.01, 1e6, 100.0)
        self.days = _spin(0.1, 3650.0, 90.0, decimals=1, suffix=" d")
        self.rate = _spin(-5.0, 50.0, 4.0, suffix=" %")
        self.dividend = _spin(0.0, 50.0, 1.2, suffix=" %")
        self.vol = _spin(0.1, 500.0, 20.0, suffix=" %")
        self.right = QtWidgets.QComboBox()
        self.right.addItems(["Put", "Call"])
        self.steps = QtWidgets.QSpinBox()
        self.steps.setRange(3, 2000)
        self.steps.setValue(200)
        form = QtWidgets.QFormLayout()
        for label, widget in (
            ("Spot", self.spot),
            ("Strike", self.strike),
            ("Time to expiry", self.days),
            ("Rate", self.rate),
            ("Dividend yield", self.dividend),
            ("Volatility", self.vol),
            ("Type", self.right),
            ("Tree steps", self.steps),
        ):
            form.addRow(label, widget)
        self._output = QtWidgets.QLabel()
        self._output.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._output.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self._output.setMinimumWidth(230)
        self.chart_metric = QtWidgets.QComboBox()
        self.chart_metric.addItems(VANILLA_METRICS)
        self._legend = QtWidgets.QLabel()
        self._legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._plot = _plot("spot", "option value")
        self._series = [self._plot.plot() for _ in _SERIES_STYLES]
        for curve, (colour, style) in zip(self._series, _SERIES_STYLES, strict=True):
            curve.setPen(pg.mkPen(colour, width=2, style=style))
        self._spot_line = self._reference_line(theme.TEXT, "spot", 0.93)
        self._strike_line = self._reference_line(theme.MUTED, "strike", 0.80)
        self._curve_data: VanillaCurves | None = None
        self._probe = HoverProbe(self._plot, self._describe)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Chart"))
        header.addWidget(self.chart_metric)
        header.addWidget(self._legend, 1)
        chart = QtWidgets.QVBoxLayout()
        chart.addLayout(header)
        chart.addWidget(self._plot, 1)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(_scrolling(form))
        layout.addLayout(chart, 1)
        layout.addWidget(self._output)
        for widget in (self.spot, self.strike, self.days, self.rate, self.dividend, self.vol):
            widget.valueChanged.connect(self.recompute)
        self.right.currentIndexChanged.connect(self.recompute)
        self.steps.valueChanged.connect(self.recompute)
        self.chart_metric.currentIndexChanged.connect(self._draw_curves)
        self.recompute()

    def _reference_line(self, colour: str, label: str, position: float) -> pg.InfiniteLine:
        line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(colour, width=1, style=QtCore.Qt.PenStyle.DotLine),
            label=label,
            labelOpts={"position": position, "color": colour, "movable": False},
        )
        self._plot.addItem(line)
        return line

    def _describe(self, spot: float) -> HoverInfo | None:
        if self._curve_data is None:
            return None
        return self._curve_data.hover(spot, self.chart_metric.currentText())

    def _draw_curves(self, *_: object) -> None:
        self._probe.hide()
        data = self._curve_data
        if data is None:
            for curve in self._series:
                curve.setData([], [])
            self._legend.setText("")
            return
        metric = self.chart_metric.currentText()
        shown = data.series(metric)
        for index, curve in enumerate(self._series):
            if index < len(shown):
                curve.setData(data.spots, shown[index][1])
            else:
                curve.setData([], [])
        self._plot.setLabel("left", "option value" if metric == "Value" else metric.lower())
        self._spot_line.setValue(data.inputs.spot)
        self._strike_line.setValue(data.inputs.strike)
        self._legend.setText(
            "&nbsp;&nbsp;".join(
                f"<span style='color:{colour}'>&#9644;</span> {name}"
                for (name, _), (colour, _) in zip(shown, _SERIES_STYLES, strict=False)
            )
        )

    def inputs(self) -> VanillaInputs:
        return VanillaInputs(
            spot=self.spot.value(),
            strike=self.strike.value(),
            days=self.days.value(),
            rate=self.rate.value() / 100,
            dividend_yield=self.dividend.value() / 100,
            vol=self.vol.value() / 100,
            is_call=self.right.currentText() == "Call",
            tree_steps=self.steps.value(),
        )

    def recompute(self, *_: object) -> VanillaQuote | None:
        try:
            inputs = self.inputs()
            quote = price_vanilla(inputs, self._backend)
            self._curve_data = vanilla_curves(inputs, self._backend)
        except Exception as exc:
            self._output.setText(f"<span style='color:{theme.NEGATIVE}'>{exc}</span>")
            self._curve_data = None
            self._draw_curves()
            return None
        self._draw_curves()
        rows = [
            ("Black-Scholes value", f"{quote.bs_price:.4f}"),
            ("American tree value", f"{quote.american_price:.4f}"),
            ("Early-exercise premium", f"{quote.early_exercise_premium:+.4f}"),
            ("Delta", f"{quote.delta:+.4f}"),
            ("Gamma", f"{quote.gamma:.5f}"),
            ("Gamma (tree)", f"{quote.american_gamma:.5f}"),
            ("Vega (per 1.00 vol)", f"{quote.vega:.4f}"),
            ("Theta (per year)", f"{quote.theta:+.4f}"),
            ("Rho", f"{quote.rho:+.4f}"),
        ]
        body = "".join(
            f"<tr><td style='color:{theme.MUTED}'>{k}</td>"
            f"<td align='right'>&nbsp;&nbsp;{v}</td></tr>"
            for k, v in rows
        )
        self._output.setText(f"<table style='font-size:13px'>{body}</table>")
        return quote


class MonteCarloTab(QtWidgets.QWidget):
    """Monte Carlo price with its standard error and convergence history."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = QtWidgets.QComboBox()
        self.model.addItems(["gbm", "heston", "rainbow"])
        self.payoff = QtWidgets.QComboBox()
        for kind in PayoffKind:
            self.payoff.addItem(kind.name.replace("_", " ").title(), kind)
        self.spot = _spin(0.01, 1e6, 100.0)
        self.strike = _spin(0.01, 1e6, 100.0)
        self.days = _spin(1.0, 3650.0, 365.0, decimals=0, suffix=" d")
        self.rate = _spin(-5.0, 50.0, 4.0, suffix=" %")
        self.vol = _spin(0.1, 500.0, 20.0, suffix=" %")
        self.v0 = _spin(0.0, 4.0, 0.04, decimals=3, step=0.01)
        self.kappa = _spin(0.0, 20.0, 1.5)
        self.theta = _spin(0.0, 4.0, 0.04, decimals=3, step=0.01)
        self.xi = _spin(0.0, 5.0, 0.3)
        self.rho = _spin(-1.0, 1.0, -0.7, step=0.1)
        self.rho_assets = _spin(-1.0, 1.0, 0.0, step=0.1)
        self.paths = QtWidgets.QSpinBox()
        self.paths.setRange(100, 20_000_000)
        self.paths.setSingleStep(50_000)
        self.paths.setValue(200_000)
        self.antithetic = QtWidgets.QCheckBox("antithetic variates")
        self.antithetic.setChecked(True)
        self.run_button = QtWidgets.QPushButton("Run")
        form = QtWidgets.QFormLayout()
        for label, widget in (
            ("Model", self.model),
            ("Payoff", self.payoff),
            ("Spot", self.spot),
            ("Strike", self.strike),
            ("Expiry", self.days),
            ("Rate", self.rate),
            ("Vol (GBM)", self.vol),
            ("v0", self.v0),
            ("kappa", self.kappa),
            ("theta", self.theta),
            ("xi", self.xi),
            ("rho", self.rho),
            ("rho assets", self.rho_assets),
            ("Paths", self.paths),
        ):
            form.addRow(label, widget)
        form.addRow(self.antithetic)
        form.addRow(self.run_button)
        self._summary = QtWidgets.QLabel("press Run")
        self._summary.setObjectName("muted")
        self._plot = _plot("paths", "price estimate")
        self._band = pg.FillBetweenItem(
            self._plot.plot(), self._plot.plot(), brush=pg.mkBrush(0, 216, 230, 50)
        )
        self._plot.addItem(self._band)
        self._line = self._plot.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self._reference = pg.InfiniteLine(
            angle=0, pen=pg.mkPen(theme.ACCENT_2, style=QtCore.Qt.PenStyle.DashLine)
        )
        self._reference.setVisible(False)
        self._plot.addItem(self._reference)
        self._outcome: McOutcome | None = None
        self._probe = HoverProbe(self._plot, self._describe)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self._plot, 1)
        right.addWidget(self._summary)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(_scrolling(form))
        layout.addLayout(right, 1)
        self.run_button.clicked.connect(self.start)
        self._task: asyncio.Future[McOutcome] | None = None

    def inputs(self) -> McInputs:
        return McInputs(
            model=self.model.currentText(),  # type: ignore[arg-type]
            payoff=self.payoff.currentData(),
            spot=self.spot.value(),
            strike=self.strike.value(),
            days=self.days.value(),
            rate=self.rate.value() / 100,
            vol=self.vol.value() / 100,
            v0=self.v0.value(),
            kappa=self.kappa.value(),
            theta=self.theta.value(),
            xi=self.xi.value(),
            rho=self.rho.value(),
            rho_assets=self.rho_assets.value(),
            n_paths=self.paths.value(),
            antithetic=self.antithetic.isChecked(),
        )

    def start(self) -> None:
        """Run on a worker thread; needs the asyncio loop that ``qasync`` provides."""
        self.run_button.setEnabled(False)
        self._summary.setText("simulating ...")
        inputs = self.inputs()
        task = asyncio.ensure_future(asyncio.to_thread(run_monte_carlo, inputs))
        task.add_done_callback(self._finished)
        self._task = task

    def _describe(self, paths: float) -> HoverInfo | None:
        return None if self._outcome is None else self._outcome.hover(paths)

    def run_blocking(self) -> McOutcome:
        """Synchronous variant (tests, scripts)."""
        outcome = run_monte_carlo(self.inputs())
        self.show_outcome(outcome)
        return outcome

    def _finished(self, task: asyncio.Future[McOutcome]) -> None:
        self.run_button.setEnabled(True)
        try:
            self.show_outcome(task.result())
        except Exception as exc:
            self._summary.setText(f"failed: {exc}")

    def show_outcome(self, outcome: McOutcome) -> None:
        self._probe.hide()
        self._outcome = outcome
        result = outcome.result
        counts, price, lower, upper = outcome.convergence()
        self._line.setData(counts, price)
        self._band.setCurves(self._plot.plot(counts, upper), self._plot.plot(counts, lower))
        pad = 0.15 * float(np.max(upper) - np.min(lower) or 1.0)
        self._plot.setYRange(float(np.min(lower)) - pad, float(np.max(upper)) + pad)
        self._plot.setXRange(float(counts[0]), float(counts[-1]), padding=0.02)
        low, high = result.ci95
        text = (
            f"{outcome.model_name}: {result.price:.4f} +/- {result.std_error:.4f}"
            f"  (95% CI {low:.4f} to {high:.4f})"
        )
        if outcome.reference is not None:
            self._reference.setValue(outcome.reference)
            self._reference.setVisible(True)
            deviation = outcome.deviation_in_se
            text += f"   Black-Scholes {outcome.reference:.4f}"
            if deviation is not None:
                text += f" ({deviation:+.2f} s.e.)"
        else:
            self._reference.setVisible(False)
        if outcome.feller_ok is False:
            text += "   Feller condition violated"
        self._summary.setText(text)


class NewtonTab(QtWidgets.QWidget):
    """Newton-Raphson implied volatility, one iteration at a time."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.spot = _spin(0.01, 1e6, 100.0)
        self.strike = _spin(0.01, 1e6, 105.0)
        self.days = _spin(1.0, 3650.0, 90.0, decimals=0, suffix=" d")
        self.market_vol = _spin(1.0, 300.0, 30.0, suffix=" %")
        self.start_vol = _spin(0.5, 300.0, 10.0, suffix=" %")
        self.right = QtWidgets.QComboBox()
        self.right.addItems(["Call", "Put"])
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self._message = QtWidgets.QLabel("")
        self._message.setObjectName("muted")
        self._message.setWordWrap(True)
        form = QtWidgets.QFormLayout()
        for label, widget in (
            ("Spot", self.spot),
            ("Strike", self.strike),
            ("Expiry", self.days),
            ("Market vol (sets the price)", self.market_vol),
            ("Starting guess", self.start_vol),
            ("Type", self.right),
            ("Iteration", self.slider),
        ):
            form.addRow(label, widget)
        form.addRow(self._message)
        self._plot = _plot("volatility %", "option price")
        self._curve = self._plot.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self._market = pg.InfiniteLine(
            angle=0, pen=pg.mkPen(theme.ACCENT_2, style=QtCore.Qt.PenStyle.DashLine)
        )
        self._plot.addItem(self._market)
        self._tangents = self._plot.plot(pen=pg.mkPen(theme.NEGATIVE, width=1.5), connect="pairs")
        self._points = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(theme.NEGATIVE), pen=None)
        self._plot.addItem(self._points)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(_scrolling(form))
        layout.addWidget(self._plot, 1)
        for widget in (self.spot, self.strike, self.days, self.market_vol, self.start_vol):
            widget.valueChanged.connect(self.recompute)
        self.right.currentIndexChanged.connect(self.recompute)
        self.slider.valueChanged.connect(self._draw)
        self._trace: NewtonTrace | None = None
        self._probe = HoverProbe(self._plot, self._describe)
        self.recompute()

    def _describe(self, vol_pct: float) -> HoverInfo | None:
        if self._trace is None:
            return None
        return self._trace.hover(vol_pct, upto=self.slider.value())

    def inputs(self) -> NewtonInputs:
        return NewtonInputs(
            spot=self.spot.value(),
            strike=self.strike.value(),
            days=self.days.value(),
            market_vol=self.market_vol.value() / 100,
            start_vol=self.start_vol.value() / 100,
            is_call=self.right.currentText() == "Call",
        )

    def recompute(self, *_: object) -> NewtonTrace:
        self._probe.hide()
        trace = trace_newton(self.inputs())
        self._trace = trace
        self._curve.setData(100 * trace.vols, trace.prices)
        self._market.setValue(trace.market_price)
        self._message.setText(trace.message)
        last = max(len(trace.steps) - 1, 0)
        self.slider.blockSignals(True)
        self.slider.setRange(0, last)
        self.slider.setValue(last)
        self.slider.blockSignals(False)
        self._draw(last)
        return trace

    def _draw(self, upto: int) -> None:
        self._probe.hide()
        trace = self._trace
        if trace is None or not trace.steps:
            self._tangents.setData([], [])
            self._points.setData([], [])
            return
        steps = trace.steps[: upto + 1]
        xs: list[float] = []
        ys: list[float] = []
        for i, step in enumerate(steps):
            nxt = trace.steps[i + 1].sigma if i + 1 < len(trace.steps) else step.sigma
            xs += [100 * step.sigma, 100 * step.sigma, 100 * step.sigma, 100 * nxt]
            ys += [0.0, step.model_price, step.model_price, trace.market_price]
        self._tangents.setData(np.array(xs), np.array(ys))
        self._points.setData([100 * s.sigma for s in steps], [s.model_price for s in steps])


class PricingLabWidget(QtWidgets.QTabWidget):
    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(900, 300)

    def __init__(
        self, backend_factory: Callable[[], PricingBackend], parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.vanilla = VanillaTab(backend_factory())
        self.monte_carlo = MonteCarloTab()
        self.newton = NewtonTab()
        self.addTab(self.vanilla, "Black-Scholes / American")
        self.addTab(self.monte_carlo, "Monte Carlo")
        self.addTab(self.newton, "Newton-Raphson IV")
