"""The option chain as a sortable, filterable table."""

import math

from PySide6 import QtCore, QtGui, QtWidgets

from volsurface.app.viewmodels import CHAIN_COLUMNS, ChainTable
from volsurface.ui import theme

SORT_ROLE = QtCore.Qt.ItemDataRole.UserRole
_NUMERIC_COLUMNS = frozenset({1, 3, 4, 5, 6, 7, 8, 9})
_DECIMALS = {1: 2, 3: 2, 4: 2, 5: 3, 6: 0, 7: 2, 8: 2, 9: 2}
_DIFF_COLUMN = 9
_EXPIRY_COLUMN = 0
_USED_COLUMN = 10
_BIG_DIFF = 1.0


def format_cell(column: int, value: str | float) -> str:
    """Text shown for a cell; missing numbers print as a dash."""
    if isinstance(value, str):
        return value
    if math.isnan(value):
        return "-"
    decimals = _DECIMALS.get(column, 2)
    text = f"{value:,.{decimals}f}"
    return f"{value:+.2f}" if column == _DIFF_COLUMN else text


class ChainTableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._table: ChainTable | None = None

    def set_table(self, table: ChainTable) -> None:
        """Replace the data; keeps scroll position and selection when the size is unchanged."""
        if self._table is not None and self._table.rows == table.rows:
            self._table = table
            if table.rows:
                self.dataChanged.emit(
                    self.index(0, 0), self.index(table.rows - 1, len(CHAIN_COLUMNS) - 1)
                )
            return
        self.beginResetModel()
        self._table = table
        self.endResetModel()

    def rowCount(
        self,
        parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),  # noqa: B008
    ) -> int:
        if parent.isValid():
            return 0
        return self._table.rows if self._table else 0

    def columnCount(
        self,
        parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),  # noqa: B008
    ) -> int:
        return 0 if parent.isValid() else len(CHAIN_COLUMNS)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> object:
        if (
            role == QtCore.Qt.ItemDataRole.DisplayRole
            and orientation == QtCore.Qt.Orientation.Horizontal
        ):
            return CHAIN_COLUMNS[section]
        return None

    def data(  # noqa: PLR0911
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> object:
        if self._table is None or not index.isValid():
            return None
        row, column = index.row(), index.column()
        value = self._table.value(row, column)
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return format_cell(column, value)
        if role == SORT_ROLE:
            return value
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            right = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
            return int(right) if column in _NUMERIC_COLUMNS else None
        if role == QtCore.Qt.ItemDataRole.ForegroundRole:
            if not self._table.update.usable[row]:
                return QtGui.QBrush(QtGui.QColor(theme.MUTED))
            if (
                column == _DIFF_COLUMN
                and isinstance(value, float)
                and not math.isnan(value)
                and abs(value) > _BIG_DIFF
            ):
                return QtGui.QBrush(QtGui.QColor(theme.WARNING))
        return None


class ChainFilter(QtCore.QSortFilterProxyModel):
    """Filters by expiry and by whether the quote was used in the fit."""

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._expiry: str | None = None
        self._used_only = False
        self.setSortRole(SORT_ROLE)
        self.setDynamicSortFilter(True)

    def set_expiry(self, expiry: str | None) -> None:
        self.beginFilterChange()
        self._expiry = expiry
        self.endFilterChange(QtCore.QSortFilterProxyModel.Direction.Rows)

    def set_used_only(self, enabled: bool) -> None:
        self.beginFilterChange()
        self._used_only = enabled
        self.endFilterChange(QtCore.QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(
        self, source_row: int, source_parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> bool:
        model = self.sourceModel()
        wrong_expiry = (
            self._expiry is not None
            and model.index(source_row, _EXPIRY_COLUMN, source_parent).data() != self._expiry
        )
        if wrong_expiry:
            return False
        if self._used_only:
            return bool(model.index(source_row, _USED_COLUMN, source_parent).data() == "yes")
        return True


class ChainTableWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = ChainTableModel(self)
        self.proxy = ChainFilter(self)
        self.proxy.setSourceModel(self.model)
        self.view = QtWidgets.QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.setAlternatingRowColors(True)
        self.view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(20)
        header = self.view.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self._expiry = QtWidgets.QComboBox()
        self._expiry.addItem("all expiries", None)
        self._used = QtWidgets.QCheckBox("used in fit only")
        self._used.setChecked(True)
        self._count = QtWidgets.QLabel("")
        self._count.setObjectName("muted")
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self._expiry)
        bar.addWidget(self._used)
        bar.addStretch(1)
        bar.addWidget(self._count)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(bar)
        layout.addWidget(self.view, 1)
        self._expiry.currentIndexChanged.connect(self._on_filter)
        self._used.toggled.connect(self._on_filter)
        self._known: list[str] = []
        self.proxy.set_used_only(True)

    def set_table(self, table: ChainTable) -> None:
        labels = [str(e) for e in table.expiries()]
        if labels != self._known:
            current = self._expiry.currentData()
            self._known = labels
            self._expiry.blockSignals(True)
            self._expiry.clear()
            self._expiry.addItem("all expiries", None)
            for label in labels:
                self._expiry.addItem(label, label)
            at = self._expiry.findData(current)
            self._expiry.setCurrentIndex(max(at, 0))
            self._expiry.blockSignals(False)
            self.proxy.set_expiry(self._expiry.currentData())
        self.model.set_table(table)
        self._count.setText(f"{self.proxy.rowCount()} of {table.rows} options")

    def _on_filter(self, *_: object) -> None:
        self.proxy.set_expiry(self._expiry.currentData())
        self.proxy.set_used_only(self._used.isChecked())
        self._count.setText(f"{self.proxy.rowCount()} of {self.model.rowCount()} options")

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(900, 230)
