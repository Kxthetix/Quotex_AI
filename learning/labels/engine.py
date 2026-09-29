"""Deterministic Label Engine for Phase 4.

Computes forward prediction targets strictly using data occurring after timestamp T.
Provides four specialized label types:
1. Direction Label (Classification: future price direction)
2. Return Label (Regression: future percentage return magnitude)
3. Volatility Label (Regression: future realized volatility)
4. Trade-Quality Label (Binary: target reached before invalidation)
"""

import numpy as np
import pandas as pd

from learning.labels.models import LabelDefinition


class LabelEngine:
    """Computes versioned future targets for statistical and machine-learning models."""

    def __init__(
        self,
        label_version: str = "v1.0.0",
        horizons: list[int] | None = None,
        trade_quality_target_pct: float = 0.005,  # 0.5% profit target
        trade_quality_stop_pct: float = 0.005,  # 0.5% stop loss
    ) -> None:
        self.label_version = label_version
        self.horizons = horizons or [5, 15, 60]
        self.trade_quality_target_pct = trade_quality_target_pct
        self.trade_quality_stop_pct = trade_quality_stop_pct
        self._definitions = self._build_definitions()

    def _build_definitions(self) -> list[LabelDefinition]:
        """Construct immutable metadata catalog of active label definitions."""
        defs: list[LabelDefinition] = []
        for h in self.horizons:
            # 1. Direction
            defs.append(
                LabelDefinition(
                    name=f"direction_{h}",
                    version=self.label_version,
                    target_type="binary_classification",
                    horizon_bars=h,
                    description=f"Binary direction indicator: 1 if close_{{t+{h}}} > close_t, else 0",
                )
            )
            # 2. Return
            defs.append(
                LabelDefinition(
                    name=f"return_{h}",
                    version=self.label_version,
                    target_type="regression",
                    horizon_bars=h,
                    description=f"Forward percentage return: (close_{{t+{h}}} - close_t) / close_t",
                )
            )
            # 3. Volatility
            defs.append(
                LabelDefinition(
                    name=f"volatility_{h}",
                    version=self.label_version,
                    target_type="regression",
                    horizon_bars=h,
                    description=f"Annualized realized standard deviation of 1m log returns over next {h} bars",
                )
            )
            # 4. Trade Quality
            defs.append(
                LabelDefinition(
                    name=f"trade_quality_{h}",
                    version=self.label_version,
                    target_type="binary_classification",
                    horizon_bars=h,
                    parameters={
                        "target_pct": self.trade_quality_target_pct,
                        "stop_pct": self.trade_quality_stop_pct,
                    },
                    description=f"Binary flag: 1 if profit target hit before stop loss within {h} bars, else 0",
                )
            )
        return defs

    def compute_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all configured forward targets from an OHLCV DataFrame.

        Strict Anti-Leakage Guard:
        Targets at row t depend ONLY on rows t+1 .. t+H.
        For rows where t + H >= len(df), target values are NaN.
        """
        if df.empty:
            return pd.DataFrame()

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n_rows = len(df)

        res = pd.DataFrame(index=df.index)

        # 1-minute log returns for future volatility calculation
        log_ret_1m = np.zeros(n_rows)
        log_ret_1m[1:] = np.log(close[1:] / close[:-1])
        annualize_factor = np.sqrt(365 * 1440)

        for h in self.horizons:
            dir_col = np.full(n_rows, np.nan)
            ret_col = np.full(n_rows, np.nan)
            vol_col = np.full(n_rows, np.nan)
            tq_col = np.full(n_rows, np.nan)

            for i in range(n_rows - h):
                c_now = close[i]
                c_future = close[i + h]

                # 1. Forward Return
                f_ret = (c_future - c_now) / c_now
                ret_col[i] = f_ret

                # 2. Forward Direction (1 for UP, 0 for DOWN)
                dir_col[i] = 1.0 if f_ret > 0 else 0.0

                # 3. Forward Realized Volatility over t+1..t+h
                window_log_rets = log_ret_1m[i + 1 : i + h + 1]
                if len(window_log_rets) > 1:
                    vol_col[i] = np.std(window_log_rets, ddof=1) * annualize_factor

                # 4. Trade Quality: check if target hit before stop loss
                target_price = c_now * (1.0 + self.trade_quality_target_pct)
                stop_price = c_now * (1.0 - self.trade_quality_stop_pct)

                future_highs = high[i + 1 : i + h + 1]
                future_lows = low[i + 1 : i + h + 1]

                hit_target = False
                hit_stop = False

                for step in range(h):
                    if future_highs[step] >= target_price:
                        hit_target = True
                        break
                    if future_lows[step] <= stop_price:
                        hit_stop = True
                        break

                tq_col[i] = 1.0 if (hit_target and not hit_stop) else 0.0

            res[f"direction_{h}"] = dir_col
            res[f"return_{h}"] = ret_col
            res[f"volatility_{h}"] = vol_col
            res[f"trade_quality_{h}"] = tq_col

        return res

    def get_definitions(self) -> list[LabelDefinition]:
        """Return catalog of active label definitions."""
        return list(self._definitions)
