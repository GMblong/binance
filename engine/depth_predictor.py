"""
Predictive Orderflow ML
========================
Lightweight model that predicts whether orderbook walls are REAL or FAKE (spoofing).
Trains on depth_history + aggTrade patterns → outputs probability wall will hold.
"""
import numpy as np
import time
from collections import deque
from utils.state import market_data
from utils.logger import log_error


class DepthPredictor:
    def __init__(self):
        # Online learning: store labeled examples as they resolve
        self.examples = deque(maxlen=2000)  # (features, label)
        self.model = None  # Lazy-loaded sklearn SGDClassifier
        self.last_train = 0
        self.predictions_cache = {}  # {symbol: (ts, prediction)}
        # Track walls to label them later
        self.pending_walls = {}  # {symbol: {ts, side, qty, price}}

    def _extract_features(self, symbol: str):
        """Extract features from current depth + trade data."""
        buf = market_data.depth_history.get(symbol)
        agg = market_data.cvd_buf.get(symbol)
        if not buf or len(buf) < 5:
            return None

        now = time.time()
        recent = [(ts, bt, at, tbq, taq) for ts, bt, at, tbq, taq in buf if now - ts < 15]
        if len(recent) < 4:
            return None

        # Feature 1-2: Bid/Ask total change rate
        t_diff = recent[-1][0] - recent[0][0]
        if t_diff == 0:
            return None
        bid_vel = (recent[-1][1] - recent[0][1]) / t_diff
        ask_vel = (recent[-1][2] - recent[0][2]) / t_diff

        # Feature 3-4: Top-of-book stability (low std = real, high std = spoof)
        top_bids = [d[3] for d in recent]
        top_asks = [d[4] for d in recent]
        bid_stability = np.std(top_bids) / (np.mean(top_bids) + 1e-8)
        ask_stability = np.std(top_asks) / (np.mean(top_asks) + 1e-8)

        # Feature 5: Bid/Ask ratio
        ratio = recent[-1][1] / (recent[-1][2] + 1e-8)

        # Feature 6-7: Aggressive trade flow alignment
        buy_vol, sell_vol = 0.0, 0.0
        if agg:
            cutoff = now - 10
            for ts, delta in reversed(agg):
                if ts < cutoff:
                    break
                if delta > 0:
                    buy_vol += delta
                else:
                    sell_vol += abs(delta)

        flow_ratio = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-8)

        # Feature 8: Size anomaly (top qty vs average)
        avg_bid = np.mean(top_bids)
        avg_ask = np.mean(top_asks)
        size_anomaly_bid = top_bids[-1] / (avg_bid + 1e-8)
        size_anomaly_ask = top_asks[-1] / (avg_ask + 1e-8)

        return np.array([
            bid_vel, ask_vel, bid_stability, ask_stability,
            ratio, flow_ratio, size_anomaly_bid, size_anomaly_ask
        ])

    def predict(self, symbol: str):
        """Predict if current orderbook walls are real.
        
        Returns: (bid_real_prob, ask_real_prob) in [0, 1]
        1.0 = definitely real wall, 0.0 = definitely fake/spoof
        """
        # Cache for 2 seconds
        cached = self.predictions_cache.get(symbol)
        if cached and time.time() - cached[0] < 2:
            return cached[1]

        features = self._extract_features(symbol)
        if features is None:
            return (0.5, 0.5)

        if self.model is None or len(self.examples) < 50:
            # Heuristic fallback before model is trained
            bid_stability = features[2]
            ask_stability = features[3]
            flow_ratio = features[5]
            # Low stability = likely spoof
            bid_real = max(0.1, min(0.9, 1.0 - bid_stability * 2))
            ask_real = max(0.1, min(0.9, 1.0 - ask_stability * 2))
            # Flow alignment boosts confidence
            if flow_ratio > 0.3:
                bid_real = min(0.95, bid_real + 0.2)
            elif flow_ratio < -0.3:
                ask_real = min(0.95, ask_real + 0.2)
            result = (bid_real, ask_real)
        else:
            try:
                X = features.reshape(1, -1)
                prob = self.model.predict_proba(X)[0][1]  # P(real)
                # Separate bid/ask based on flow direction
                flow = features[5]
                bid_real = prob + flow * 0.1
                ask_real = prob - flow * 0.1
                result = (max(0, min(1, bid_real)), max(0, min(1, ask_real)))
            except Exception:
                result = (0.5, 0.5)

        self.predictions_cache[symbol] = (time.time(), result)
        return result

    def label_wall(self, symbol: str, side: str, held: bool):
        """Called when we observe a wall resolved (held or pulled).
        
        side: 'bid' or 'ask'
        held: True if wall held (real), False if pulled (fake)
        """
        features = self._extract_features(symbol)
        if features is not None:
            self.examples.append((features, 1 if held else 0))
            # Retrain periodically
            if time.time() - self.last_train > 300 and len(self.examples) >= 50:
                self._train()

    def observe_and_label(self, symbol: str):
        """Auto-label walls by tracking if large orders disappear without fills."""
        buf = market_data.depth_history.get(symbol)
        if not buf or len(buf) < 10:
            return

        now = time.time()
        # Check if we have a pending wall to resolve
        pending = self.pending_walls.get(symbol)
        if pending and now - pending['ts'] > 5:
            # Check if wall is still there
            latest = buf[-1]
            if pending['side'] == 'bid':
                current_qty = latest[3]
                # Wall disappeared without price moving through it = FAKE
                price_now = market_data.prices.get(symbol, 0)
                if current_qty < pending['qty'] * 0.3 and price_now >= pending['price'] * 0.999:
                    self.label_wall(symbol, 'bid', held=False)
                elif current_qty >= pending['qty'] * 0.5:
                    self.label_wall(symbol, 'bid', held=True)
            del self.pending_walls[symbol]

        # Detect new large walls
        if buf and len(buf) >= 2:
            latest = buf[-1]
            prev = buf[-2]
            # Sudden large bid appearance
            if latest[3] > prev[3] * 3 and latest[3] > 0:
                self.pending_walls[symbol] = {
                    'ts': now, 'side': 'bid',
                    'qty': latest[3], 'price': market_data.prices.get(symbol, 0)
                }

    def _train(self):
        """Incremental train with SGDClassifier."""
        try:
            from sklearn.linear_model import SGDClassifier
            X = np.array([e[0] for e in self.examples])
            y = np.array([e[1] for e in self.examples])
            if len(np.unique(y)) < 2:
                return
            self.model = SGDClassifier(loss='log_loss', max_iter=100, random_state=42)
            self.model.fit(X, y)
            self.last_train = time.time()
        except Exception as e:
            log_error(f"DepthPredictor train error: {str(e)}")


depth_predictor = DepthPredictor()
