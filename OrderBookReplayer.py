import struct
from collections import defaultdict
import pandas as pd

# ITCH message type costanti
ADD_ORDER       = b'A'
ADD_ORDER_MPID  = b'F'
ORDER_EXECUTED  = b'E'
ORDER_CANCELLED = b'C'
ORDER_DELETE    = b'D'
ORDER_REPLACE   = b'U'


class OrderBookReplayer:
    """Rebuild an order book for a single symbol and esporta snapshot."""
    def __init__(self, symbol: str, depth: int = 10):
        self.symbol   = symbol.encode()
        self.depth    = depth
        self.bids     = defaultdict(dict)   # {price: {order_id: size}}
        self.asks     = defaultdict(dict)
        self.order_map = {}                 # {order_id: (price, size, side)}
        self.snapshots = []

    # ---------- PARSING ----------
    def parse_itch(self, path: str):
        """Streamma il file ITCH e aggiorna l'order book."""
        with open(path, "rb") as f:
            while header := f.read(2):
                length  = struct.unpack(">H", header)[0]
                payload = f.read(length)
                if not payload:
                    break
                self._route(payload)

    def _route(self, msg: bytes):
        t = msg[0:1]
        if t in {ADD_ORDER, ADD_ORDER_MPID}:
            self._add(msg)
        elif t == ORDER_CANCELLED:
            self._cancel(msg)
        elif t == ORDER_EXECUTED:
            self._cancel(msg)      # gestito come cancel
        elif t == ORDER_DELETE:
            self._delete(msg)
        elif t == ORDER_REPLACE:
            self._replace(msg)

    # ---------- HANDLER ----------
    def _add(self, msg: bytes):
        fmt = ">QcI8sI1s8sI"
        (_, _, _, ts, oid, side, stock, shares, price) = struct.unpack(fmt, msg[:35])
        if stock.strip() != self.symbol:
            return
        side  = b'B' if side == b'B' else b'S'
        price = price / 10_000
        self.order_map[oid] = (price, shares, side)
        book = self.bids if side == b'B' else self.asks
        book[price][oid] = shares

    def _cancel(self, msg: bytes):
        fmt = ">QcI8sI"
        (_, _, _, ts, oid, cancelled) = struct.unpack(fmt, msg[:23])
        if oid not in self.order_map:
            return
        price, size, side = self.order_map[oid]
        remaining = size - cancelled
        book = self.bids if side == b'B' else self.asks
        if remaining <= 0:
            book[price].pop(oid, None)
            self.order_map.pop(oid, None)
        else:
            self.order_map[oid] = (price, remaining, side)
            book[price][oid]   = remaining

    def _delete(self, msg: bytes):
        fmt = ">QcI8s"
        (_, _, _, ts, oid) = struct.unpack(fmt, msg[:19])
        if oid not in self.order_map:
            return
        price, _, side = self.order_map.pop(oid)
        (self.bids if side == b'B' else self.asks)[price].pop(oid, None)

    def _replace(self, msg: bytes):
        fmt = ">QcI8s8sI"
        (_, _, _, ts, old_oid, new_oid, new_shares) = struct.unpack(fmt, msg[:31])
        if old_oid not in self.order_map:
            return
        price, _, side = self.order_map.pop(old_oid)
        self._delete(msg[:19] + struct.pack(">I", 0))  # rimuovi old
        self.order_map[new_oid] = (price, new_shares, side)
        (self.bids if side == b'B' else self.asks)[price][new_oid] = new_shares

    # ---------- SNAPSHOT ----------
    def snapshot(self, timestamp: float):
        bids = sorted(self.bids.items(), key=lambda x: -x[0])[:self.depth]
        asks = sorted(self.asks.items(), key=lambda x:  x[0])[:self.depth]
        snap = {'ts': timestamp}
        for i, (p, orders) in enumerate(bids, 1):
            snap[f'bid_p{i}'], snap[f'bid_s{i}'] = p, sum(orders.values())
        for i, (p, orders) in enumerate(asks, 1):
            snap[f'ask_p{i}'], snap[f'ask_s{i}'] = p, sum(orders.values())
        self.snapshots.append(snap)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.snapshots).set_index('ts')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rebuild L2 order book from ITCH")
    parser.add_argument("itch", help="Path al file ITCH")
    parser.add_argument("--symbol", required=True, help="Ticker es. AAPL")
    parser.add_argument("--depth", type=int, default=10, help="Profondità livelli")
    parser.add_argument("--out", default="orderbook.parquet", help="File di output")
    args = parser.parse_args()

    ob = OrderBookReplayer(args.symbol, args.depth)
    ob.parse_itch(args.itch)
    ob.to_df().to_parquet(args.out)
    print(f"Snapshot salvate in {args.out}")
