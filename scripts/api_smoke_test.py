"""Call the running API for every trip and print the server-side metrics.

Owner: Hansika R (PES2UG24CS178)

Run (from any machine on the same network):
  python scripts/api_smoke_test.py --api http://<laptop1-ip>:8000
"""
import argparse
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    a = ap.parse_args()
    base = a.api.rstrip("/")
    h = requests.get(f"{base}/health", timeout=10).json()
    print("health:", h)
    trips = requests.get(f"{base}/trips", timeout=30).json()["trips"]
    print(f"{len(trips)} trips\n")
    print(f"{'trip':>10} {'fixes':>6} {'acc_hmm':>8} {'acc_clf':>8} {'routeP':>7} {'routeR':>7} "
          f"{'routeF1':>7} {'contin':>7} {'secs':>6}")
    for t in trips:
        t0 = time.time()
        r = requests.post(f"{base}/predict", json={"trip_id": t["trip_id"]}, timeout=600)
        r.raise_for_status()
        m = r.json()["metrics"]
        print(f"{t['trip_id']:>10} {t['n_obs']:>6} {m.get('observation_accuracy', float('nan')):>8.4f} "
              f"{m.get('classifier_only_accuracy', float('nan')):>8.4f} {m.get('route_precision', float('nan')):>7.3f} "
              f"{m.get('route_recall', float('nan')):>7.3f} {m.get('route_f1', float('nan')):>7.3f} "
              f"{m['route_continuity']:>7.4f} {time.time() - t0:>6.2f}")


if __name__ == "__main__":
    main()
