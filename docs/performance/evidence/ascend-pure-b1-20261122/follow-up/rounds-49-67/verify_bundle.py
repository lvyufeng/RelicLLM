"""Check the evidence bundle is internally consistent: hashes, JSON, and the numbers the README states."""
import hashlib
import json
import re
from pathlib import Path

OUT = Path('docs/performance/evidence/ascend-pure-b1-20261122/follow-up/rounds-49-67')
summary = json.loads((OUT / 'summary.json').read_text())

print('=== manifest integrity ===')
bad = []
for name, digest in summary['manifest'].items():
    path = OUT / name
    if not path.exists():
        bad.append((name, 'missing'))
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        bad.append((name, 'hash mismatch'))
print('files in manifest:', len(summary['manifest']))
print('integrity:', 'ALL MATCH' if not bad else bad)

print()
print('=== every JSON in the bundle parses ===')
for path in sorted(OUT.glob('*.json')):
    try:
        json.loads(path.read_text())
        status = 'ok'
    except Exception as error:  # noqa: BLE001
        status = 'FAILED: %s' % error
    if status != 'ok':
        print('  %-52s %s' % (path.name, status))
print('  json files:', len(list(OUT.glob('*.json'))))

print()
print('=== README numbers against the registration receipt ===')
reg = json.loads((OUT / 'indexer-sort-registration-three-pair.json').read_text())['registration']
readme = (OUT / 'README.md').read_text()
checks = [
    ('mean', '%.6f' % reg['mean_candidate_TPS'], '14.287598'),
    ('ms', '%.4f' % reg['mean_candidate_ms_per_token'], '69.9922'),
    ('gain', '%.3f' % reg['gain_over_previous_best_percent'], '1.586'),
    ('best', '%.6f' % reg['best_candidate_TPS'], '14.376778'),
    ('worst', '%.6f' % reg['worst_candidate_TPS'], '14.221390'),
]
for label, from_receipt, quoted in checks:
    in_readme = quoted.rstrip('0').rstrip('.') in readme or quoted in readme
    agree = from_receipt.rstrip('0').rstrip('.') == quoted.rstrip('0').rstrip('.') or \
            abs(float(from_receipt) - float(quoted)) < 1e-9
    print('  %-8s receipt %-12s README %-12s agree %s  present %s'
          % (label, from_receipt, quoted, agree, in_readme))

print()
print('=== routing numbers against their receipt ===')
routing = json.loads((OUT / 'routing-imbalance-receipt.json').read_text())
print('  dispatches', routing.get('calls_recorded'))
print('  spread>=2 fraction %.3f' % (routing['calls_with_spread_at_least_2'] / routing['calls_recorded']))
print('  mean max %.3f  mean min %.3f' % (routing['mean_max_count'], routing['mean_min_count']))
