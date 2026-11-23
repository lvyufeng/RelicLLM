"""Do the per-pair numbers quoted in the README match the pair receipts exactly?"""
import json
from pathlib import Path

OUT = Path('docs/performance/evidence/ascend-pure-b1-20261122/follow-up/rounds-49-67')
readme = (OUT / 'README.md').read_text()

print('%-6s %-34s %-34s %s' % ('pair', 'from receipt', 'quoted in README', 'ok'))
all_ok = True
for index, name in enumerate(('indexer-sort-pair-20.json',
                              'indexer-sort-pair-21.json',
                              'indexer-sort-pair-22.json'), start=20):
    d = json.loads((OUT / name).read_text())
    c_tps, c_ms = d['control_TPS'], d['control_ms_per_token']
    a_tps, a_ms = d['candidate_TPS'], d['candidate_ms_per_token']
    expected = ('%.4f tok/s, %.3f ms' % (c_tps, c_ms), '%.4f tok/s, %.3f ms' % (a_tps, a_ms))
    present = ('%.4f' % c_tps) in readme and ('%.3f' % c_ms) in readme \
        and ('%.4f' % a_tps) in readme and ('%.3f' % a_ms) in readme
    all_ok = all_ok and present
    print('%-6d %-34s %-34s %s'
          % (index, 'control ' + expected[0], 'candidate ' + expected[1], present))

print()
print('=== the vs-previous-best column ===')
for index, name in enumerate(('indexer-sort-pair-20.json',
                              'indexer-sort-pair-21.json',
                              'indexer-sort-pair-22.json'), start=20):
    d = json.loads((OUT / name).read_text())
    gain = 100 * (d['candidate_TPS'] / 14.064584 - 1)
    quoted = '%+.3f%%' % gain
    print('  pair %d  computed %-10s quoted in README %s' % (index, quoted, quoted in readme))
    all_ok = all_ok and (quoted in readme)

print()
print('ALL README PAIR NUMBERS VERIFIED' if all_ok else 'MISMATCH FOUND')
