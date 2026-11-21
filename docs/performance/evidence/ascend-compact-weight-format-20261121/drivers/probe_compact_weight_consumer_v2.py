"""Apply the same independent corpus and real-projection gates to native v2."""
from pathlib import Path
import probe_compact_weight_consumer as base


def main():
    path = Path(base.__file__)
    source = path.read_text()
    marker = "'logs/compact-build.json'"
    if source.count(marker) != 1:
        raise RuntimeError('the immutable build receipt boundary moved')
    source = source.replace(marker, "'logs/compact-build-v2.json'", 1)
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), base.__dict__)
    return base.main()


if __name__ == '__main__':
    main()
