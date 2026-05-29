"""Train t5 via NNTile graph API (scaffold)."""
from nntile import Context, NNGraph


def main() -> None:
    ctx = Context()
    _ = ctx, NNGraph()
    raise SystemExit('Example scaffold: implement t5 training')

if __name__ == '__main__':
    main()
