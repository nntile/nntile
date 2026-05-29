"""Inference gptneox via NNTile graph API (scaffold)."""
from nntile import Context, NNGraph

def main() -> None:
    ctx = Context()
    _ = ctx, NNGraph()
    raise SystemExit('Example scaffold: implement gptneox inference')

if __name__ == '__main__':
    main()
