"""`forecast` -- the front door to catalog_mlp.

    forecast                  # this listing
    forecast train --help     # the trainer's own flags, unchanged
    forecast train --catalog-path catalogs/catalog_current.csv \
        --catalog-span 2000-01-01 2026-08-12 --horizon-days 14 --cv-folds 2

Arguments after the command are passed through untouched, and each command is
exactly the underlying module's `main()`. A result here is expected to be
traceable to the command that produced it, and a front end that rewrote
arguments would make the recorded command and the real one diverge. `train`
also runs standalone as `python -m forecast.train`.
"""
import importlib
import sys

COMMANDS = {
    "train": ("forecast.train", "train catalog_mlp and score it against its floor"),
}


def usage():
    """Prints the command listing."""
    print("forecast -- catalog_mlp: earthquake forecasting from the catalogue\n")
    print("usage: forecast <command> [args...]"
          "        (each command has its own --help)\n")
    for name, (_, summary) in COMMANDS.items():
        print(f"  {name:<7} {summary}")
    print("\nThe input is one catalogue CSV. There is no waveform archive here:")
    print("of the fourteen forecasters this project tried, the catalogue-derived")
    print("one is the one that beat persistence, and the waveform and chaotic-")
    print("feature arms are a documented negative that stays in cnn_earthquake.")
    print("\nEvery number is reported beside the floor it had to clear on its own")
    print("fold. Fold spread on this data is 0.07-0.16, so a pooled AUC misleads.")
    return 0


def main():
    """Dispatches to one command's `main()`, leaving its arguments untouched."""
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        return usage()

    name = argv[0]
    if name not in COMMANDS:
        print(f"forecast: unknown command {name!r}", file=sys.stderr)
        print("run `forecast` for the list", file=sys.stderr)
        return 2

    module = COMMANDS[name][0]
    sys.argv = [f"forecast {name}"] + argv[1:]
    mod = importlib.import_module(module)
    return mod.main() or 0


if __name__ == "__main__":
    sys.exit(main())
