"""catalog_mlp: forecasting an M>=threshold event within a horizon.

A small MLP on features derived from the earthquake catalogue alone -- the one
forecaster of fourteen tried in this project that clears its own fold's floor.

`forecast.cli.main` is the `forecast` entry point. Importing this package pulls
in nothing but the docstring, so `import forecast` does not drag torch behind
it.
"""
