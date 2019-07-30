
"""
Library of useful routines for virtual epileptic patient workflows.

"""


from .io.stan import (
    cmdstan_path,
    compile_model,
    parse_csv,
    rdump
)

from .plots.stan import (
    pair_plots,
    trace_nuts
)

from .plots.network import (
    phase_space
)

from .plots.seeg import (
    ppc_seeg,
    violin_x0
)
