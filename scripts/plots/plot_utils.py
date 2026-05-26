import matplotlib.pyplot as plt
import seaborn as sb
from tueplots import figsizes, fonts, fontsizes

PALETTE_NAME = "colorblind"


def get_palette_colors() -> list[str]:
    return sb.palettes.SEABORN_PALETTES[PALETTE_NAME]


def setup_tueplots(
    nrows: int,
    ncols: int,
    rel_width: float = 1.0,
    hw_ratio: float | None = None,
    default_smaller: int = -2,
    use_tex: bool = False,
    tight_layout: bool = False,
    constrained_layout: bool = False,
    **kwargs,
):
    if use_tex:
        font_config = fonts.neurips2024_tex(family="sans-serif")
    else:
        font_config = fonts.neurips2024(family="sans-serif")
    if hw_ratio is not None:
        kwargs["height_to_width_ratio"] = hw_ratio
    size = figsizes.neurips2024(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        tight_layout=tight_layout,
        constrained_layout=constrained_layout,
        **kwargs,
    )
    fontsize_config = fontsizes.neurips2024(default_smaller=default_smaller)
    rc_params = {
        **font_config,
        **size,
        **fontsize_config,
    }
    rc_params.update({"text.latex.preamble": r"\usepackage{amsfonts}"})
    plt.rcParams.update(rc_params)
    sb.color_palette(PALETTE_NAME)
