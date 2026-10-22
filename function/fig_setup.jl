# Figure size in cm with a print font size, for Makie (CairoMakie).
# Makie sizes and font sizes are in pt (1/72 inch); save with px_per_unit = dpi/72, so a figure built
# 18 cm wide prints 18 cm wide and fontsize 10 prints at 10 pt.
# Default: 10 pt on an 18 cm page. A figure wider than the page will be shrunk to fit it, so its font is
# scaled up by w/page to still read as 10 pt; narrower figures keep 10 pt (placed at their own width).
#   fig = Figure(; figkw(18, 12)...)                                # 18 x 12 cm, 10 pt
#   fig = Figure(; figkw(9, 8)..., figure_padding = (5, 5, 5, 5))   # half-page panel, 10 pt
#   figsave(figpath("sub", "name.png"), fig)                        # 300 dpi
const cm_to_pt = 72 / 2.54
const PAGEW_CM = 18.0     # cm, page (text) width
const FONT_PT  = 10.0     # pt, font size on that page

figkw(w_cm, h_cm; fs = FONT_PT, page = PAGEW_CM) = (; size = (w_cm * cm_to_pt, h_cm * cm_to_pt), fontsize = fs * max(w_cm / page, 1.0))

figsave(path, fig; dpi = 300) = save(path, fig; px_per_unit = dpi / 72)   # needs CairoMakie loaded in the calling script
