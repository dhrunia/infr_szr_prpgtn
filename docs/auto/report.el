(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "a4paper")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "geometry"
    "graphicx"
    "subcaption"
    "placeins"
    "amsmath"
    "amsfonts")
   (LaTeX-add-labels
    "fig:syn-data"
    "fig:data-features"
    "fig:vep_model"
    "tab:priors"
    "fig:nuts_diags_chain1"
    "fig:nuts_diags_chain2"
    "fig:nuts_diags_chain3"
    "fig:nuts_diags_chain4"))
 :latex)

