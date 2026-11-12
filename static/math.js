// Renders the LaTeX in paper text ($...$, $$...$$, \(...\), \[...\]) with KaTeX.
// Loaded with `defer` after KaTeX and its auto-render extension (see templates).
// Formulas KaTeX can't parse (e.g. paper-specific macros) are left as their source, not shown as errors.
window.renderMath = function (el) {
  if (!el || typeof renderMathInElement !== "function") return;
  renderMathInElement(el, {
    delimiters: [
      {left: "$$", right: "$$", display: true},
      {left: "\\[", right: "\\]", display: true},
      {left: "\\(", right: "\\)", display: false},
      {left: "$", right: "$", display: false},
    ],
    // Common author-defined macros seen in the collection (KaTeX only knows standard LaTeX).
    macros: {
      "\\GL": "\\mathrm{GL}", "\\SL": "\\mathrm{SL}", "\\SO": "\\mathrm{SO}", "\\SU": "\\mathrm{SU}",
      "\\PGL": "\\mathrm{PGL}", "\\Hom": "\\operatorname{Hom}", "\\End": "\\operatorname{End}",
      "\\Aut": "\\operatorname{Aut}", "\\Img": "\\operatorname{Im}", "\\Mod": "\\operatorname{Mod}",
      "\\SMod": "\\operatorname{SMod}", "\\Homeo": "\\operatorname{Homeo}", "\\arsinh": "\\operatorname{arsinh}",
      "\\arcosh": "\\operatorname{arcosh}", "\\tr": "\\operatorname{tr}", "\\rank": "\\operatorname{rank}",
      "\\supp": "\\operatorname{supp}", "\\sgn": "\\operatorname{sgn}", "\\diag": "\\operatorname{diag}",
      "\\prescript": "{}^{#1}_{#2}{#3}", "\\R": "\\mathbb{R}", "\\N": "\\mathbb{N}", "\\Z": "\\mathbb{Z}",
      "\\Q": "\\mathbb{Q}", "\\C": "\\mathbb{C}", "\\E": "\\mathbb{E}",
    },
    throwOnError: true,   // with an errorCallback, auto-render then keeps the formula's raw source
    errorCallback: () => {},
    ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "input", "button"],
  });
};
document.addEventListener("DOMContentLoaded", () => window.renderMath(document.querySelector("main") || document.body));
