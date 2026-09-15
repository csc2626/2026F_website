// Works around a reveal.js + MathJax race condition: MathJax typesets
// formulas asynchronously, after reveal.js has already scaled/painted the
// slide (via the "smaller: true" transform, and the chalkboard canvas
// overlay). The result is formulas that render as blank/white until the
// user scrolls or clicks, which forces reveal.js to recompute layout.
//
// This forces that layout recompute automatically whenever MathJax injects
// rendered math into the current slide, so formulas show up immediately.
(function () {
  var layoutTimer = null;

  function scheduleLayout() {
    if (layoutTimer) clearTimeout(layoutTimer);
    layoutTimer = setTimeout(function () {
      if (window.Reveal && typeof Reveal.layout === "function") {
        Reveal.layout();
      }
    }, 30);
  }

  function init() {
    if (!window.Reveal) return;

    Reveal.on("ready", scheduleLayout);
    Reveal.on("slidechanged", function () {
      // MathJax typesets the newly shown slide asynchronously; nudge
      // reveal.js again shortly after in case the mutation observer below
      // misses it (e.g. math re-typeset without adding new nodes).
      setTimeout(scheduleLayout, 50);
      setTimeout(scheduleLayout, 300);
    });

    var slides = document.querySelector(".reveal .slides");
    if (slides && window.MutationObserver) {
      var observer = new MutationObserver(function (mutations) {
        for (var i = 0; i < mutations.length; i++) {
          var added = mutations[i].addedNodes;
          for (var j = 0; j < added.length; j++) {
            var node = added[j];
            if (
              node.nodeType === 1 &&
              node.tagName &&
              node.tagName.toLowerCase() === "mjx-container"
            ) {
              scheduleLayout();
              return;
            }
          }
        }
      });
      observer.observe(slides, { childList: true, subtree: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
