// Works around a reveal.js + MathJax race condition: MathJax typesets
// formulas asynchronously, after reveal.js has already scaled/painted the
// slide (via the "smaller: true" transform, and the chalkboard canvas
// overlay). The result is formulas that render as blank/white until the
// user scrolls or clicks, which forces reveal.js to recompute layout.
//
// This forces that layout recompute automatically whenever MathJax injects
// rendered math into the current slide, so formulas show up immediately.
//
// It also typesets any formula that is still raw LaTeX. Quarto loads
// MathJax 3 through reveal.js's MathJax 2 plugin, which crashes on
// MathJax.Hub before it can hook into slide changes, so MathJax's single
// startup pass is otherwise the only chance formulas get typeset.
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

  function untypesetMath() {
    var spans = document.querySelectorAll(".reveal .slides span.math");
    return Array.prototype.filter.call(spans, function (el) {
      return !el.querySelector("mjx-container");
    });
  }

  // Returns false while MathJax 3 has not finished loading yet.
  function typesetPending() {
    var mj = window.MathJax;
    if (!mj || !mj.startup || !mj.startup.promise ||
        typeof mj.typesetPromise !== "function") {
      return false;
    }
    // Chaining onto startup.promise keeps this from running concurrently
    // with MathJax's own startup typeset or a previous call.
    var run = function () {
      var pending = untypesetMath();
      if (pending.length === 0) return;
      return mj.typesetPromise(pending).then(scheduleLayout);
    };
    mj.startup.promise = mj.startup.promise
      .then(run, run)
      .catch(function (err) {
        console.error("MathJax typeset failed:", err);
      });
    return true;
  }

  function typesetWhenMathJaxLoads() {
    var attempts = 0;
    var poll = setInterval(function () {
      if (typesetPending() || ++attempts > 120) clearInterval(poll);
    }, 250);
  }

  function onReady() {
    scheduleLayout();
    typesetPending();
  }

  function init() {
    if (!window.Reveal) return;

    if (typeof Reveal.isReady === "function" && Reveal.isReady()) {
      onReady();
    } else {
      Reveal.on("ready", onReady);
    }
    typesetWhenMathJaxLoads();

    Reveal.on("slidechanged", function () {
      typesetPending();
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
