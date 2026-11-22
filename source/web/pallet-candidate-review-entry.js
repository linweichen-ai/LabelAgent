(() => {
  "use strict";

  const entry = document.getElementById("palletCandidateReviewEntry");
  const badge = document.getElementById("palletCandidateReviewBadge");
  if (!entry || !badge) return;

  function hideEntry() {
    entry.hidden = true;
    entry.removeAttribute("data-ready");
    badge.textContent = "0";
  }

  async function refreshEntry() {
    hideEntry();
    try {
      const response = await fetch("/api/pallet-candidate-review", {
        cache: "no-store",
        headers: {Accept: "application/json"},
      });
      if (!response.ok) return;
      const status = await response.json();
      if (status.available !== true || status.read_only !== true
          || status.production_deployed !== false
          || status.human_truth_writes !== 0
          || status.training_data_writes !== 0
          || !Number.isInteger(status.candidate_count)
          || status.candidate_count < 1
          || typeof status.url !== "string"
          || !status.url.startsWith("/pallet-candidate-review/")) return;
      entry.href = status.url;
      badge.textContent = String(status.candidate_count);
      entry.dataset.ready = "true";
      entry.hidden = false;
    } catch (_) {
      hideEntry();
    }
  }

  refreshEntry();
  window.addEventListener("pageshow", refreshEntry);
})();
