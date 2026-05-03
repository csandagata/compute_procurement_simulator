import { init } from "./ui.js";

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
