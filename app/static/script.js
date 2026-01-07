// app/static/script.js

const messagesDiv = document.getElementById("messages");
const promptInput = document.getElementById("prompt");
const sendBtn = document.getElementById("send-btn");
const statusText = document.getElementById("status-text");

// Safety panel elements
const actionBadge = document.getElementById("action-badge");
const driftLabel = document.getElementById("drift-label");
const driftFill = document.getElementById("drift-fill");
const sanitizedDiv = document.getElementById("sanitized-prompt");
const flaggedList = document.getElementById("flagged-list");

// USER history only (no llm responses)
let userHistory = [];

// ---------------------- UI Helpers ----------------------

function addMessage(text, role) {
  const div = document.createElement("div");
  div.classList.add("msg", role);
  div.textContent = text;
  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function setSafetyPanel(safety) {
  if (!safety) return;

  const action = safety.action || "unknown";
  actionBadge.textContent = action;

  // Badge colours
  actionBadge.classList.remove("allow", "soft_review", "block_or_rewrite");
  actionBadge.classList.add(action);

  driftLabel.textContent = safety.drift_score.toFixed(2);
  driftFill.style.width = Math.min(safety.drift_score, 100) + "%";

  sanitizedDiv.textContent = safety.sanitized_prompt || "";

  flaggedList.innerHTML = "";
  (safety.flagged_texts || []).forEach((frag) => {
    const li = document.createElement("li");
    li.textContent = frag;
    flaggedList.appendChild(li);
  });
}


// ---------------------- MAIN SEND FUNCTION ----------------------

async function sendPrompt() {
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  addMessage(prompt, "user");
  promptInput.value = "";
  promptInput.focus();

  sendBtn.disabled = true;
  statusText.textContent = "Analyzing & calling LLM...";

  try {
    const res = await fetch("/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        history: userHistory,  // ALREADY fixed to ONLY user messages
      }),
    });

    const data = await res.json();
    const safety = data.safety_output;
    const llmText = data.llm_output || "(no response)";

    // Show LLM output
    addMessage(llmText, "llm");

    // Update safety diagnostics panel
    setSafetyPanel(safety);

    // ⬇⬇⬇ SUPER IMPORTANT FIX ⬇⬇⬇
    // ONLY store prompt into history if "allow"
    if (safety.action === "allow") {
      userHistory.push(prompt);
    }

    statusText.textContent = "Done.";
  } catch (err) {
    console.error(err);
    addMessage("Error: " + err.message, "system");
    statusText.textContent = "Backend error";
  } finally {
    sendBtn.disabled = false;
  }
}

// ---------------------- Event Listeners ----------------------

sendBtn.addEventListener("click", (e) => {
  e.preventDefault();
  sendPrompt();
});

promptInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendPrompt();
  }
});
