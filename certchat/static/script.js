document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("chatbot-btn");
  const chatWindow = document.getElementById("chatbot-window");
  const sendBtn = document.getElementById("send-btn");
  const userInput = document.getElementById("user-input");
  const chatMessages = document.getElementById("chat-messages");
  const icon = btn.querySelector("img"); // ✅ targets image icon inside the button

  // Modal elements
  const modal = document.getElementById("full-answer-modal");
  const modalText = document.getElementById("modal-text");
  const modalClose = document.getElementById("modal-close");

  if (!btn || !chatWindow) {
    console.error("Missing chatbot elements!");
    return;
  }

  // -----------------------------
  // Format text to preserve line breaks, spaces, and code blocks
  // -----------------------------
  function formatText(text) {
    if (!text) return "";
    let escaped = text.replace(/&/g, "&amp;")
                      .replace(/</g, "&lt;")
                      .replace(/>/g, "&gt;");
    escaped = escaped.replace(/```([\s\S]*?)```/g, "<pre>$1</pre>");
    escaped = escaped.replace(/\n/g, "<br>");
    escaped = escaped.replace(/ {2}/g, "&nbsp;&nbsp;");
    return escaped;
  }

  // -----------------------------
  // Add Message to Chat Window
  // -----------------------------
  function addMessage(text, sender = "bot") {
    const wrapper = document.createElement("div");
    wrapper.className = sender === "user"
      ? "message-wrapper user-side"
      : "message-wrapper bot-side";

    const bubble = document.createElement("div");
    bubble.className = sender === "user" ? "user-message" : "bot-message";
    bubble.innerHTML = formatText(text);

    wrapper.appendChild(bubble);
    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // -----------------------------
  // Typing Animation (3 dots)
  // -----------------------------
  function showTyping() {
    if (document.getElementById("typing-wrapper")) return;
    const w = document.createElement("div");
    w.id = "typing-wrapper";
    w.className = "message-wrapper bot-side";
    w.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
    chatMessages.appendChild(w);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function removeTyping() {
    const t = document.getElementById("typing-wrapper");
    if (t) t.remove();
  }

  // -----------------------------
  // Fullscreen Modal for Long Answers
  // -----------------------------
  function showFullAnswer(text) {
    modalText.innerHTML = formatText(text);
    modal.style.display = "block";

    modalClose.onclick = () => {
      modal.style.display = "none";
    };

    window.onclick = (event) => {
      if (event.target === modal) {
        modal.style.display = "none";
      }
    };
  }

  // -----------------------------
  // Letter-by-letter Typing Effect for Bot
  // -----------------------------
  function typeBotMessage(text) {
    // Show in modal if answer is long or contains table/code
    if (text.length > 500 || text.includes("|") || text.includes("```")) {
      removeTyping();
      showFullAnswer(text);
      return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "message-wrapper bot-side";
    const msgDiv = document.createElement("div");
    msgDiv.className = "bot-message";
    wrapper.appendChild(msgDiv);
    chatMessages.appendChild(wrapper);

    let i = 0;
    const speed = 25;

    function typeNext() {
      if (i < text.length) {
        const char = text.charAt(i);
        if (char === "\n") msgDiv.innerHTML += "<br>";
        else if (char === " ") msgDiv.innerHTML += "&nbsp;";
        else msgDiv.innerHTML += char;
        i++;
        chatMessages.scrollTop = chatMessages.scrollHeight;
        setTimeout(typeNext, speed);
      }
    }

    typeNext();
  }

  // -----------------------------
  // First Greeting Message
  // -----------------------------
  let greeted = false;
  function showGreeting() {
    if (greeted) return;
    greeted = true;

    showTyping();
    setTimeout(() => {
      removeTyping();
      typeBotMessage("👋 Hi! I’m your document assistant. Ask me anything from your uploaded PDFs.");
    }, 1000);
  }

  // -----------------------------
  // Toggle Chat Window
  // -----------------------------
  btn.addEventListener("click", () => {
    chatWindow.classList.toggle("open");
    const isOpen = chatWindow.classList.contains("open");

    // Toggle icon image
    icon.src = isOpen
      ? "https://bb.branding-element.com/prod/130995/130995-ChatGPT_Image_Jun_24__2025__03_38_10_PM-removebg-preview.png" // close icon
      : "https://bb.branding-element.com/prod/130995/130995-ChatGPT_Image_Jun_24__2025__03_38_10_PM-removebg-preview.png"; // chat icon

    icon.alt = isOpen ? "Close chat" : "Open chat";

    if (isOpen && userInput) {
      userInput.focus();
      showGreeting();
    }
  });

  // -----------------------------
  // Send Message to Backend
  // -----------------------------
  async function sendMessage() {
    const question = userInput.value.trim();
    if (!question) return;

    addMessage(question, "user");
    userInput.value = "";
    userInput.focus();

    showTyping();

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();
      removeTyping();

      if (!res.ok || data.error) {
        typeBotMessage(data.error || "⚠️ Server error");
      } else {
        typeBotMessage(data.answer || "🤔 No answer available");
      }
    } catch (err) {
      console.error("Fetch error:", err);
      removeTyping();
      typeBotMessage("⚠️ Error connecting to server.");
    }
  }

  // -----------------------------
  // Event Listeners
  // -----------------------------
  if (sendBtn) sendBtn.addEventListener("click", sendMessage);
  if (userInput) {
    userInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") sendMessage();
    });
  }
});
