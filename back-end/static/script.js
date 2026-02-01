function sendMessage() {
    const input = document.getElementById("userInput");
    const message = input.value.trim();
    if (!message) return;

    addMessage(message, "user");
    input.value = "";

    // Fake bot reply (replace with Flask later)
    setTimeout(() => {
        addMessage("This is a bot reply 👋", "bot");
    }, 500);
}

function addMessage(text, sender) {
    const chat = document.getElementById("chatMessages");

    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", sender);

    const bubble = document.createElement("div");
    bubble.classList.add("bubble");
    bubble.innerText = text;

    messageDiv.appendChild(bubble);
    chat.appendChild(messageDiv);

    chat.scrollTop = chat.scrollHeight;
}

