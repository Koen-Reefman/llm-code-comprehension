// @ts-check

// Helper function to add elements to the history panel
function _addToHistory(role, message) {
  const div = document.createElement("div");
  div.className = "history-item";
  div.innerHTML = `${role}: ${message}`;
  document.getElementById("historyBody")?.appendChild(div);
}

// Helper function to add text to the last child of the history body (<should> always be llm text)
function _addLLMText(message) {
  const llmChild = document.getElementById("historyBody")?.lastChild;

  if (llmChild) {
    // @ts-ignore - We know llmChild (and therefore innerHTML exist, so ignore error)
    llmChild.innerHTML = llmChild.innerHTML + message;
  }
}

// Main function
(function () {
  // @ts-ignore - We know this function exists, so we can ignore the error
  const vscode = acquireVsCodeApi();

  let llmBusy = false;
  let questionAsked = false;

  // Add event listener to receive data from the extension
  window.addEventListener("message", (evt) => {
    const message = evt.data;

    switch (message.type) {
      // Add user input to the history div
      case "user-input":
        _addToHistory("User", message.value);

        this.document.getElementById("feedback-container")?.classList.add("hidden");
        break;
      // Add llm response to the history div; Since this is a stream, we have two different functions for this
      case "llm-response":
        if (llmBusy && !message.done) {
          _addLLMText(message.value);
          llmBusy = true;
        } else if (!llmBusy && !message.done) {
          _addToHistory("LLM", message.value);
          llmBusy = true;
        } else if (llmBusy && message.done) {
          _addLLMText(message.value);
          llmBusy = false;
          questionAsked = false;

          this.document.getElementById("feedback-container")?.classList.remove("hidden");
        }

        break;
      case "llm-error":
        _addToHistory("LLM", message.value);
        llmBusy = false;
        questionAsked = false;
        break;
      case "selected-code":
        const messageVal = message.value;

        if (messageVal !== "") {
          // @ts-ignore - selected-code is never null
          document.getElementById("selected-code").innerText = messageVal;
        } else {
          // @ts-ignore - selected-code is never null
          document.getElementById("selected-code").innerText =
            "No code selected, please select code before asking a question";
        }
        break;
      case "error-message":
        _addToHistory("Error", message.value);
        llmBusy = false;
        questionAsked = false;
        break;
      default:
        break;
    }
  });

  // Event listener for pressing enter to submit a prompt to the LLM
  const userInput = document.getElementById("userInput");
  userInput?.addEventListener("keypress", (evt) => {
    if (evt.key === "Enter") {
      if (questionAsked) {return;};

      vscode.postMessage({
        type: "submit_request",
        // @ts-ignore - Assume userInput exists
        value: userInput?.value,
        persistentHistory:
          // @ts-ignore - checked does exists
          document.getElementById("persistent-history")?.checked,
        // @ts-ignore - value field does exist on a dropdown menu
        selectedModel: document.getElementById("selected-model")?.value,
      });

      questionAsked = true;

      // @ts-ignore - Assume userInput exists
      userInput.value = "";
    }
  });

  // Event listener for clicking the submit "send" button
  const userInputBtn = document.getElementById("userInputBtn");
  userInputBtn?.addEventListener("click", () => {
    if (questionAsked) {return;};

    vscode.postMessage({
      type: "submit_request",
      // @ts-ignore - Assume userInput exists
      value: userInput?.value,
      persistentHistory:
        // @ts-ignore - checked does exists
        document.getElementById("persistent-history")?.checked,
      // @ts-ignore - value field does exist on a dropdown menu
      selectedModel: document.getElementById("selected-model")?.value,
    });

    questionAsked = true;

    // @ts-ignore - Assume userInput exists
    userInput.value = "";
  });

  // Event listener for the "clear history" button
  const clearBtn = document.getElementById("eraseHistory");
  clearBtn?.addEventListener("click", () => {
    if (questionAsked || llmBusy) {return;};

    // @ts-ignore - Assume historyBody exists
    document.getElementById("historyBody").innerHTML = "";

    vscode.postMessage({ type: "reset-llm" });
  });

  document.getElementById("common-q1")?.addEventListener("click", () => {
    if (questionAsked) {return;};

    vscode.postMessage({
      type: "submit_request",
      value: "What does the selected code do?",
      persistentHistory:
        // @ts-ignore - checked does exists
        document.getElementById("persistent-history")?.checked,
      // @ts-ignore - value field does exist on a dropdown menu
      selectedModel: document.getElementById("selected-model")?.value,
    });

    questionAsked = true;
  });

  document.getElementById("common-q2")?.addEventListener("click", () => {
    if (questionAsked) {return;};

    vscode.postMessage({
      type: "submit_request",
      value: "How can I use the selected code?",
      persistentHistory:
        // @ts-ignore - checked does exists
        document.getElementById("persistent-history")?.checked,
      // @ts-ignore - value field does exist on a dropdown menu
      selectedModel: document.getElementById("selected-model")?.value,
    });

    questionAsked = true;
  });

  document.getElementById("common-q3")?.addEventListener("click", () => {
    if (questionAsked) {return;};

    vscode.postMessage({
      type: "submit_request",
      value: "Where is the selected code used?",
      persistentHistory:
        // @ts-ignore - checked does exists
        document.getElementById("persistent-history")?.checked,
      // @ts-ignore - value field does exist on a dropdown menu
      selectedModel: document.getElementById("selected-model")?.value,
    });

    questionAsked = true;
  });

  setInterval(() => {
    vscode.postMessage({ type: "get_selected" });
  }, 500);

  //
})();
