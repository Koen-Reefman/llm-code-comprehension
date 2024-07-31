import * as vscode from "vscode";

// Import types
import { RepoType, callHierarchyRepresentation, codeSnippet } from "./types";

// To parse HTML, so we don't need to put entire HTML template in this ts file
const Handlebars = require("handlebars");
import html from "../resources/main.html"; // The actual HTML file

// All LangChain stuff
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { ConversationSummaryBufferMemory } from "langchain/memory";
import { ConversationChain } from "langchain/chains";
import { Ollama } from "@langchain/community/llms/ollama";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "langchain/document";

// Used to calculate amount of tokens for a given string/prompt
// @ts-ignore - We're sure mistral-tokenizer-js is installed
// import llmTokenizer from "mistral-tokenizer-js";

//@ts-ignore - We're sure llama3-tokenizer-js is installed
import llmTokenizer from "llama3-tokenizer-js";

import { appendFile, readdirSync, writeFile, writeFileSync, writeSync } from "fs";
import { loadEvaluator } from "langchain/evaluation";

// Used to keep track if new chain with new history has to be made (when selecting a new piece of code)
let fullPrompt: boolean = true;
let lastMethod: string = "";

// The actual LangChain chain
let chain: ConversationChain;

// Controls how large the context size of the LLM is
let maxTokensPrompt: number = 2048; // 2048
let historyTokens: number = 2048;

// Variables to control what gets put in the prompt
let includeFileName: boolean = true;
let useSimilar: boolean = true;
let similarAmount: number = 20; // 8
let useHierarchy: boolean = true;
let hierarchyMaxDepth: number = 3; // 2
let hierarchyMaxWidth: number = 10; // 6
let useRepoStructure: boolean = true;

// Variables for the automated experiments
let maxCHTokens: number = 500;
let maxSimilarTokens: number = 500;
let maxRepoTokens: number = 500;

// Keep track of everything (for experiments)
let amountOfTokensCH: number;
let amountOfTokensSimilar: number;
let amountofTokensRepo: number;
let fullLLMPrompt: string;

// Variables to control the way the repository structure is put into the context
let includeFiles: boolean = false;
let includeTests: boolean = true;
let onlyCHPaths: boolean = false;

// Model used for generation
// mistral, mistral:7b-instruct-v0.2-fp16, llama3:instruct, llama3:8b-instruct-q8_0
let ollamaModel: string = "llama3:instruct";

// Log everything user does
let sessionLog: any[] = [];

// https://github.com/microsoft/vscode-extension-samples/blob/main/webview-view-sample/src/extension.ts
export class SidebarProvider implements vscode.WebviewViewProvider {
  private _view?: vscode.WebviewView;
  private _filePath: vscode.Uri;

  constructor(private readonly _extensionUri: vscode.Uri, private readonly _sessionID: number) {
    // Create the log file, which we will use to log all interactions with the tool
    const fileName: string = `/../plugin-logs/session-${_sessionID}.txt`;
    const wsEdit: vscode.WorkspaceEdit = new vscode.WorkspaceEdit();
    const wsPath: string = (vscode.workspace.workspaceFolders as vscode.WorkspaceFolder[])[0].uri.fsPath;
    const filePath: vscode.Uri = vscode.Uri.file(wsPath + fileName);
    this._filePath = filePath;
    wsEdit.createFile(filePath, {
      ignoreIfExists: false,
      contents: undefined,
      overwrite: true,
    });
    vscode.workspace.applyEdit(wsEdit);
  }

  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    context: vscode.WebviewViewResolveContext<unknown>,
    token: vscode.CancellationToken
  ): void | Thenable<void> {
    this._view = webviewView;

    webviewView.webview.options = {
      // Allow scripts in the webview
      enableScripts: true,

      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

    // Listen for anything received from the extension window (inputs)
    webviewView.webview.onDidReceiveMessage(async (data) => {
      if (data.type === "user-feedback") {
        sessionLog.push({
          type: "user-feedback",
          timestamp: Date.now(),
          feedbackType: data.value,
        });

        this.writeFile();
      } else if (data.type === "reset-llm") {
        this.createChain();
      } else if (data.type === "get_selected") {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
          return;
        }

        // Get selected text in the editor
        const textStart = editor!.selection.start;
        const textEnd = editor!.selection.end;
        const textRange = new vscode.Range(textStart!, textEnd!);
        const selectedText = editor!.document.getText(textRange);

        // Print prompt in the Extension window
        this._view?.webview.postMessage({
          type: "selected-code",
          value: selectedText,
        });
      } else if (data.type === "submit_request") {
        // Check if user has an active text editor, if not: we return
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
          return;
        }

        // @ts-ignore - selectedModel is of type string, not any
        if (["mistral", "llama3:instruct", "llama3:8b-instruct-q8_0"].includes(data.selectedModel)) {
          ollamaModel = data.selectedModel;
        }

        // Print prompt in the Extension window
        this._view?.webview.postMessage({
          type: "user-input",
          value: data.value,
        });

        // Get selected text in the editor
        const textStart = editor!.selection.start;
        const textEnd = editor!.selection.end;
        const textRange = new vscode.Range(textStart!, textEnd!);

        try {
          this.handleRequest(data, editor, textRange);
        } catch (error) {
          // Print exception in the Extension window
          this._view?.webview.postMessage({
            type: "error-message",
            value: `An unknown error was caught - ${error}`,
          });
        }
      }
    });
  }

  private _getHtmlForWebview(webview: vscode.Webview) {
    // Get the local path to main script run in the webview, then convert it to a uri we can use in the webview.
    const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, "resources", "main.js"));

    // Do the same for the stylesheet.
    const styleResetUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, "resources", "reset.css"));
    const styleVSCodeUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, "resources", "vscode.css"));
    const styleMainUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, "resources", "main.css"));
    const trashIcon = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, "resources", "trash-icon.svg"));

    // Use a nonce to only allow a specific script to be run.
    const nonce = getNonce();

    const handleRes = Handlebars.compile(html)({
      cspSource: webview.cspSource,
      nonce: nonce,
      styleResetUri: styleResetUri,
      styleVSCodeUri: styleVSCodeUri,
      styleMainUri: styleMainUri,
      scriptUri: scriptUri,
      trashIcon: trashIcon,
    });

    return handleRes;
  }

  /**
   * Helper method to handle a request (either sent from frontend or automated tests)
   *
   * @param data - Data sent from the frontend, which contains the user query
   * @param editor - VSCode textedit object, which we use to get the selected text and filename
   * @param textRange - Range of the selected text
   * @param args - Custom args, just used in experiments to quickly change which settings should be used
   * @returns
   */
  private async handleRequest(
    data: any,
    editor: vscode.TextEditor,
    textRange: vscode.Range,
    args: string = "grammar-list"
  ): Promise<string> {
    const selectedText = editor!.document.getText(textRange);

    // Don't do anything if selected text is empty
    if (selectedText === "" && !data.persistentHistory) {
      this._view?.webview.postMessage({
        type: "llm-error",
        value: "Please select code for which you want an answer",
      });
      return "no-code";
    }

    // Dont do anything if the user's question is empty
    if (data.value === "") {
      this._view?.webview.postMessage({
        type: "llm-error",
        value: "Please enter a question",
      });
      return "no-question";
    }

    console.log(`persistentHistory: ${data.persistentHistory}`);
    console.log(typeof data.persistentHistory);
    console.log(selectedText);

    if (lastMethod === "" || selectedText !== lastMethod) {
      lastMethod = selectedText;
      fullPrompt = true;

      // Create new LangChain chain to reset memory for a new code snippet
      if (!data.persistentHistory) {
        this.createChain();
      }
    }

    // Initialize parts empty
    let repoStructure = "";
    let callHierarchy = "";
    let similarCode = "";

    let before = performance.now();
    // If full prompt needs to be constructed (when new code is selected), we gather relevant context for the prompt
    if (fullPrompt) {
      let callHierarchyList: any[] = [];
      try {
        if (useHierarchy) {
          before = performance.now();
          callHierarchyList = await getCallHierarchy(
            editor,
            textRange,
            selectedText,
            hierarchyMaxDepth,
            hierarchyMaxWidth
          );

          console.log("CH list");
          console.log(callHierarchyList);
          console.log("CH list done");

          console.log("ARGS:");
          console.log(args);
          console.log("-");

          callHierarchy = flattenHierarchy(callHierarchyList, "", 0, maxCHTokens, 0, args);
          console.log("Call hierarchy:");
          console.log(callHierarchy);
          console.log("CH done");
          console.log(`Call hierarchy took ${performance.now() - before} ms`);

          // Note how many tokens are used for call hierarchy
          amountOfTokensCH = llmTokenizer.encode(callHierarchy).length;
        }
      } catch (error) {
        // Print exception in the Extension window
        this._view?.webview.postMessage({
          type: "error-message",
          value:
            "Something went wrong while getting the call hierarchy - please refresh the plugin and wait for the Java extension to fully load",
        });

        console.error(error);
        return "error - problem with getting the call hierarchy";
      }

      try {
        if (useRepoStructure) {
          repoStructure = await getRepoStructure(editor.document.uri, callHierarchyList);
          console.log(`Repo structure took ${performance.now() - before} ms`);

          // Note how many tokens are used for repo structure
          amountofTokensRepo = llmTokenizer.encode(repoStructure).length;
        }
      } catch (error) {
        // Print exception in the Extension window
        this._view?.webview.postMessage({
          type: "error-message",
          value: "Something went wrong while getting the repository structure - please refresh the plugin",
        });

        return "error - problem with getting the repository structure";
      }

      try {
        if (useSimilar) {
          before = performance.now();

          console.log("Getting similar code");
          similarCode = await getSimilar(selectedText, similarAmount);
          console.log(`Getting similar code took ${performance.now() - before} ms`);
          // console.log(similarCode);

          // Note how many tokens are used for similar code snippets
          amountOfTokensSimilar = llmTokenizer.encode(similarCode).length;
        }
      } catch (error) {
        // Print exception in the Extension window
        this._view?.webview.postMessage({
          type: "error-message",
          value:
            "Something went wrong while getting similar code snippets - please try to refresh the plugin, make sure the chroma service and Ollama are running, and the `nomic-embed-text` model is pulled",
        });
        return "error - problem with getting similar code snippets";
      }
    }

    before = performance.now();
    const llmPrompt = await createPrompt(data.value, selectedText, repoStructure, callHierarchy, similarCode);
    console.log(`Creating pro mpt took ${performance.now() - before} ms`);
    // console.log(llmPrompt);

    console.log("===FULL PROMPT===");
    console.log(llmPrompt);
    console.log("===");

    const llmResponse = this.sendToLLM(llmPrompt, this._view?.webview!);

    // Save data about this request for analysis purposes
    sessionLog.push({
      type: "llm-interaction",
      timestamp: Date.now(),
      input: {
        selectedModel: ollamaModel,
        query: data.value,
        selectedCode: selectedText,
        persistent: data.persistentHistory,
      },
      llmResponse: await llmResponse,
    });

    // Write changes to file
    await this.writeFile();

    // await this.writeFile(
    //   `/../plugin-logs/interaction-${Date.now()}.txt`,
    //   JSON.stringify({
    //     timestamp: Date.now(),
    //     data: {
    //       query: data.value,
    //       selectedCode: selectedText,
    //       persistent: data.persistentHistory,
    //     },
    //     llmResponse: await llmResponse,
    //   })
    // );

    return llmResponse;
  }

  /**
   * Function that writes the array `sessionLog` to the current session log file
   * This function is called after every interaction with the LLM tool
   */
  private async writeFile(): Promise<void> {
    const wsEdit: vscode.WorkspaceEdit = new vscode.WorkspaceEdit();

    const enc = new TextEncoder();
    wsEdit.createFile(this._filePath, {
      ignoreIfExists: false,
      contents: enc.encode(JSON.stringify(sessionLog, null, 2)),
      overwrite: true,
    });

    await vscode.workspace.applyEdit(wsEdit);
  }

  /**
   * Helper function to create files in this workspace - individual logs per interaction
   * @param content
   */
  private async writeFileIndividual(location: string, content: string) {
    // Log interaction to a log file
    const enc = new TextEncoder();
    // https://stackoverflow.com/a/75702504
    const fileName = location;
    const wsEdit = new vscode.WorkspaceEdit();
    const wsPath = (vscode.workspace.workspaceFolders as vscode.WorkspaceFolder[])[0].uri.fsPath;
    const filePath = vscode.Uri.file(wsPath + fileName);
    wsEdit.createFile(filePath, {
      ignoreIfExists: false,
      contents: enc.encode(content),
      overwrite: true,
    });
    await vscode.workspace.applyEdit(wsEdit);
  }

  /**
   * Function used to send a query to the LLM and show the response on the frontend
   *
   * @param prompt - Query sent to the LLM
   * @param webview - Frontend component which we can send the response to
   */
  private async sendToLLM(prompt: string, webview: vscode.Webview) {
    // Save full prompt
    fullLLMPrompt = prompt;

    try {
      const timeAtStart = performance.now();
      let response = "";

      // https://stackoverflow.com/a/77230371
      await chain.call({
        input: prompt,
        callbacks: [
          {
            handleLLMNewToken(chunk: string) {
              webview.postMessage({
                type: "llm-response",
                value: chunk,
                done: false,
              });

              response += `${chunk}`;
            },
            handleLLMEnd(chunk: string) {
              webview.postMessage({
                type: "llm-response",
                value: "",
                done: true,
              });

              console.log(`Generating LLM response took ${performance.now() - timeAtStart}`);

              // @ts-ignore
              console.log(chain.memory);
              // console.log(llmTokenizer.encode(chain.memory).length);
            },
          },
        ],
      });

      return response;
    } catch (error) {
      // Print exception in the Extension window
      this._view?.webview.postMessage({
        type: "error-message",
        value:
          "Something went wrong when sending the prompt to the LLM - please check if Ollama is running and you have pulled the Mistral model",
      });

      return "";
    }
  }

  /**
   * Function to create a new langChain ConversationChain
   * New chain with new memory is created when selecting a new code snippet
   *
   * @param contextSize - Max amount of tokens in the context of the LLM used to generate the response
   * @param historySize - Max amount of tokens kept in the conversation history
   */
  private async createChain(contextSize: number = maxTokensPrompt, historySize: number = historyTokens) {
    try {
      let ollama: Ollama;
      try {
        // In container - prod
        await fetch("http://host.docker.internal:11434");

        ollama = new Ollama({
          baseUrl: "http://host.docker.internal:11434",
          model: ollamaModel,
          numCtx: contextSize,
          temperature: 0.1,
        });
      } catch (error) {
        // Local - dev
        ollama = new Ollama({
          model: ollamaModel,
          numCtx: contextSize,
          temperature: 0.1,
        });
      }

      // summary buffer memory
      const memory = new ConversationSummaryBufferMemory({
        llm: ollama,
        maxTokenLimit: historySize,
      });

      // Update global chain value
      chain = new ConversationChain({
        llm: ollama,
        memory: memory,
      });

      console.log("Created chain");
    } catch (error) {
      // Print exception in the Extension window
      this._view?.webview.postMessage({
        type: "error-message",
        value: "Something went wrong creating a LangChain conversation chain - please refresh the plugin",
      });
    }
  }
}

function getNonce() {
  let text = "";
  const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}

/**
 * Function that returns similar snippets of code as the given snippet. These
 * results are all obtained from a ChromaDB containing vector embeddings of a code base
 *
 * @param snippet - The code snippet for which you want to find similar snippets
 * @param maxAmount - The amount of results to return
 * @returns A promise with the results from Chroma DB
 */
async function getSimilar(snippet: string, maxAmount: number): Promise<string> {
  let embeddings: OllamaEmbeddings;
  try {
    // In container - prod
    await fetch("http://host.docker.internal:11434");

    embeddings = new OllamaEmbeddings({
      model: "nomic-embed-text",
      requestOptions: { temperature: 0.1 },
      baseUrl: "http://host.docker.internal:11434",
    });
  } catch (error) {
    // Local - dev
    embeddings = new OllamaEmbeddings({
      model: "nomic-embed-text",
      requestOptions: { temperature: 0.1 },
    });
  }

  const snippetCharLength = snippet.length;

  // If length of the selected code is above a certain threshold, we split and search for smaller parts
  if (Math.abs(1000 - snippetCharLength) < Math.abs(400 - snippetCharLength)) {
    // First, get similar (large) pieces of code
    let db: Chroma;

    try {
      // In container - prod
      db = await Chroma.fromExistingCollection(embeddings, {
        url: "http://chroma-service:8000",
        collectionName: "generated_1000_250",
      });
    } catch (error) {
      // Local - dev
      db = await Chroma.fromExistingCollection(embeddings, {
        collectionName: "generated_1000_250",
      });
    }

    // Filter out any "similar" results from the same file the text was selected in to avoid using the actual selected piece of code as "similar"
    // https://docs.trychroma.com/usage-guide#using-where-filters
    const res = await db.similaritySearchWithScore(snippet, maxAmount, {
      source: { $ne: `${vscode.window.activeTextEditor?.document.fileName}` },
    });

    // Get similar large pieces of code
    // Split into smaller chunks
    // Get pieces of code similar to these small chunks
    // Get all results in a list
    // Filter out duplicates
    // Get top K

    let smallDb: Chroma;
    try {
      // In container - prod
      smallDb = await Chroma.fromExistingCollection(embeddings, {
        url: "http://chroma-service:8000",
        collectionName: "generated_400_100",
      });
    } catch (error) {
      // Local - dev
      smallDb = await Chroma.fromExistingCollection(embeddings, {
        collectionName: "generated_400_100",
      });
    }

    // Splitter to split the larger pieces of code into smaller chunks
    const javaSplitter = RecursiveCharacterTextSplitter.fromLanguage("java", {
      chunkSize: 400,
      chunkOverlap: 100,
    });

    // Store all small similar snippets
    let similarList: any[] = [];

    for (const largeSimilar of res) {
      const doc = new Document({ pageContent: largeSimilar[0].pageContent });
      const split = await javaSplitter.splitDocuments([doc]);

      // Find similar pieces of code to small chunk
      for (const smallChunk of split) {
        const smallRes = await smallDb.similaritySearchWithScore(smallChunk.pageContent, maxAmount, {
          source: {
            $ne: `${vscode.window.activeTextEditor?.document.fileName}`,
          },
        });

        similarList = similarList.concat(smallRes);
      }
    }

    // Remove duplicates
    let dedup: any[] = [];
    let dupSet = new Set();

    similarList.forEach((e) => {
      if (!dupSet.has(e[0].pageContent)) {
        dupSet.add(e[0].pageContent);
        dedup.push(e);
      }
    });

    // Sort by highest similarity score
    dedup.sort((a, b) => a[1] - b[1]);
    let curSimilarTokens = 0;
    let similarString = "";
    for (let i = 0; i < maxAmount; i++) {
      const similar = dedup[i];

      // @ts-ignore - metadata does exist on the object
      const filePath = similar[0].metadata.source.split("/");

      const sString = `file name: 
${filePath[filePath.length - 1]}

file content: 
${
  // @ts-ignore - pageContent does exist on the object
  similar[0].pageContent
}
  
`;

      // If the current amount of tokens used for similarstring + the amount of tokens needed for the next similar snippet, we break
      if (llmTokenizer.encode(sString).length + curSimilarTokens > maxSimilarTokens) {
        break;
      }
      similarString += sString;

      curSimilarTokens = llmTokenizer.encode(similarString).length;
    }

    console.log(similarList);
    console.log(dedup);

    return similarString;
  } else {
    let db: Chroma;
    try {
      // In container - prod
      db = await Chroma.fromExistingCollection(embeddings, {
        url: "http://chroma-service:8000",
        collectionName: "generated_400_100",
      });
    } catch (error) {
      // Local - dev
      db = await Chroma.fromExistingCollection(embeddings, {
        collectionName: "generated_400_100",
      });
    }

    // Filter out any "similar" results from the same file the text was selected in to avoid using the actual selected piece of code as "similar"
    // https://docs.trychroma.com/usage-guide#using-where-filters
    const res = await db.similaritySearchWithScore(snippet, maxAmount, {
      source: { $ne: `${vscode.window.activeTextEditor?.document.fileName}` },
    });

    console.log("Similar:");
    console.log(res);

    // Keep track of amount of tokens used
    let curSimilarTokens = 0;
    let similarString = "";
    for (const similar of res) {
      // @ts-ignore - metadata does exist on the object
      const filePath = similar[0].metadata.source.split("/");

      const sString = `file name: 
${filePath[filePath.length - 1]}

file content: 
${
  // @ts-ignore - pageContent does exist on the object
  similar[0].pageContent
}

`;

      // If the current amount of tokens used for similarstring + the amount of tokens needed for the next similar snippet, we break
      if (llmTokenizer.encode(sString).length + curSimilarTokens > maxSimilarTokens) {
        break;
      }
      similarString += sString;

      curSimilarTokens = llmTokenizer.encode(similarString).length;
    }

    return similarString;
  }

}

/**
 * Method that returns a call hierarchy for a given editor and text range
 *
 * @param editor - The editor that is currently opened
 * @param textRange - Range selected in the editor
 * @param selectedText - Text selected in the editor
 * @param maxDepth - Maximum depth of the call hierarchy
 * @param maxWidth - Maximum width of the call hierarchy we will explore
 * @returns
 */
async function getCallHierarchy(
  editor: vscode.TextEditor,
  textRange: vscode.Range,
  selectedText: string = "",
  maxDepth: number = 2,
  maxWidth: number = 10
): Promise<any[]> {
  const selectedStartOffset = editor.document.offsetAt(textRange.start);
  const selectedEndOffset = editor.document.offsetAt(textRange.end);

  // Get document symbols
  const documentSymbols: vscode.DocumentSymbol[] = await vscode.commands.executeCommand(
    "vscode.executeDocumentSymbolProvider",
    editor.document.uri
  );

  // TODO: better way of excluding methods?
  const blacklist = ["toString", "hashCode", "equals"];
  const ch: any[] = [];

  // Loop over all method in the file
  // documentSymbold have different 'kinds'
  // 7 is for variables, 5 is for methods, 4 is for class, 3 is for package?
  for (const documentSymbol of documentSymbols) {
    for (const c of documentSymbol.children) {
      const startOffset = editor.document.offsetAt(c.range.start);
      const endOffset = editor.document.offsetAt(c.range.end);

      // If this method starts or ends within the selected range:
      if (
        (selectedStartOffset >= startOffset && selectedStartOffset <= endOffset) ||
        (selectedEndOffset >= startOffset && selectedEndOffset <= endOffset)
      ) {
        const identifier = editor!.document.getText(c.selectionRange);

        let ct = false;

        // Ignore any blacklisted methods
        for (const blacklisted of blacklist) {
          if (identifier.includes(blacklisted)) {
            ct = true;
            break;
          }
        }

        if (ct) {
          continue;
        }

        // selectionRange starts at the identifier of the method
        const hierarchy: vscode.CallHierarchyItem[] = await vscode.commands.executeCommand(
          "vscode.prepareCallHierarchy",
          editor.document.uri,
          c.selectionRange.start
        );

        for (const callHierarchyItem of hierarchy) {
          ch.push(
            await prepareNestedCallHierarchy(
              callHierarchyItem,
              0,
              maxDepth,
              maxWidth,
              editor!.document.getText(c.selectionRange)
            )
          );
        }
      }
    }
  }

  return ch;
}

/**
 * Recursively expand the call hierarchy tree, until a certain depth has been reached
 *
 * @param callHierarchyItem - A node in the call hierarchy tree, which will be used to get it's callees
 * @param depth - Current depth of the call hierarchy
 * @param maxDepth - Maximum depth of the call hierarchy
 * @param maxWidth - Maximum width of the call hierarchy we will explore
 * @returns - A list of incoming calls
 */
async function prepareNestedCallHierarchy(
  callHierarchyItem: vscode.CallHierarchyItem,
  depth: number,
  maxDepth: number,
  maxWidth: number,
  startMethod: string = ""
): Promise<any[]> {
  const hierarchy: any[] = [];

  // If depth has been reached, append the 'execution termindated mesasge to the leaf of the tree'
  if (depth >= maxDepth) {
    return [];
  }

  const incomingCalls: vscode.CallHierarchyIncomingCall[] = await vscode.commands.executeCommand(
    "vscode.provideIncomingCalls",
    callHierarchyItem
  );

  for (const [i, inc] of incomingCalls.entries()) {
    if (i >= maxWidth) {
      break;
    }

    hierarchy.push({
      path: inc.from.uri.path,
      name: inc.from.name,
      from: await prepareNestedCallHierarchy(inc.from, depth + 1, maxDepth, maxWidth),
      to: startMethod,
    });
  }

  return hierarchy;
}

/**
 * This function gets all Java files in the current workspace folder and puts them in an object
 *
 * @param documentUri - Currently opened document's uri
 * @param chList - List representation of call hierarchy, used to filter out irrelevant results from repo structure (if onlyCHPaths set to true)
 * @returns - An object containing paths to all files
 */
async function getRepoStructure(documentUri: vscode.Uri, chList: any[] = []): Promise<string> {
  let allFiles: any[] = [];
  if (onlyCHPaths) {
    for (const x of chList) {
      for (const y of x) {
        allFiles.push(y);
      }
    }
  } else {
    allFiles = await vscode.workspace.findFiles("**/*.java");
  }

  const workspace = vscode.workspace.getWorkspaceFolder(documentUri);
  const structure = {};

  // Loop over all files and add them to an object
  for (const file of allFiles) {
    // Can't use any files that are not in the workspace
    if (!file.path.includes(workspace?.name!)) {
      continue;
    }

    const relFilePathArr = file.path.split(`${workspace?.name!}/`)[1].split("/");

    const nestedPath = relFilePathArr.slice(0, relFilePathArr.length - 1);
    const nestedVal = relFilePathArr[relFilePathArr.length - 1];

    const before = { ...structure };
    setNested(structure, nestedPath, nestedVal);

    // If the string representation of the repo structure is larger than the max amount of tokens fo repo structure, we break
    if (llmTokenizer.encode(JSON.stringify(structure, null, 2)).length > maxRepoTokens) {
      return JSON.stringify(before, null, 2);
    }
  }

  console.log(structure);

  return JSON.stringify(structure, null, 2);
}

/**
 * Helper function to set a value nested in an object
 *
 * @param obj - Object which we want to update
 * @param path - Path that we want to set (in array form)
 * @param val - Value we want to assign to the given path
 * @returns - Nothing, obj is updated
 */
function setNested(obj: RepoType, path: string[], val: string) {
  // If includeTests if false: removes all test files & folders
  if (!includeTests && path.includes("test")) {
    return 0;
  }

  if (path.length === 1) {
    if (obj[path[0]]) {
      const innerObj = { ...obj[path[0]] };

      // If set to true, we include all files
      if (includeFiles) {
        innerObj.files = [...innerObj.files!, val];
      }
      obj[path[0]] = innerObj;
    } else {
      if (includeFiles) {
        obj[path[0]] = { files: [val] } as RepoType;
      } else {
        obj[path[0]] = {} as RepoType;
      }
    }
    return;
  }

  if (obj[path[0]] === undefined) {
    obj[path[0]] = {};
  }

  return setNested(obj[path[0]], path.slice(1), val);
}

/**
 * Method that flattens a call hierarchy into a string representation used in the prompt for the LLM
 *
 * @param hierarchy - Call Hierarchy (list of objects)
 * @param to - Only used for the grammer-like representation, used to keep track what a certain method is making calls to
 * @param indent - Only used for the yaml-like representation, used to keep track of (nested) indenting
 * @param maxCHTokens - Maximum amount of tokens the CH part of the prompt can take
 * @param curCHTokens - Current amount of tokens the CH part of the prompt takes - Is a parameter for recursive calls
 * @param args - Custom parameter to control which representation should be used
 * @returns - String representation of call hierarchy
 */
function flattenHierarchy(
  hierarchy: callHierarchyRepresentation[] | callHierarchyRepresentation[][],
  to: string = "",
  indent: number = 0,
  maxTokens: number = maxCHTokens,
  curCHTokens: number = 0,
  args: string = "grammar-list"
) {
  // Yaml like indenting
  // a
  //     b
  //     c
  //         d

  if (args === "yaml") {
    let totalStr = "";
    for (let i = 0; i < hierarchy.length; i++) {
      const element = hierarchy[i];

      // Indent
      for (let j = 0; j < indent; j++) {
        totalStr += " ";
      }

      // @ts-ignore - name exists
      if (element.name !== undefined) {
        // @ts-ignore - name exists
        totalStr += `${element.name}\n`;
        // @ts-ignore - from exists
        totalStr += flattenHierarchy(element.from, "", indent + 4, maxTokens, curCHTokens, args);
      } else {
        totalStr += "START\n";
        // @ts-ignore - using element is fine
        totalStr += flattenHierarchy(element, "", indent + 4, maxTokens, curCHTokens, args);
      }
    }
    return totalStr;
  }

  // Depth first::
  // Grammar-like expression
  // a <- b
  // a <- c
  // c <- d
  if (args === "grammar") {
    if (to === "") {
      to = "START";
    }

    console.log("---");
    console.log(to);
    console.log(hierarchy);
    console.log("---");

    let totalStr = "";
    for (let i = 0; i < hierarchy.length; i++) {
      let element = hierarchy[i];
      console.log(element);
      // @ts-ignore - Check if element is a list or callHierarchyRepresentation
      if (element.name !== undefined) {
        // element = element as callHierarchyRepresentation;
        // @ts-ignore - name exists
        const CHString = `${to} <- ${element.name.split(":")[0]} (${
          // @ts-ignore - path exists
          element.path.split("/")[element.path.split("/").length - 1]
        })\n`;

        // This messes up the CH somehow
        // if (
        //   llmTokenizer.encode(CHString).length + curCHTokens >
        //   maxCHTokens
        // ) {
        //   break;
        // }

        totalStr += CHString;
        totalStr += flattenHierarchy(
          // @ts-ignore - from exists
          element.from,
          // @ts-ignore - name exists
          element.name,
          0,
          maxCHTokens,
          llmTokenizer.encode(totalStr).length,
          args
        );
      } else {
        // @ts-ignore - We know element is a list
        totalStr += flattenHierarchy(element, "", 0, maxCHTokens, curCHTokens, args);
      }
    }

    console.log(`Returning ${totalStr}`);
    return totalStr;
  }

  // Breadth-first grammar-like with lists:
  // a <- [b, c, d]
  // b <- [e, f]
  // c <- [g]

  if (args === "grammar-list") {
    let queue: any[] = [];
    let totalStr = "";

    // Get direct calls and add them to queue
    for (let i = 0; i < hierarchy.length; i++) {
      const element = hierarchy[i] as callHierarchyRepresentation[];

      for (let j = 0; j < element.length; j++) {
        const c = element[j];
        queue.push({ ...c });
      }
    }

    let prevElem: any = undefined;

    // Dequeue top element from list, add any nested calls to bottom of list
    while (queue.length > 0) {
      const firstElem = queue.shift();

      // Ignore anonymous functions
      if (firstElem.name.split(":")[0] === "{...}") {
        continue;
      }

      // Add all child elements to the back of the queue
      const from = firstElem.from;
      for (let i = 0; i < from.length; i++) {
        queue.push({ ...from[i], to: `${firstElem.name.split(":")[0]}` });
      }

      if (prevElem !== undefined && firstElem.to === prevElem.to) {
        totalStr += `, ${firstElem.name.split(":")[0]} (${
          firstElem.path.split("/")[firstElem.path.split("/").length - 1]
        })`;
      } else {
        prevElem = { ...firstElem };
        totalStr += `${totalStr.length > 0 ? "]" : ""}\n${firstElem.to} <- [${firstElem.name.split(":")[0]} (${
          firstElem.path.split("/")[firstElem.path.split("/").length - 1]
        })`;
      }

      // If the amount of tokens used for the string exceeds the maximum amount, we break
      if (llmTokenizer.encode(totalStr).length > maxCHTokens) {
        break;
      }
    }

    if (totalStr.length > 0) {
      totalStr += "]";
    }
    return totalStr;
  }

  //
  //
  //

  // Breadth first
  // Grammar-like expression
  // a <- b
  // a <- c
  // c <- d

  // let queue: any[] = [];
  // let totalStr = "";

  // // Get direct calls and add them to queue
  // for (let i = 0; i < hierarchy.length; i++) {
  //   const element = hierarchy[i] as callHierarchyRepresentation[];

  //   for (let j = 0; j < element.length; j++) {
  //     const c = element[j];
  //     queue.push({ ...c });
  //   }
  // }

  // // Dequeue top element from list, add any nested calls to bottom of list
  // while (queue.length > 0) {
  //   const firstElem = queue.shift();

  //   // Ignore anonymous functions
  //   if (firstElem.name.split(":")[0] === "{...}") {
  //     continue;
  //   }

  //   // Add all child elements to the back of the queue
  //   const from = firstElem.from;
  //   for (let i = 0; i < from.length; i++) {
  //     queue.push({ ...from[i], to: `${firstElem.name.split(":")[0]}` });
  //   }

  //   // Create string representation for this element
  //   const callStr = `${firstElem.to} <- ${firstElem.name.split(":")[0]} (${
  //     firstElem.path.split("/")[firstElem.path.split("/").length - 1]
  //   })\n`;

  //   //
  //   curCHTokens += llmTokenizer.encode(callStr).length;
  //   if (curCHTokens > maxCHTokens) {
  //     break;
  //   }

  //   // Add to total string
  //   totalStr += callStr;
  // }

  // return totalStr;

  //
  //
  //

  // Nested representation
  // [a <- [b, c <- d]]

  if (args === "nested-list") {
    if (hierarchy.length === 0) {
      return "";
    }

    let str = "";
    for (let i = 0; i < hierarchy.length; i++) {
      const element = hierarchy[i];

      // @ts-ignore - name exists
      if (element.name !== undefined) {
        // @ts-ignore - from exists
        const children = flattenHierarchy(element.from, to, indent, maxTokens, curCHTokens, args);
        const childStr =
          children === ""
            ? // @ts-ignore - name exists
              `${element.name.split(":")[0]} (${element.path.split("/")[element.path.split("/").length - 1]})`
            : // @ts-ignore - name exists
              `${element.name.split(":")[0]} (${
                // @ts-ignore - path exists
                element.path.split("/")[element.path.split("/").length - 1]
              }) <- [${children}]`;
        str += childStr;

        if (i < hierarchy.length - 1) {
          str += ", ";
        }
      } else {
        // Nested list
        // @ts-ignore - using element is fine
        str += `[START] <- [${flattenHierarchy(element, to, indent, maxTokens, curCHTokens, args)}]`;
      }
    }

    return str;
  }

  // If any other param is specified, return empty string
  return "";
}

//
//
//
//

//
//
//
//

/**
 * Helper function to create a prompt the LLM can use
 *
 * @param userInput - The question the user asked the LLM
 * @param codeSnippet - The selected piece of code which the user is asking a question about
 * @param repoStructure - The string representation of the repository structure, created in getRepoStructure
 * @param callHierarchy - The string representation of the call hierarchy, created in getCallHierarchy
 * @param similarCode - The string representation of similar code snippets, created in getSimilar
 * @returns - A full prompt with all relevant information, ready to be sent to the LLM
 */
async function createPrompt(
  userInput: string,
  codeSnippet: string,
  repoStructure: string,
  callHierarchy: string,
  similarCode: string
) {
  let promptTemplate = "";

  // Creating a new prompt (when selecting a new piece of code)
  if (fullPrompt) {
    promptTemplate += `
You are an expert developer who is tasked with onboarding a new developer on your codebase, only answer the questions asked by the developer. Keep your answers concise

The following piece of code is selected by the user:
\`\`\`${codeSnippet}\`\`\`

Use the context below to generate your answer, but ignore any information that is not relevant to the question:

`;
    // Include file name
    if (includeFileName) {
      const before = llmTokenizer.encode(promptTemplate).length;
      promptTemplate += `This code is found in the file: ${vscode.window.activeTextEditor?.document.fileName}`;
      console.log(`File name token amount: ${llmTokenizer.encode(promptTemplate).length - before}`);
    }

    // Include similar code snippets
    if (useSimilar) {
      const before = llmTokenizer.encode(promptTemplate).length;
      promptTemplate += `Here are code snippets that are similar to the selected piece of code:
${similarCode}
      `;

      console.log(`Similar code snippets token amount: ${llmTokenizer.encode(promptTemplate).length - before}`);
    }

    // Include repository structure
    if (useRepoStructure) {
      const before = llmTokenizer.encode(promptTemplate).length;

      promptTemplate += `The repository structure below can also be used to give some more context in your answer to the question
${repoStructure}
      `;

      console.log(`Repo structure token amount: ${llmTokenizer.encode(promptTemplate).length - before}`);
    }

    // Include call hierarchy
    if (useHierarchy) {
      const before = llmTokenizer.encode(promptTemplate).length;

      promptTemplate += `Here is a call hierarchy of all methods that access the selected code snippet, in which elements are represented by {callee} <- [caller], where the list of callers directly call the callee:
${callHierarchy}
      `;
      // , represented by a nested list in which all elements of the list call the parent, indicated with parent <- [children] where [START] is the selected method
      // , represented in a YAML structure in which indented methods call their respective parent and the START node is the selected method:
      // , in which elements are represented by {callee} <- {caller}, where the caller directly calls the callee and the START node is the selected method:

      console.log(`Call hierarchy token amount: ${llmTokenizer.encode(promptTemplate).length - before}`);
    }
  }
  // CoT line at the bottom
  promptTemplate += `
Developer: ${userInput}

LLM: Let's think step by step.
`;

  // Self-consistency?
  //   promptTemplate += `
  // Generate five different answers and pick the answer that appears the most, print all these answers and pick the most common answer as the correct one
  //   `;

  console.log(`Total prompt tokens: ${llmTokenizer.encode(promptTemplate).length}`);

  fullPrompt = false;

  return promptTemplate;
}
