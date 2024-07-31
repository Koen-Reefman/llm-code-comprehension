// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import { readdirSync, writeFileSync } from "fs";
import * as vscode from "vscode";
const { SidebarProvider } = require("./SidebarProvider");

let sideBar: typeof SidebarProvider;

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {
  // Register the Sidebar Panel
  sideBar = new SidebarProvider(context.extensionUri, Date.now());
  context.subscriptions.push(vscode.window.registerWebviewViewProvider("llm-view", sideBar));
}

// This method is called when your extension is deactivated
export async function deactivate() {}
