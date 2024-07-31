import * as vscode from "vscode";

interface RepoNested {
  [key: string]: RepoType;
}

export type RepoType = RepoNested & {
  files?: string[];
};

export type callHierarchyRepresentation = {
  path: string;
  name: string;
  from: callHierarchyRepresentation[];
};

export type codeSnippet = {
  filePath: string;
  textRange: vscode.Range;
  snippetText?: string;
};
