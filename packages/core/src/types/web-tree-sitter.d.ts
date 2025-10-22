/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Minimal type definitions for the web-tree-sitter package, covering the
 * surface that the core shell utilities rely on at runtime.
 */
declare module 'web-tree-sitter' {
  export interface ParserInitOptions {
    wasmBinary?: ArrayBuffer | Uint8Array;
  }

  export class Language {
    static load(input: ArrayBuffer | Uint8Array | string): Promise<Language>;
  }

  export interface Tree {
    rootNode: Node;
  }

  export interface Node {
    type: string;
    text: string;
    startIndex: number;
    endIndex: number;
    namedChildCount: number;
    hasError: boolean;
    child(index: number): Node | null;
    childForFieldName(name: string): Node | null;
    namedChild(index: number): Node | null;
  }

  export class Parser {
    static init(options?: ParserInitOptions): Promise<void>;
    setLanguage(language: Language): void;
    parse(source: string): Tree;
  }
}
