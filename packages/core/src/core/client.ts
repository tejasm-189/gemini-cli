/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  EmbedContentParameters,
  GenerateContentConfig,
  Part,
  SchemaUnion,
  PartListUnion,
  Content,
  Tool,
  GenerateContentResponse,
} from '@google/genai';
import { getFolderStructure } from '../utils/getFolderStructure.js';
import {
  Turn,
  ServerGeminiStreamEvent,
  GeminiEventType,
  ChatCompressionInfo,
} from './turn.js';
import { Config } from '../config/config.js';
import { getCoreSystemPrompt } from './prompts.js';
import { ReadManyFilesTool } from '../tools/read-many-files.js';
import { getResponseText } from '../utils/generateContentResponseUtilities.js';
import { checkNextSpeaker } from '../utils/nextSpeakerChecker.js';
import { reportError } from '../utils/errorReporting.js';
import { GeminiChat } from './geminiChat.js';
import { retryWithBackoff } from '../utils/retry.js';
import { getErrorMessage } from '../utils/errors.js';
import { tokenLimit } from './tokenLimits.js';
import {
  ContentGenerator,
  ContentGeneratorConfig,
  createContentGenerator,
} from './contentGenerator.js';
import { ProxyAgent, setGlobalDispatcher } from 'undici';
import { DEFAULT_GEMINI_FLASH_MODEL } from '../config/models.js';
import { AuthType } from './contentGenerator.js';

function isThinkingSupported(model: string) {
  if (model.startsWith('gemini-2.5')) return true;
  return false;
}

export class GeminiClient {
  private chat?: GeminiChat;
  private contentGenerator?: ContentGenerator;
  private model: string;
  private embeddingModel: string;
  private generateContentConfig: GenerateContentConfig = {
    temperature: 0,
    topP: 1,
  };
  private readonly MAX_TURNS = 100;

  constructor(private config: Config) {
    if (config.getProxy()) {
      setGlobalDispatcher(new ProxyAgent(config.getProxy() as string));
    }

    this.model = config.getModel();
    this.embeddingModel = config.getEmbeddingModel();
  }

  async initialize(contentGeneratorConfig: ContentGeneratorConfig) {
    this.contentGenerator = await createContentGenerator(
      contentGeneratorConfig,
    );
    this.chat = await this.startChat();
  }

  getContentGenerator(): ContentGenerator {
    if (!this.contentGenerator) {
      throw new Error('Content generator not initialized');
    }
    return this.contentGenerator;
  }

  async addHistory(content: Content) {
    this.getChat().addHistory(content);
  }

  getChat(): GeminiChat {
    if (!this.chat) {
      throw new Error('Chat not initialized');
    }
    return this.chat;
  }

  async getHistory(): Promise<Content[]> {
    return this.getChat().getHistory();
  }

  async setHistory(history: Content[]): Promise<void> {
    this.getChat().setHistory(history);
  }

  async resetChat(): Promise<void> {
    this.chat = await this.startChat();
    await this.chat;
  }

  private async getEnvironment(): Promise<Part[]> {
    const cwd = this.config.getWorkingDir();
    const today = new Date().toLocaleDateString(undefined, {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
    const platform = process.platform;
    const folderStructure = await getFolderStructure(cwd, {
      fileService: this.config.getFileService(),
    });
    const context = `
  This is the Gemini CLI. We are setting up the context for our chat.
  Today's date is ${today}.
  My operating system is: ${platform}
  I'm currently working in the directory: ${cwd}
  ${folderStructure}
          `.trim();

    const initialParts: Part[] = [{ text: context }];
    const toolRegistry = await this.config.getToolRegistry();

    // Add full file context if the flag is set
    if (this.config.getFullContext()) {
      try {
        const readManyFilesTool = toolRegistry.getTool(
          'read_many_files',
        ) as ReadManyFilesTool;
        if (readManyFilesTool) {
          // Read all files in the target directory
          const result = await readManyFilesTool.execute(
            {
              paths: ['**/*'], // Read everything recursively
              useDefaultExcludes: true, // Use default excludes
            },
            AbortSignal.timeout(30000),
          );
          if (result.llmContent) {
            initialParts.push({
              text: `\n--- Full File Context ---\n${result.llmContent}`,
            });
          } else {
            console.warn(
              'Full context requested, but read_many_files returned no content.',
            );
          }
        } else {
          console.warn(
            'Full context requested, but read_many_files tool not found.',
          );
        }
      } catch (error) {
        // Not using reportError here as it's a startup/config phase, not a chat/generation phase error.
        console.error('Error reading full file context:', error);
        initialParts.push({
          text: '\n--- Error reading full file context ---',
        });
      }
    }

    return initialParts;
  }

  private async startChat(extraHistory?: Content[]): Promise<GeminiChat> {
    const envParts = await this.getEnvironment();
    const toolRegistry = await this.config.getToolRegistry();
    const toolDeclarations = toolRegistry.getFunctionDeclarations();
    const tools: Tool[] = [{ functionDeclarations: toolDeclarations }];
    const initialHistory: Content[] = [
      {
        role: 'user',
        parts: envParts,
      },
      {
        role: 'model',
        parts: [{ text: 'Got it. Thanks for the context!' }],
      },
    ];
    const history = initialHistory.concat(extraHistory ?? []);
    try {
      const userMemory = this.config.getUserMemory();
      const systemInstruction = getCoreSystemPrompt(userMemory);
      const generateContentConfigWithThinking = isThinkingSupported(this.model)
        ? {
            ...this.generateContentConfig,
            thinkingConfig: {
              includeThoughts: true,
            },
          }
        : this.generateContentConfig;
      return new GeminiChat(
        this.config,
        this.getContentGenerator(),
        {
          systemInstruction,
          ...generateContentConfigWithThinking,
          tools,
        },
        history,
      );
    } catch (error) {
      await reportError(
        error,
        'Error initializing Gemini chat session.',
        history,
        'startChat',
      );
      throw new Error(`Failed to initialize chat: ${getErrorMessage(error)}`);
    }
  }

  async *sendMessageStream(
    request: PartListUnion,
    signal: AbortSignal,
    turns: number = this.MAX_TURNS,
  ): AsyncGenerator<ServerGeminiStreamEvent, Turn> {
    if (!turns) {
      return new Turn(this.getChat());
    }

    const compressed = await this.tryCompressChat();
    if (compressed) {
      yield { type: GeminiEventType.ChatCompressed, value: compressed };
    }
    const turn = new Turn(this.getChat());
    // Pass the signal to the turn's run method, which passes it to chat.sendMessageStream
    const resultStream = turn.run(request, signal);
    for await (const event of resultStream) {
      yield event;
    }
    if (!turn.pendingToolCalls.length && signal && !signal.aborted) {
      const nextSpeakerCheck = await checkNextSpeaker(
        this.getChat(),
        this,
        signal,
      );
      if (nextSpeakerCheck?.next_speaker === 'model') {
        const nextRequest = [{ text: 'Please continue.' }];
        // This recursive call's events will be yielded out, but the final
        // turn object will be from the top-level call.
        yield* this.sendMessageStream(nextRequest, signal, turns - 1);
      }
    }
    return turn;
  }

  async generateJson(
    contents: Content[],
    schema: SchemaUnion,
    abortSignal: AbortSignal,
    model: string = DEFAULT_GEMINI_FLASH_MODEL,
    config: GenerateContentConfig = {},
  ): Promise<Record<string, unknown>> {
    try {
      const userMemory = this.config.getUserMemory();
      const systemInstruction = getCoreSystemPrompt(userMemory);
      const requestConfig = {
        abortSignal,
        ...this.generateContentConfig,
        ...config,
      };

      const apiCall = () =>
        this.getContentGenerator().generateContent({
          model,
          config: {
            ...requestConfig,
            systemInstruction,
            responseSchema: schema,
            responseMimeType: 'application/json',
          },
          contents,
        });

      const result = await retryWithBackoff(apiCall, {
        onPersistent429: async (authType?: string) =>
          await this.handleFlashFallback(authType),
        authType: this.config.getContentGeneratorConfig()?.authType,
      });

      const text = getResponseText(result);
      if (!text) {
        const error = new Error(
          'API returned an empty response for generateJson.',
        );
        await reportError(
          error,
          'Error in generateJson: API returned an empty response.',
          contents,
          'generateJson-empty-response',
        );
        throw error;
      }
      try {
        return JSON.parse(text);
      } catch (parseError) {
        await reportError(
          parseError,
          'Failed to parse JSON response from generateJson.',
          {
            responseTextFailedToParse: text,
            originalRequestContents: contents,
          },
          'generateJson-parse',
        );
        throw new Error(
          `Failed to parse API response as JSON: ${getErrorMessage(parseError)}`,
        );
      }
    } catch (error) {
      if (abortSignal.aborted) {
        throw error;
      }

      // Avoid double reporting for the empty response case handled above
      if (
        error instanceof Error &&
        error.message === 'API returned an empty response for generateJson.'
      ) {
        throw error;
      }

      await reportError(
        error,
        'Error generating JSON content via API.',
        contents,
        'generateJson-api',
      );
      throw new Error(
        `Failed to generate JSON content: ${getErrorMessage(error)}`,
      );
    }
  }

  async generateContent(
    contents: Content[],
    generationConfig: GenerateContentConfig,
    abortSignal: AbortSignal,
  ): Promise<GenerateContentResponse> {
    const modelToUse = this.model;
    const configToUse: GenerateContentConfig = {
      ...this.generateContentConfig,
      ...generationConfig,
    };

    try {
      const userMemory = this.config.getUserMemory();
      const systemInstruction = getCoreSystemPrompt(userMemory);

      const requestConfig = {
        abortSignal,
        ...configToUse,
        systemInstruction,
      };

      const apiCall = () =>
        this.getContentGenerator().generateContent({
          model: modelToUse,
          config: requestConfig,
          contents,
        });

      const result = await retryWithBackoff(apiCall, {
        onPersistent429: async (authType?: string) =>
          await this.handleFlashFallback(authType),
        authType: this.config.getContentGeneratorConfig()?.authType,
      });
      return result;
    } catch (error: unknown) {
      if (abortSignal.aborted) {
        throw error;
      }

      await reportError(
        error,
        `Error generating content via API with model ${modelToUse}.`,
        {
          requestContents: contents,
          requestConfig: configToUse,
        },
        'generateContent-api',
      );
      throw new Error(
        `Failed to generate content with model ${modelToUse}: ${getErrorMessage(error)}`,
      );
    }
  }

  async generateEmbedding(texts: string[]): Promise<number[][]> {
    if (!texts || texts.length === 0) {
      return [];
    }
    const embedModelParams: EmbedContentParameters = {
      model: this.embeddingModel,
      contents: texts,
    };

    const embedContentResponse =
      await this.getContentGenerator().embedContent(embedModelParams);
    if (
      !embedContentResponse.embeddings ||
      embedContentResponse.embeddings.length === 0
    ) {
      throw new Error('No embeddings found in API response.');
    }

    if (embedContentResponse.embeddings.length !== texts.length) {
      throw new Error(
        `API returned a mismatched number of embeddings. Expected ${texts.length}, got ${embedContentResponse.embeddings.length}.`,
      );
    }

    return embedContentResponse.embeddings.map((embedding, index) => {
      const values = embedding.values;
      if (!values || values.length === 0) {
        throw new Error(
          `API returned an empty embedding for input text at index ${index}: "${texts[index]}"`,
        );
      }
      return values;
    });
  }

  async tryCompressChat(
    force: boolean = false,
  ): Promise<ChatCompressionInfo | null> {
    const history = this.getChat().getHistory(true); // Get curated history

    // Regardless of `force`, don't do anything if the history is empty.
    if (history.length === 0) {
      return null;
    }

    const { totalTokens: originalTokenCount } =
      await this.getContentGenerator().countTokens({
        model: this.model,
        contents: history,
      });

    // If not forced, check if we should compress based on context size.
    if (!force) {
      if (originalTokenCount === undefined) {
        // If token count is undefined, we can't determine if we need to compress.
        console.warn(
          `Could not determine token count for model ${this.model}. Skipping compression check.`,
        );
        return null;
      }
      const tokenCount = originalTokenCount; // Now guaranteed to be a number

      const limit = tokenLimit(this.model);
      if (!limit) {
        // If no limit is defined for the model, we can't compress.
        console.warn(
          `No token limit defined for model ${this.model}. Skipping compression check.`,
        );
        return null;
      }

      if (tokenCount < 0.95 * limit) {
        return null;
      }
    }

    const summarizationRequestMessage = {
      text: 'Summarize our conversation up to this point. The summary should be a concise yet comprehensive overview of all key topics, questions, answers, and important details discussed. This summary will replace the current chat history to conserve tokens, so it must capture everything essential to understand the context and continue our conversation effectively as if no information was lost.',
    };
    const response = await this.getChat().sendMessage({
      message: summarizationRequestMessage,
    });
    const newHistory = [
      {
        role: 'user',
        parts: [summarizationRequestMessage],
      },
      {
        role: 'model',
        parts: [{ text: response.text }],
      },
    ];
    this.chat = await this.startChat(newHistory);
    const newTokenCount = (
      await this.getContentGenerator().countTokens({
        model: this.model,
        contents: newHistory,
      })
    ).totalTokens;

    return originalTokenCount && newTokenCount
      ? {
          originalTokenCount,
          newTokenCount,
        }
      : null;
  }

  /**
   * Handles fallback to Flash model when persistent 429 errors occur for OAuth users.
   * Uses a fallback handler if provided by the config, otherwise returns null.
   */
  private async handleFlashFallback(authType?: string): Promise<string | null> {
    // Only handle fallback for OAuth users
    if (authType !== AuthType.LOGIN_WITH_GOOGLE_PERSONAL) {
      return null;
    }

    const currentModel = this.model;
    const fallbackModel = DEFAULT_GEMINI_FLASH_MODEL;

    // Don't fallback if already using Flash model/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  Content,
  Models,
  GenerateContentConfig,
  Part,
  GenerateContentResponse,
} from '@google/genai';
import { GeminiChat } from './geminiChat.js';
import { Config } from '../config/config.js';
import { setSimulate429 } from '../utils/testUtils.js';

// Mocks
const mockModelsModule = {
  generateContent: vi.fn(),
  generateContentStream: vi.fn(),
  countTokens: vi.fn(),
  embedContent: vi.fn(),
  batchEmbedContents: vi.fn(),
} as unknown as Models;

describe('GeminiChat', () => {
  let chat: GeminiChat;
  let mockConfig: Config;
  const config: GenerateContentConfig = {};

  beforeEach(() => {
    vi.clearAllMocks();
    mockConfig = {
      getSessionId: () => 'test-session-id',
      getTelemetryLogPromptsEnabled: () => true,
      getUsageStatisticsEnabled: () => true,
      getDebugMode: () => false,
      getContentGeneratorConfig: () => ({
        authType: 'oauth-personal',
        model: 'test-model',
      }),
      getModel: vi.fn().mockReturnValue('gemini-pro'),
      setModel: vi.fn(),
      flashFallbackHandler: undefined,
    } as unknown as Config;

    // Disable 429 simulation for tests
    setSimulate429(false);
    // Reset history for each test by creating a new instance
    chat = new GeminiChat(mockConfig, mockModelsModule, config, []);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.resetAllMocks();
  });

  describe('sendMessage', () => {
    it('should call generateContent with the correct parameters', async () => {
      const response = {
        candidates: [
          {
            content: {
              parts: [{ text: 'response' }],
              role: 'model',
            },
            finishReason: 'STOP',
            index: 0,
            safetyRatings: [],
          },
        ],
        text: () => 'response',
      } as unknown as GenerateContentResponse;
      vi.mocked(mockModelsModule.generateContent).mockResolvedValue(response);

      await chat.sendMessage({ message: 'hello' });

      expect(mockModelsModule.generateContent).toHaveBeenCalledWith({
        model: 'gemini-pro',
        contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
        config: { signal: undefined },
      });
    });

    it('should call generateContent with the abortSignal if provided', async () => {
      const response = {
        candidates: [
          {
            content: {
              parts: [{ text: 'response' }],
              role: 'model',
            },
            finishReason: 'STOP',
            index: 0,
            safetyRatings: [],
          },
        ],
        text: () => 'response',
      } as unknown as GenerateContentResponse;
      vi.mocked(mockModelsModule.generateContent).mockResolvedValue(response);
      const abortController = new AbortController();
      const signal = abortController.signal;

      await chat.sendMessage({ message: 'hello' }, signal);

      expect(mockModelsModule.generateContent).toHaveBeenCalledWith({
        model: 'gemini-pro',
        contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
        config: { signal },
      });
    });

    it('should throw an error if generateContent is aborted', async () => {
      const abortController = new AbortController();
      const signal = abortController.signal;
      const error = new Error('Aborted');
      error.name = 'AbortError';
      vi.mocked(mockModelsModule.generateContent).mockRejectedValue(error);

      abortController.abort(); // Abort the signal

      await expect(
        chat.sendMessage({ message: 'hello' }, signal),
      ).rejects.toThrow('Aborted');

      expect(mockModelsModule.generateContent).toHaveBeenCalledWith({
        model: 'gemini-pro',
        contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
        config: { signal },
      });
    });
  });

  describe('sendMessageStream', () => {
    it('should call generateContentStream with the correct parameters', async () => {
      const response = (async function* () {
        yield {
          candidates: [
            {
              content: {
                parts: [{ text: 'response' }],
                role: 'model',
              },
              finishReason: 'STOP',
              index: 0,
              safetyRatings: [],
            },
          ],
          text: () => 'response',
        } as unknown as GenerateContentResponse;
      })();
      vi.mocked(mockModelsModule.generateContentStream).mockResolvedValue(
        response,
      );

      await chat.sendMessageStream({ message: 'hello' });

      expect(mockModelsModule.generateContentStream).toHaveBeenCalledWith({
        model: 'gemini-pro',
        contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
        config: { signal: undefined },
      });
    });

    it('should call generateContentStream with the abortSignal if provided', async () => {
      const response = (async function* () {
        yield {
          candidates: [
            {
              content: {
                parts: [{ text: 'response' }],
                role: 'model',
              },
              finishReason: 'STOP',
              index: 0,
              safetyRatings: [],
            },
          ],
          text: () => 'response',
        } as unknown as GenerateContentResponse;
      })();
      vi.mocked(mockModelsModule.generateContentStream).mockResolvedValue(
        response,
      );
      const abortController = new AbortController();
      const signal = abortController.signal;

      await chat.sendMessageStream({ message: 'hello' }, signal);

      expect(mockModelsModule.generateContentStream).toHaveBeenCalledWith({
        model: 'gemini-pro',
        contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
        config: { signal },
      });
    });

    it('should throw an error if generateContentStream is aborted', async () => {
      const abortController = new AbortController();
      const signal = abortController.signal;
      const error = new Error('Aborted');
      error.name = 'AbortError';

      vi.mocked(mockModelsModule.generateContentStream).mockImplementation(
        async function* () {
          await new Promise((resolve) => setTimeout(resolve, 0)); // Ensure microtask queue is processed
          if (signal.aborted) {
            throw error;
          }
          // Should not reach here if aborted before call
          yield { text: () => 'should not yield' } as GenerateContentResponse;
        },
      );

      abortController.abort(); // Abort the signal

      try {
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        for await (const _ of await chat.sendMessageStream(
          { message: 'hello' },
          signal,
        )) {
          // This loop should not run if an error is thrown.
        }
        // If the loop completes without throwing, the test fails.
        expect.fail('Stream did not throw an error as expected.');
      } catch (e: any) {
        expect(e.message).toBe('Aborted');
      }

      expect(mockModelsModule.generateContentStream).toHaveBeenCalledWith({
        model: 'gemini-pro',
        contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
        config: { signal },
      });
    });
  });

  describe('recordHistory', () => {
    const userInput: Content = {
      role: 'user',
      parts: [{ text: 'User input' }],
    };

    it('should add user input and a single model output to history', () => {
      const modelOutput: Content[] = [
        { role: 'model', parts: [{ text: 'Model output' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutput);
      const history = chat.getHistory();
      expect(history).toEqual([userInput, modelOutput[0]]);
    });

    it('should consolidate adjacent model outputs', () => {
      const modelOutputParts: Content[] = [
        { role: 'model', parts: [{ text: 'Model part 1' }] },
        { role: 'model', parts: [{ text: 'Model part 2' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutputParts);
      const history = chat.getHistory();
      expect(history.length).toBe(2);
      expect(history[0]).toEqual(userInput);
      expect(history[1].role).toBe('model');
      expect(history[1].parts).toEqual([{ text: 'Model part 1Model part 2' }]);
    });

    it('should handle a mix of user and model roles in outputContents (though unusual)', () => {
      const mixedOutput: Content[] = [
        { role: 'model', parts: [{ text: 'Model 1' }] },
        { role: 'user', parts: [{ text: 'Unexpected User' }] }, // This should be pushed as is
        { role: 'model', parts: [{ text: 'Model 2' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, mixedOutput);
      const history = chat.getHistory();
      expect(history.length).toBe(4); // user, model1, user_unexpected, model2
      expect(history[0]).toEqual(userInput);
      expect(history[1]).toEqual(mixedOutput[0]);
      expect(history[2]).toEqual(mixedOutput[1]);
      expect(history[3]).toEqual(mixedOutput[2]);
    });

    it('should consolidate multiple adjacent model outputs correctly', () => {
      const modelOutputParts: Content[] = [
        { role: 'model', parts: [{ text: 'M1' }] },
        { role: 'model', parts: [{ text: 'M2' }] },
        { role: 'model', parts: [{ text: 'M3' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutputParts);
      const history = chat.getHistory();
      expect(history.length).toBe(2);
      expect(history[1].parts).toEqual([{ text: 'M1M2M3' }]);
    });

    it('should not consolidate if roles are different between model outputs', () => {
      const modelOutputParts: Content[] = [
        { role: 'model', parts: [{ text: 'M1' }] },
        { role: 'user', parts: [{ text: 'Interjecting User' }] },
        { role: 'model', parts: [{ text: 'M2' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutputParts);
      const history = chat.getHistory();
      expect(history.length).toBe(4); // user, M1, Interjecting User, M2
      expect(history[1].parts).toEqual([{ text: 'M1' }]);
      expect(history[3].parts).toEqual([{ text: 'M2' }]);
    });

    it('should merge with last history entry if it is also a model output', () => {
      // @ts-expect-error Accessing private property for test setup
      chat.history = [
        userInput,
        { role: 'model', parts: [{ text: 'Initial Model Output' }] },
      ]; // Prime the history

      const newModelOutput: Content[] = [
        { role: 'model', parts: [{ text: 'New Model Part 1' }] },
        { role: 'model', parts: [{ text: 'New Model Part 2' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, newModelOutput); // userInput here is for the *next* turn, but history is already primed

      // Reset and set up a more realistic scenario for merging with existing history
      chat = new GeminiChat(mockConfig, mockModelsModule, config, []);
      const firstUserInput: Content = {
        role: 'user',
        parts: [{ text: 'First user input' }],
      };
      const firstModelOutput: Content[] = [
        { role: 'model', parts: [{ text: 'First model response' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(firstUserInput, firstModelOutput);

      const secondUserInput: Content = {
        role: 'user',
        parts: [{ text: 'Second user input' }],
      };
      const secondModelOutput: Content[] = [
        { role: 'model', parts: [{ text: 'Second model response part 1' }] },
        { role: 'model', parts: [{ text: 'Second model response part 2' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(secondUserInput, secondModelOutput);

      const finalHistory = chat.getHistory();
      expect(finalHistory.length).toBe(4); // user1, model1, user2, model2(consolidated)
      expect(finalHistory[0]).toEqual(firstUserInput);
      expect(finalHistory[1]).toEqual(firstModelOutput[0]);
      expect(finalHistory[2]).toEqual(secondUserInput);
      expect(finalHistory[3].role).toBe('model');
      expect(finalHistory[3].parts).toEqual([
        { text: 'Second model response part 1Second model response part 2' },
      ]);
    });

    it('should correctly merge consolidated new output with existing model history', () => {
      // Setup: history ends with a model turn
      const initialUser: Content = {
        role: 'user',
        parts: [{ text: 'Initial user query' }],
      };
      const initialModel: Content = {
        role: 'model',
        parts: [{ text: 'Initial model answer.' }],
      };
      chat = new GeminiChat(mockConfig, mockModelsModule, config, [
        initialUser,
        initialModel,
      ]);

      // New interaction
      const currentUserInput: Content = {
        role: 'user',
        parts: [{ text: 'Follow-up question' }],
      };
      const newModelParts: Content[] = [
        { role: 'model', parts: [{ text: 'Part A of new answer.' }] },
        { role: 'model', parts: [{ text: 'Part B of new answer.' }] },
      ];

      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(currentUserInput, newModelParts);
      const history = chat.getHistory();

      // Expected: initialUser, initialModel, currentUserInput, consolidatedNewModelParts
      expect(history.length).toBe(4);
      expect(history[0]).toEqual(initialUser);
      expect(history[1]).toEqual(initialModel);
      expect(history[2]).toEqual(currentUserInput);
      expect(history[3].role).toBe('model');
      expect(history[3].parts).toEqual([
        { text: 'Part A of new answer.Part B of new answer.' },
      ]);
    });

    it('should handle empty modelOutput array', () => {
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, []);
      const history = chat.getHistory();
      // If modelOutput is empty, it might push a default empty model part depending on isFunctionResponse
      // Assuming isFunctionResponse(userInput) is false for this simple text input
      expect(history.length).toBe(2);
      expect(history[0]).toEqual(userInput);
      expect(history[1].role).toBe('model');
      expect(history[1].parts).toEqual([]);
    });

    it('should handle aggregating modelOutput', () => {
      const modelOutputUndefinedParts: Content[] = [
        { role: 'model', parts: [{ text: 'First model part' }] },
        { role: 'model', parts: [{ text: 'Second model part' }] },
        { role: 'model', parts: undefined as unknown as Part[] }, // Test undefined parts
        { role: 'model', parts: [{ text: 'Third model part' }] },
        { role: 'model', parts: [] }, // Test empty parts array
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutputUndefinedParts);
      const history = chat.getHistory();
      expect(history.length).toBe(5);
      expect(history[0]).toEqual(userInput);
      expect(history[1].role).toBe('model');
      expect(history[1].parts).toEqual([
        { text: 'First model partSecond model part' },
      ]);
      expect(history[2].role).toBe('model');
      expect(history[2].parts).toBeUndefined();
      expect(history[3].role).toBe('model');
      expect(history[3].parts).toEqual([{ text: 'Third model part' }]);
      expect(history[4].role).toBe('model');
      expect(history[4].parts).toEqual([]);
    });

    it('should handle modelOutput with parts being undefined or empty (if they pass initial every check)', () => {
      const modelOutputUndefinedParts: Content[] = [
        { role: 'model', parts: [{ text: 'Text part' }] },
        { role: 'model', parts: undefined as unknown as Part[] }, // Test undefined parts
        { role: 'model', parts: [] }, // Test empty parts array
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutputUndefinedParts);
      const history = chat.getHistory();
      expect(history.length).toBe(4); // userInput, model1 (text), model2 (undefined parts), model3 (empty parts)
      expect(history[0]).toEqual(userInput);
      expect(history[1].role).toBe('model');
      expect(history[1].parts).toEqual([{ text: 'Text part' }]);
      expect(history[2].role).toBe('model');
      expect(history[2].parts).toBeUndefined();
      expect(history[3].role).toBe('model');
      expect(history[3].parts).toEqual([]);
    });

    it('should correctly handle automaticFunctionCallingHistory', () => {
      const afcHistory: Content[] = [
        { role: 'user', parts: [{ text: 'AFC User' }] },
        { role: 'model', parts: [{ text: 'AFC Model' }] },
      ];
      const modelOutput: Content[] = [
        { role: 'model', parts: [{ text: 'Regular Model Output' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutput, afcHistory);
      const history = chat.getHistory();
      expect(history.length).toBe(3);
      expect(history[0]).toEqual(afcHistory[0]);
      expect(history[1]).toEqual(afcHistory[1]);
      expect(history[2]).toEqual(modelOutput[0]);
    });

    it('should add userInput if AFC history is present but empty', () => {
      const modelOutput: Content[] = [
        { role: 'model', parts: [{ text: 'Model Output' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutput, []); // Empty AFC history
      const history = chat.getHistory();
      expect(history.length).toBe(2);
      expect(history[0]).toEqual(userInput);
      expect(history[1]).toEqual(modelOutput[0]);
    });

    it('should skip "thought" content from modelOutput', () => {
      const modelOutputWithThought: Content[] = [
        { role: 'model', parts: [{ thought: true }, { text: 'Visible text' }] },
        { role: 'model', parts: [{ text: 'Another visible text' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutputWithThought);
      const history = chat.getHistory();
      expect(history.length).toBe(2); // User input + consolidated model output
      expect(history[0]).toEqual(userInput);
      expect(history[1].role).toBe('model');
      // The 'thought' part is skipped, 'Another visible text' becomes the first part.
      expect(history[1].parts).toEqual([{ text: 'Another visible text' }]);
    });

    it('should skip "thought" content even if it is the only content', () => {
      const modelOutputOnlyThought: Content[] = [
        { role: 'model', parts: [{ thought: true }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutputOnlyThought);
      const history = chat.getHistory();
      expect(history.length).toBe(1); // User input + default empty model part
      expect(history[0]).toEqual(userInput);
    });

    it('should correctly consolidate text parts when a thought part is in between', () => {
      const modelOutputMixed: Content[] = [
        { role: 'model', parts: [{ text: 'Part 1.' }] },
        {
          role: 'model',
          parts: [{ thought: true }, { text: 'Should be skipped' }],
        },
        { role: 'model', parts: [{ text: 'Part 2.' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutputMixed);
      const history = chat.getHistory();
      expect(history.length).toBe(2);
      expect(history[0]).toEqual(userInput);
      expect(history[1].role).toBe('model');
      expect(history[1].parts).toEqual([{ text: 'Part 1.Part 2.' }]);
    });

    it('should handle multiple thought parts correctly', () => {
      const modelOutputMultipleThoughts: Content[] = [
        { role: 'model', parts: [{ thought: true }] },
        { role: 'model', parts: [{ text: 'Visible 1' }] },
        { role: 'model', parts: [{ thought: true }] },
        { role: 'model', parts: [{ text: 'Visible 2' }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutputMultipleThoughts);
      const history = chat.getHistory();
      expect(history.length).toBe(2);
      expect(history[0]).toEqual(userInput);
      expect(history[1].role).toBe('model');
      expect(history[1].parts).toEqual([{ text: 'Visible 1Visible 2' }]);
    });

    it('should handle thought part at the end of outputContents', () => {
      const modelOutputThoughtAtEnd: Content[] = [
        { role: 'model', parts: [{ text: 'Visible text' }] },
        { role: 'model', parts: [{ thought: true }] },
      ];
      // @ts-expect-error Accessing private method for testing purposes
      chat.recordHistory(userInput, modelOutputThoughtAtEnd);
      const history = chat.getHistory();
      expect(history.length).toBe(2);
      expect(history[0]).toEqual(userInput);
      expect(history[1].role).toBe('model');
      expect(history[1].parts).toEqual([{ text: 'Visible text' }]);
    });
  });

  describe('addHistory', () => {
    it('should add a new content item to the history', () => {
      const newContent: Content = {
        role: 'user',
        parts: [{ text: 'A new message' }],
      };
      chat.addHistory(newContent);
      const history = chat.getHistory();
      expect(history.length).toBe(1);
      expect(history[0]).toEqual(newContent);
    });

    it('should add multiple items correctly', () => {
      const content1: Content = {
        role: 'user',
        parts: [{ text: 'Message 1' }],
      };
      const content2: Content = {
        role: 'model',
        parts: [{ text: 'Message 2' }],
      };
      chat.addHistory(content1);
      chat.addHistory(content2);
      const history = chat.getHistory();
      expect(history.length).toBe(2);
      expect(history[0]).toEqual(content1);
      expect(history[1]).toEqual(content2);
    });
  });
});

    if (currentModel === fallbackModel) {
      return null;
    }

    // Check if config has a fallback handler (set by CLI package)
    const fallbackHandler = this.config.flashFallbackHandler;
    if (typeof fallbackHandler === 'function') {
      try {
        const accepted = await fallbackHandler(currentModel, fallbackModel);
        if (accepted) {
          this.config.setModel(fallbackModel);
          this.model = fallbackModel;
          return fallbackModel;
        }
      } catch (error) {
        console.warn('Flash fallback handler failed:', error);
      }
    }

    return null;
  }
}
