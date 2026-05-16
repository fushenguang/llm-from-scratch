import type { Tokenizer, TokenizerState } from "./Tokenizer.js";

interface CharTokenizerState extends TokenizerState {
  type: "char";
  vocab: Record<string, number>;
}

export class CharTokenizer implements Tokenizer {
  private charToIndex!: Map<string, number>;
  private indexToChar!: string[];

  public readonly vocabSize!: number;

  public constructor(trainingText: string) {
    const characters = [...new Set(Array.from(trainingText))].sort();
    this.initialize(characters);
  }

  public encode(text: string): number[] {
    return Array.from(text, (character) => {
      const token = this.charToIndex.get(character);

      if (token === undefined) {
        throw new Error(`Unknown character: ${JSON.stringify(character)}`);
      }

      return token;
    });
  }

  public decode(tokens: number[]): string {
    return tokens
      .map((token) => {
        const character = this.indexToChar[token];

        if (character === undefined) {
          throw new Error(`Unknown token: ${token}`);
        }

        return character;
      })
      .join("");
  }

  public serialize(): CharTokenizerState {
    return {
      type: "char",
      vocabSize: this.vocabSize,
      vocab: Object.fromEntries(
        this.indexToChar.map((character, index) => [character, index])
      )
    };
  }

  public static fromState(state: TokenizerState): CharTokenizer {
    if (state.type !== "char") {
      throw new Error(`Unsupported tokenizer state type: ${state.type}`);
    }

    if (state.vocab === undefined) {
      throw new Error("CharTokenizer state is missing vocab.");
    }

    const entries = Object.entries(state.vocab).sort((left, right) => left[1] - right[1]);

    if (entries.length !== state.vocabSize) {
      throw new Error(
        `CharTokenizer state vocab size ${entries.length} does not match ${state.vocabSize}.`
      );
    }

    entries.forEach(([, token], expectedToken) => {
      if (!Number.isInteger(token) || token < 0 || token !== expectedToken) {
        throw new Error("CharTokenizer state must use contiguous token indices starting at 0.");
      }
    });

    return CharTokenizer.create(entries.map(([character]) => character));
  }

  private initialize(characters: readonly string[]): void {
    const indexToChar = [...characters];
    const charToIndex = new Map(indexToChar.map((character, index) => [character, index]));

    this.indexToChar = indexToChar;
    this.charToIndex = charToIndex;
    Object.defineProperty(this, "vocabSize", {
      value: indexToChar.length,
      enumerable: true,
      configurable: false,
      writable: false
    });
  }

  private static create(characters: readonly string[]): CharTokenizer {
    const tokenizer = Object.create(CharTokenizer.prototype) as CharTokenizer;
    tokenizer.initialize(characters);
    return tokenizer;
  }
}
