/**
 * Text-mode intent routing (Issue #429).
 *
 * A message sent with the default type "text" is routed by detectIntent(): to the image, video,
 * audio or code path when it asks for one, and to plain chat otherwise. It used to look for its
 * trigger words as substrings, so "withdraw" was a request to draw, "essay" a request to say
 * something, "invoice" a request for a voice and "functionality" a request for code, and the bare
 * words "say" and "function" sent ordinary sentences ("say hello", "the function of a database
 * index") to audio and to code.
 *
 * A trigger is now a whole word, with the endings of its own word. "say" asks for audio only
 * together with "aloud" or "out loud", and "function" and "program" name code only when one is
 * being asked for. The genuine requests below were routed the same way before the change, and
 * must keep being routed that way.
 */

const { describe, it, beforeEach, afterEach } = require("node:test");
const assert = require("node:assert/strict");

const bridge = require("../src/services/pythonBridge");
const { detectIntent, processMessage } = require("../src/services/model");

// Each row is [what the user typed, where it goes]. The message is part of the test name, so a
// failure says which one it was.
function routes(rows) {
  for (const [message, intent] of rows) {
    it(`${JSON.stringify(message)} goes to ${intent}`, () => {
      assert.equal(detectIntent(message), intent);
    });
  }
}

describe("the messages in the issue", () => {
  routes([
    ["I want to withdraw money", "text"], // "withdraw" contains "draw"
    ["say hello", "text"],
    ["Explain the function of a database index", "text"],
  ]);
});

describe("a trigger inside another word is not a trigger", () => {
  routes([
    ["Please show me the withdrawal limits for my account", "text"],
    ["What is the main drawback of this plan?", "text"],
    ["Where is the kitchen drawer?", "text"],
    ["Write an essay about climate change", "text"], // "essay " contains "say "
    ["Show me Gordon Ramsay recipes", "text"],
    ["Gordon Ramsay shouted it out loud", "text"], // "ramsay " ends in "say ", and he said it out loud
    ["That is only hearsay and rumor", "text"],
    ["Please draft an invoice for the client", "text"], // "invoice" contains "voice"
    ["Why can inanimate objects not think?", "text"], // "inanimate" contains "animate"
    ["Explain the functionality of a router", "text"],
    ["The malfunction left a dysfunction in the system", "text"],
    ["Why is this trait method still unimplemented?", "text"], // "unimplemented" contains "implement"
    ["What does a programmer earn?", "text"],
    ["I enjoy programming on weekends", "text"],
    ["Who is the best speaker at the conference?", "text"],
    ["What is the postcode for London?", "text"], // "postcode for" contains "code for"
    ["Please decode for me this message", "text"],
    ["Do not overwrite code that already works", "text"], // "overwrite code" contains "write code"
    ["Please generate imagery that is vivid", "text"], // "generate imagery" contains "generate image"
    ["How does draw_card work in my deck class?", "text"], // an identifier is one word
    ["Why does withdraw_funds() throw?", "text"],
  ]);
});

describe("ordinary sentences are not requests", () => {
  routes([
    // "function" and "program" are everyday nouns
    ["What is the function of the liver?", "text"],
    ["The function of this module is to parse the input", "text"],
    ["Show me the function of the heart", "text"],
    ["Write an essay about the function of the liver", "text"],
    ["How can I improve my kidney function?", "text"],
    ["Define the function of a database index", "text"],
    ["Make sure you know the function of each part", "text"],
    ["Write an essay about kidney function", "text"],
    ["Write a blog post on liver function", "text"],
    ["Write a summary of the main function", "text"],
    ["Write a note regarding the new function", "text"],
    ["Write it down. Functions are everywhere", "text"],
    ["Explain this function to me", "text"],
    ["Explain the function to the class", "text"],
    ["Show this program to my boss", "text"],
    ["Which exercise program is best for beginners?", "text"],
    ["What is on the TV program tonight?", "text"],
    ["He joined a training program last year", "text"],
  ]);
});

// The old rule was the substring "say " (a space after it), so it matched the word "say" followed
// by a space anywhere in a message, and the tail of every word that ends in "say". It said
// nothing about speech: "say hello" is as much a message to the assistant as "let's say we have
// a list" is. The ways to ask for speech are "speak", "voice", "text to speech" and "generate
// audio", and saying something out loud, which is the one request that depended on "say".
describe("'say' asks for audio only out loud", () => {
  routes([
    ["I'd say that is a fair price", "text"],
    ["Let's say we have a list of numbers", "text"],
    ["They say it will rain tomorrow", "text"],
    ["Can you say more about the plan?", "text"],
    ["Say hello to my little friend", "text"],
    ["What did she say?", "text"], // no space after "say": never matched, and still does not
    ["I couldn't say.", "text"],
    ["Say: hello there", "text"],
    ["Say, what time is it?", "text"],
    ["Say hello out loud", "audio"],
    ["Say 'good morning' aloud", "audio"],
    ["Please say this sentence out loud: good evening", "audio"],
    ["Say it in a robot voice", "audio"], // a voice is asked for in its own words
  ]);

  it("does not take 'aloud' from another sentence", () => {
    assert.equal(detectIntent("Let's say we have a list. Read the first item aloud."), "text");
  });
});

describe("genuine requests are still routed", () => {
  routes([
    // image
    ["Draw a cat", "image"],
    ["Please draw me a house with a red roof", "image"],
    ["Can you draw a dragon?", "image"],
    ["I want you to draw a mountain", "image"],
    ["Make a drawing of my dog", "image"],
    ["A hand-drawn map of Paris", "image"],
    ["Redraw this logo in blue", "image"],
    ["Generate image of a sunset", "image"],
    ["create images of cats", "image"],
    ["Give me a picture of a dog on a beach", "image"],
    ["Give me three drawings of cats", "image"],
    // video
    ["Generate video of ocean waves", "video"],
    ["create video about cats", "video"],
    ["Create videos of ocean waves", "video"],
    ["Animate this logo", "video"],
    ["Reanimate this old photo", "video"],
    ["Make an animated cat", "video"],
    ["Make a logo that animates in", "video"],
    ["a video of a dog surfing", "video"],
    // audio
    ["Generate audio of rain", "audio"],
    ["Generate audio that says hello and welcome", "audio"], // a suggestion on the welcome screen
    ["text to speech: hello world", "audio"],
    ["Speak this sentence aloud: good morning", "audio"],
    ["A clip of a robot speaking calmly", "audio"],
    ["Read this in a calm voice", "audio"],
    ["Use two different voices for the narrator", "audio"],
    ["I need a voiceover for my video", "audio"],
    ["the sound of thunder", "audio"],
    // code
    ["Write a fibonacci function in Python", "code"], // a suggestion on the welcome screen
    ["Write me a really fast fibonacci function in Python", "code"],
    ["Write code to reverse a string", "code"],
    ["Rewrite code to be faster", "code"],
    ["Implement a binary search tree", "code"],
    ["Reimplement it in Rust", "code"],
    ["Show me an implementation of quicksort in Python", "code"],
    ["Compare two implementations of quicksort", "code"],
    ["I am implementing a queue and need the pop method", "code"],
    ["Show how a stack is implemented in C", "code"],
    ["A class that implements the Comparable interface", "code"],
    ["Create a function that adds two numbers", "code"],
    ["Write functions to parse dates", "code"],
    ["Write programs that sort lists", "code"],
    ["Make this function faster", "code"],
    ["Fix this function: it returns undefined", "code"],
    ["Add a function to my class", "code"],
    ["Define a function that returns the sum", "code"],
    ["Show me a function that debounces calls", "code"],
    ["Python function to reverse a string", "code"],
    ["C program to find the sum of digits", "code"],
    ["A program for solving sudoku", "code"],
    ["A function that adds two numbers", "code"],
    ["A function which sorts a list", "code"],
    ["A function called parseDate", "code"],
    ["A program named fizzbuzz", "code"],
    ["Write a program that prints primes", "code"],
    ["code for a login form", "code"],
    ["What is wrong with this?\n```js\nconsole.log(1)\n```", "code"],
    // plain chat
    ["Hello! What can you do?", "text"],
    ["Summarise the attached file.", "text"],
    ["", "text"],
  ]);
});

// Each of these asks for code on its own, with nothing after the noun to say what it should do.
describe("every verb that asks for code on a function", () => {
  routes([
    ["Write a function", "code"],
    ["Create a function", "code"],
    ["Make a function", "code"],
    ["Build a function", "code"],
    ["Generate a function", "code"],
    ["Code a function", "code"],
    ["Develop a function", "code"],
    ["Define a function", "code"],
    ["Add a function", "code"],
    ["Fix this function", "code"],
    ["Debug this function", "code"],
    ["Refactor this function", "code"],
    ["Rewrite this function", "code"],
    ["Optimize this function", "code"],
    ["Optimise this function", "code"],
    ["Convert this function", "code"],
  ]);
});

// "a function to reverse a string" says what the function should do; "a function to me" says who
// it is explained to. A purpose is followed by a verb, a person or a thing by a pronoun or a
// determiner.
describe("'to' followed by a person or a thing is not what a function should do", () => {
  const objects = [
    ["me", ""],
    ["you", ""],
    ["him", ""],
    ["her", ""],
    ["us", ""],
    ["them", ""],
    ["it", ""],
    ["my", "boss"],
    ["your", "team"],
    ["his", "team"],
    ["its", "users"],
    ["our", "team"],
    ["their", "team"],
    ["the", "class"],
    ["a", "beginner"],
    ["an", "expert"],
    ["this", "audience"],
    ["that", "audience"],
    ["these", "students"],
    ["those", "students"],
  ];
  routes(objects.map(([word, rest]) => [`Describe the function to ${word} ${rest}`.trim(), "text"]));
  routes([["Describe the program to me", "text"]]);
});

describe("where the word sits in the message", () => {
  routes([
    // at the start and at the end
    ["Draw", "image"],
    ["Animate", "video"],
    ["Speak", "audio"],
    ["Implement", "code"],
    ["A dragon, please draw", "image"],
    ["Make it move, please animate", "video"],
    ["Read the poem, I want you to speak", "audio"],
    ["Write a program", "code"],
    ["Write a function", "code"],
    ["withdraw", "text"],
    ["essay", "text"],
    ["invoice", "text"],
    ["say", "text"],
    ["function", "text"],
    // between punctuation
    ["(draw) a cat", "image"],
    ["'draw' a cat", "image"],
    ["Draw: a cat", "image"],
    ["draw, then paint", "image"],
    ["Draw!", "image"],
    ["a cat... draw?", "image"],
    ["withdraw.", "text"],
    ["(withdraw)", "text"],
    ["'withdraw'", "text"],
    ["...withdraw!", "text"],
    ['"say hello"', "text"],
    ["say: hello", "text"],
    ["say\nhello", "text"],
    ["What did he say", "text"],
    ["Say hello, out loud!", "audio"],
    ["Say hello aloud!", "audio"],
    // in capitals
    ["DRAW A CAT", "image"],
    ["I WANT TO WITHDRAW MONEY", "text"],
    ["WITHDRAWAL", "text"],
    ["SAY HELLO", "text"],
    ["SAY HELLO OUT LOUD", "audio"],
    ["Say HELLO Aloud", "audio"],
    ["WRITE A FUNCTION THAT ADDS", "code"],
    ["EXPLAIN THE FUNCTION OF A DATABASE INDEX", "text"],
    // a hyphen or a new line is a boundary
    ["re-draw it", "image"],
    ["draw\na cat", "image"],
  ]);
});

describe("a message with several triggers", () => {
  routes([
    // the order is image, video, audio, code, and the first intent with a match wins, as before
    ["Draw a cat and animate it", "image"],
    ["Animate it, then speak", "video"],
    ["Speak the text, then write code for it", "audio"],
    ["Draw a function graph", "image"],
    ["Animate the program output", "video"],
    ["Say hello out loud and write a function that adds", "audio"],
    // a word that is not a request does not hide one that is
    ["Withdraw money and write code for the ATM", "code"],
    ["Write a function to withdraw money", "code"],
    ["I'd say that is nice, now generate audio of rain", "audio"],
    ["Let's say we draw a cat", "image"],
    ["Say hello, then draw a dragon", "image"],
    ["The invoice says hello, now animate it", "video"],
  ]);
});

// Trigger patterns with a gap in them ("write ... function", "say ... aloud") are linear because
// the gap is bounded. A trigger that can start a match and never finishes it is their worst case.
describe("the cost of classifying a long message", () => {
  // Classifying this much takes 3 to 14 ms. Left unbounded, the same gaps take 7 to 18 seconds.
  const CEILING_MS = 1000;
  const SIZE = 256 * 1024;
  const worstCases = {
    "'say '": "say ".repeat(SIZE / 4),
    "'write a '": "write a ".repeat(SIZE / 8),
    "'write' and then a lot of words": `write ${"a ".repeat(SIZE / 2)}`,
    "'function '": "function ".repeat(SIZE / 9),
    "'say' and no space": `say${"x".repeat(SIZE)}`,
  };

  for (const [name, message] of Object.entries(worstCases)) {
    it(`${SIZE / 1024} KB of ${name} is classified in well under ${CEILING_MS} ms`, () => {
      const started = process.hrtime.bigint();
      assert.equal(detectIntent(message), "text");
      const elapsed = Number(process.hrtime.bigint() - started) / 1e6;
      assert.ok(elapsed < CEILING_MS, `took ${elapsed.toFixed(0)} ms`);
    });
  }
});

describe("processMessage: the intent picks the path", () => {
  const originalAvailable = bridge.available;
  const originalRequest = bridge.request;
  let ops = [];

  beforeEach(() => {
    ops = [];
    bridge.available = () => true;
    bridge.request = async (op) => {
      ops.push(op);
      return { content: "a reply", file: "out.bin", finish_reason: "stop" };
    };
  });

  afterEach(() => {
    bridge.available = originalAvailable;
    bridge.request = originalRequest;
  });

  // What the worker was asked to do, and what kind of reply the user gets.
  async function route(content, type) {
    ops = [];
    const reply = await processMessage([{ role: "user", content }], type);
    return { op: ops[0], type: reply.type };
  }

  it("sends the messages in the issue to plain chat", async () => {
    for (const message of ["I want to withdraw money", "say hello", "Explain the function of a database index"]) {
      assert.deepEqual(await route(message), { op: "chat", type: "text" }, message);
    }
  });

  it("still sends genuine requests to their own path", async () => {
    assert.deepEqual(await route("Draw a cat"), { op: "image", type: "image" });
    assert.deepEqual(await route("Animate this logo"), { op: "video", type: "video" });
    assert.deepEqual(await route("Say hello out loud"), { op: "audio", type: "audio" });
    assert.deepEqual(await route("Write a fibonacci function in Python"), { op: "code", type: "code" });
  });

  it("does not second-guess a type the client chose", async () => {
    assert.deepEqual(await route("I want to withdraw money", "image"), { op: "image", type: "image" });
    assert.deepEqual(await route("Draw a cat", "code"), { op: "code", type: "code" });
  });
});
