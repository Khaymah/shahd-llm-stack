// Frozen v1 multi-turn cases (30). Each conversation ends with a User turn that only makes sense with
// the earlier context (a rename, a follow-up edit, a recall question). MustContain holds short,
// unambiguous substrings (identifiers, numbers, names; matched case-insensitively); MustNotContain
// holds what a reply that LOST the context would show (the old name after a rename, the replaced
// syntax after a conversion).

import type { MultiTurnCase } from "../SuiteTypes.ts";

export const MultiTurnCases: readonly MultiTurnCase[] = [
  { Id: "mt-001", Slice: "ts", Turns: [
    { Role: "User", Content: "Write a TypeScript function that adds up an array of prices." },
    { Role: "Assistant", Content: "```ts\nfunction sumPrices(prices: number[]): number {\n  return prices.reduce((acc, p) => acc + p, 0);\n}\n```" },
    { Role: "User", Content: "Call it `total` instead." },
  ], MustContain: ["total("], MustNotContain: ["sumPrices"] },
  { Id: "mt-002", Slice: "ts", Turns: [
    { Role: "User", Content: "Write a function that fetches a user by id from /api/users/:id." },
    { Role: "Assistant", Content: "```ts\nfunction fetchUser(id: string): Promise<unknown> {\n  return fetch(`/api/users/${id}`).then((res) => res.json());\n}\n```" },
    { Role: "User", Content: "Now make it use async/await." },
  ], MustContain: ["async", "await", "fetchUser"], MustNotContain: [".then("] },
  { Id: "mt-003", Slice: "general", Turns: [
    { Role: "User", Content: "I'll give you some numbers: 42, 17 and 8." },
    { Role: "Assistant", Content: "Got it: 42, 17 and 8." },
    { Role: "User", Content: "What was the first number I gave you?" },
  ], MustContain: ["42"] },
  { Id: "mt-004", Slice: "general", Turns: [
    { Role: "User", Content: "My name is Layla and I'm building a recipe app." },
    { Role: "Assistant", Content: "Nice to meet you, Layla! A recipe app sounds fun. How can I help?" },
    { Role: "User", Content: "What's my name?" },
  ], MustContain: ["Layla"] },
  { Id: "mt-005", Slice: "ts", Turns: [
    { Role: "User", Content: "Define a TypeScript interface for a product with an id and a name." },
    { Role: "Assistant", Content: "```ts\ninterface Product {\n  id: number;\n  name: string;\n}\n```" },
    { Role: "User", Content: "Add a price field of type number." },
  ], MustContain: ["Product", "price: number"] },
  { Id: "mt-006", Slice: "js", Turns: [
    { Role: "User", Content: "Write a JavaScript function `double(n)` that doubles a number." },
    { Role: "Assistant", Content: "```js\nfunction double(n) {\n  return n * 2;\n}\n```" },
    { Role: "User", Content: "Change it to triple the number and rename it to `triple`." },
  ], MustContain: ["triple(", "3"], MustNotContain: ["double("] },
  { Id: "mt-007", Slice: "css", Turns: [
    { Role: "User", Content: "Give me CSS for a .card class with an 8px border radius." },
    { Role: "Assistant", Content: "```css\n.card {\n  border-radius: 8px;\n}\n```" },
    { Role: "User", Content: "Make the radius 12px instead." },
  ], MustContain: [".card", "12px"], MustNotContain: ["8px"] },
  { Id: "mt-008", Slice: "html", Turns: [
    { Role: "User", Content: "Write the HTML for a button that says Save." },
    { Role: "Assistant", Content: "```html\n<button type=\"button\">Save</button>\n```" },
    { Role: "User", Content: "Change the label to Submit and make it a submit button." },
  ], MustContain: ["<button", "type=", "Submit"], MustNotContain: [">Save<"] },
  { Id: "mt-009", Slice: "sql", Turns: [
    { Role: "User", Content: "Write SQL to select all rows from the customers table." },
    { Role: "Assistant", Content: "```sql\nSELECT * FROM customers;\n```" },
    { Role: "User", Content: "Only the ones from Cairo. The column is called city." },
  ], MustContain: ["customers", "WHERE", "city", "Cairo"] },
  { Id: "mt-010", Slice: "go", Turns: [
    { Role: "User", Content: "Write a Go function Greet(name string) string that returns \"Hello, \" + name." },
    { Role: "Assistant", Content: "```go\nfunc Greet(name string) string {\n\treturn \"Hello, \" + name\n}\n```" },
    { Role: "User", Content: "Rename it to Welcome." },
  ], MustContain: ["func Welcome("], MustNotContain: ["func Greet("] },
  { Id: "mt-011", Slice: "general", Turns: [
    { Role: "User", Content: "Let's use port 8080 for the dev server." },
    { Role: "Assistant", Content: "Sounds good, the dev server will listen on port 8080." },
    { Role: "User", Content: "Which port did we pick?" },
  ], MustContain: ["8080"] },
  { Id: "mt-012", Slice: "general", Turns: [
    { Role: "User", Content: "I'm using Bun with Hono for my API." },
    { Role: "Assistant", Content: "Great combination: Hono is small and fast, and Bun runs TypeScript directly." },
    { Role: "User", Content: "Which runtime did I say I'm using?" },
  ], MustContain: ["Bun"] },
  { Id: "mt-013", Slice: "ts", Turns: [
    { Role: "User", Content: "Write a TypeScript function `isEven(n: number): boolean`." },
    { Role: "Assistant", Content: "```ts\nfunction isEven(n: number): boolean {\n  return n % 2 === 0;\n}\n```" },
    { Role: "User", Content: "Add a JSDoc comment to it." },
  ], MustContain: ["/**", "isEven("] },
  { Id: "mt-014", Slice: "js", Turns: [
    { Role: "User", Content: "Here's my array: [3, 1, 2]." },
    { Role: "Assistant", Content: "Got it, your array is [3, 1, 2]. What would you like to do with it?" },
    { Role: "User", Content: "What is the sum of the numbers in my array?" },
  ], MustContain: ["6"] },
  { Id: "mt-015", Slice: "general", Turns: [
    { Role: "User", Content: "The project I'm working on is called Nebula." },
    { Role: "Assistant", Content: "Nice name! What is Nebula about?" },
    { Role: "User", Content: "Suggest a name for its main package, using the project name in lower case." },
  ], MustContain: ["nebula"] },
  { Id: "mt-016", Slice: "ts", Turns: [
    { Role: "User", Content: "Write a function `formatPrice(cents: number): string` that returns dollars like $4.99." },
    { Role: "Assistant", Content: "```ts\nfunction formatPrice(cents: number): string {\n  return \"$\" + (cents / 100).toFixed(2);\n}\n```" },
    { Role: "User", Content: "Change the currency symbol to €." },
  ], MustContain: ["formatPrice", "€"] },
  { Id: "mt-017", Slice: "ts", Turns: [
    { Role: "User", Content: "Here's my code:\n```ts\nconst usr = await getUser(id);\nconsole.log(usr.name);\n```" },
    { Role: "Assistant", Content: "That looks fine: it loads the user and logs their name." },
    { Role: "User", Content: "Rename the variable to currentUser." },
  ], MustContain: ["currentUser"], MustNotContain: ["usr."] },
  { Id: "mt-018", Slice: "js", Turns: [
    { Role: "User", Content: "Write an arrow function called square that squares a number." },
    { Role: "Assistant", Content: "```js\nconst square = (n) => n * n;\n```" },
    { Role: "User", Content: "Convert it to a regular function declaration." },
  ], MustContain: ["function square("], MustNotContain: ["=>"] },
  { Id: "mt-019", Slice: "css", Turns: [
    { Role: "User", Content: "Make a flex container class .row with an 8px gap." },
    { Role: "Assistant", Content: "```css\n.row {\n  display: flex;\n  gap: 8px;\n}\n```" },
    { Role: "User", Content: "Now make it stack vertically instead." },
  ], MustContain: [".row", "flex-direction", "column"] },
  { Id: "mt-020", Slice: "html", Turns: [
    { Role: "User", Content: "Write an img tag for /cat.jpg." },
    { Role: "Assistant", Content: "```html\n<img src=\"/cat.jpg\" alt=\"\">\n```" },
    { Role: "User", Content: "Add the alt text: A sleeping cat." },
  ], MustContain: ["/cat.jpg", "A sleeping cat"] },
  { Id: "mt-021", Slice: "general", Turns: [
    { Role: "User", Content: "I have 3 apples." },
    { Role: "Assistant", Content: "Okay, you have 3 apples." },
    { Role: "User", Content: "I eat one of them." },
    { Role: "Assistant", Content: "Then you have 2 apples left." },
    { Role: "User", Content: "Now I buy 5 more. How many apples do I have?" },
  ], MustContain: ["7"] },
  { Id: "mt-022", Slice: "go", Turns: [
    { Role: "User", Content: "Write a Go struct Point with X and Y int fields." },
    { Role: "Assistant", Content: "```go\ntype Point struct {\n\tX int\n\tY int\n}\n```" },
    { Role: "User", Content: "Add a Z field too." },
  ], MustContain: ["type Point struct", "Z int"] },
  { Id: "mt-023", Slice: "sql", Turns: [
    { Role: "User", Content: "Count the rows in the orders table." },
    { Role: "Assistant", Content: "```sql\nSELECT COUNT(*) FROM orders;\n```" },
    { Role: "User", Content: "Now group that count by the status column." },
  ], MustContain: ["orders", "GROUP BY status"] },
  { Id: "mt-024", Slice: "ts", Turns: [
    { Role: "User", Content: "My API returns { data: User[] }. Write a type for it named UsersResponse." },
    { Role: "Assistant", Content: "```ts\ntype UsersResponse = {\n  data: User[];\n};\n```" },
    { Role: "User", Content: "Make it generic over the item type, keeping the same name." },
  ], MustContain: ["UsersResponse<"] },
  { Id: "mt-025", Slice: "general", Turns: [
    { Role: "User", Content: "Answer in one word from now on. What's the capital of France?" },
    { Role: "Assistant", Content: "Paris." },
    { Role: "User", Content: "And of Egypt?" },
  ], MustContain: ["Cairo"] },
  { Id: "mt-026", Slice: "js", Turns: [
    { Role: "User", Content: "I have const users = [{ name: 'Ali', age: 30 }, { name: 'Mona', age: 25 }];" },
    { Role: "Assistant", Content: "Got it: an array of two user objects with name and age." },
    { Role: "User", Content: "Write one line that gets the names from that array." },
  ], MustContain: ["users", ".map(", "name"] },
  { Id: "mt-027", Slice: "ts", Turns: [
    { Role: "User", Content: "Write a function `greet(name: string): string` that returns 'Hello, ' + name." },
    { Role: "Assistant", Content: "```ts\nfunction greet(name: string): string {\n  return \"Hello, \" + name;\n}\n```" },
    { Role: "User", Content: "Add an optional second parameter `greeting` that defaults to 'Hello'." },
  ], MustContain: ["greet(", "greeting", "Hello"] },
  { Id: "mt-028", Slice: "css", Turns: [
    { Role: "User", Content: "My favourite colour is teal." },
    { Role: "Assistant", Content: "Teal is a lovely colour!" },
    { Role: "User", Content: "Write a CSS custom property for my favourite colour, using the colour keyword." },
  ], MustContain: ["--", "teal"] },
  { Id: "mt-029", Slice: "ts", Turns: [
    { Role: "User", Content: "This line fails to compile: `const n: number = '5';`" },
    { Role: "Assistant", Content: "You're assigning a string to a number. Either write `const n: number = 5;` or convert the string with `Number('5')`." },
    { Role: "User", Content: "Show me the version that converts the string." },
  ], MustContain: ["Number("] },
  { Id: "mt-030", Slice: "general", Turns: [
    { Role: "User", Content: "I'll call the first variable alpha." },
    { Role: "Assistant", Content: "Okay, the first variable is alpha." },
    { Role: "User", Content: "And the second one beta." },
    { Role: "Assistant", Content: "Got it, the second variable is beta." },
    { Role: "User", Content: "What did I name the first variable?" },
  ], MustContain: ["alpha"] },
];
