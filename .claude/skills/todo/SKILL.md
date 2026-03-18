---
name: todo
description: Add a todo item to the project TODO.md file
user-invocable: true
argument-hint: [task description]
allowed-tools: Read, Write, Edit, Bash
---

Add a todo item to the project's `TODO.md` file.

## Task

The user wants to add: **$ARGUMENTS**

## Instructions

1. Read `TODO.md` in the project root. If it does not exist, create it with a `# TODO` heading.
2. Append a new unchecked item at the end of the list: `- [ ] $ARGUMENTS`
3. If the user included a priority (e.g. "high", "low"), format as: `- [ ] **[priority]** description`
4. Print the added item back to the user for confirmation.

Do NOT reorganize, reorder, or modify any existing items in the file.
