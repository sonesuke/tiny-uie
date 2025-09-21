# Create GitHub Issue

Start an interactive GitHub issue creation session. This command will guide you through defining requirements and creating a well-structured GitHub issue.

## Usage

```
/issue
```

This will start an interactive session where I'll help you:
1. Define the issue requirements through conversation
2. Structure the information into a clear issue format
3. Create the GitHub issue when you're ready

## Workflow

1. **Start**: Use `/issue` to begin the issue creation process
2. **Requirements Gathering**: I'll ask questions to understand:
   - What problem needs to be solved or what feature is needed
   - Context and background information
   - Expected behavior or acceptance criteria
   - Any technical considerations
3. **Review**: I'll summarize and structure the collected information
4. **Create**: Say "make issue" to create the GitHub issue in the current repository

## Important Notes

- **Read-only mode**: During the issue creation session, no code files will be modified or created
- The session is focused solely on gathering requirements and creating the GitHub issue
- All code analysis will be read-only for understanding context only

## Features

- Interactive requirements gathering
- Structured issue format with clear sections
- Uses GitHub CLI (`gh`) for issue creation
- Automatically detects current repository
- Creates professional, well-formatted issues
- **Read-only mode**: No code modifications or file changes during issue creation session

## Example Session

```
User: /issue
Assistant: I'll help you create a GitHub issue. Let's start by discussing what you need...

[Interactive conversation to gather requirements]

User: make issue
Assistant: Creating GitHub issue...
[Issue created successfully]
```

## Requirements

- GitHub CLI (`gh`) must be installed and authenticated
- Current directory must be a Git repository
- Repository must be hosted on GitHub

## Issue Template

The created issues will follow this structure:

```markdown
## Summary
[Brief description of the issue]

## Background
[Context and background information]

## Requirements
[Detailed requirements or problem description]

## Acceptance Criteria
- [ ] [Specific criteria for completion]
- [ ] [Additional criteria as needed]

## Technical Notes
[Any technical considerations or constraints]
```