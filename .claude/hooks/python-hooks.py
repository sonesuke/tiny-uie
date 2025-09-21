"""Python hooks for Claude Code.

This module contains hooks that intercept and process tool calls to enforce
project-specific rules and constraints.
"""

import json
import re
import subprocess
import sys


def main() -> None:
    """Main entry point for the hook script.

    Reads JSON input from stdin, processes tool commands, and decides whether to approve or block them.
    """
    # Read JSON input from stdin
    input_data = json.load(sys.stdin)

    tool_name = input_data.get('tool_name', '')
    command = input_data.get('tool_input', {}).get('command', '')

    # Only process if this is a Bash tool call
    if tool_name == 'Bash':
        # Check if the command starts with 'python' (but not 'uv run python' or 'uv python')
        if (re.match(r'^python\s', command) or command == 'python') and not re.match(r'^uv\s+(run\s+)?python', command):
            # Return JSON response to block the tool use
            response = {
                'decision': 'block',
                'message': (
                    'Error: Direct python execution is not allowed in this project. '
                    'Please use \'uv run python\' instead of \'python\'. '
                    'Example: uv run python script.py'
                )
            }
            print(json.dumps(response))
            return 0

        # Check if the command is 'git commit'
        if re.match(r'^git\s+commit', command):
            print('Running commit checks...', file=sys.stderr)
            print('This may take a moment...', file=sys.stderr)

            try:
                # Find the project root by looking for pyproject.toml
                result = subprocess.run(['/usr/bin/git', 'rev-parse', '--show-toplevel'],
                                        capture_output=True, text=True, check=True)
                project_root = result.stdout.strip()
            except subprocess.CalledProcessError:
                response = {
                    'decision': 'block',
                    'message': 'Error: Not in a git repository'
                }
                print(json.dumps(response))
                return 0

            # Run make check
            print('Running make check...', file=sys.stderr)
            try:
                subprocess.run(['/usr/bin/make', 'check'], cwd=project_root, check=True,
                               stdout=sys.stderr, stderr=sys.stderr)
            except subprocess.CalledProcessError:
                response = {
                    'decision': 'block',
                    'message': 'Error: Checks failed. Please fix issues before committing.'
                }
                print(json.dumps(response))
                return 0

            print('All checks passed! Proceeding with commit...', file=sys.stderr)

    # Allow the tool to proceed
    response = {
        'decision': 'approve'
    }
    print(json.dumps(response))
    return None

if __name__ == '__main__':
    main()
