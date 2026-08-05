# Contributing

Thank you for your interest in contributing to the Neural Network Notions project.

## Before opening an issue
- Search existing issues.
- Make sure the problem has not already been reported.

## Before submitting a pull request
Please:
- Keep changes focused.
- Follow the existing code style.
- Avoid introducing any external dependencies.
- Ensure the framework builds successfully.
- Include tests when applicable.
- Document any public API changes.

## Coding style
### C#
- Follow existing naming conventions (eg. PascalCase for class, function and property names, camelCase for local variables, etc.).
- Avoid unnecessary allocations.
- Explicitly release native C++ memory whenever possible/applicable.
- Clearly comment code which is complex, hard to read, etc.
- Avoid introducing external dependencies.
- Prefer Span<T> where appropriate.

### C++
- Follow existing naming and formatting conventions (eg. PascalCase for class names, camelCase for function names, etc.).
- Avoid unnecessary allocations.
- Clearly comment code which is complex, hard to read, etc.
- Prefer standard library facilities.
- Avoid introducing external dependencies.

## Performance
Although not explicitly focused on maximizing performance, Neural Network Notions still prioritizes:
- Low allocations
- Efficient tensor operations

Avoid changes which reduce performance without measurable benefits.

## Pull Requests
Describe:
- What changed
- Why the change was made
- Any performance impact
