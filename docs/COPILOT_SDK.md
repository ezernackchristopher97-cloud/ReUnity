# GitHub Copilot SDK Guide for ReUnity

This guide explains how to use the GitHub Copilot SDK (technical preview) to assist with development, testing, and workflows in this repository. It is written for beginners and provides step-by-step instructions with copy-paste commands.

---

## Table of Contents

1. [What is the GitHub Copilot SDK?](#what-is-the-github-copilot-sdk)
2. [What Can the SDK Do?](#what-can-the-sdk-do)
3. [How the SDK Fits Into This Repository](#how-the-sdk-fits-into-this-repository)
4. [Prerequisites](#prerequisites)
5. [Installing and Authenticating Copilot CLI](#installing-and-authenticating-copilot-cli)
6. [Using the Copilot SDK in This Repo](#using-the-copilot-sdk-in-this-repo)
7. [Example Workflows for Faraz](#example-workflows-for-faraz)
8. [What Not to Do](#what-not-to-do)
9. [Relationship to GitHub Models, Actions, and Codespaces](#relationship-to-github-models-actions-and-codespaces)

---

## What is the GitHub Copilot SDK?

The GitHub Copilot SDK is a software development kit (currently in **technical preview**) that lets you use the same AI agent system that powers the GitHub Copilot CLI, but from your own code or scripts.

**In simple terms:**
- It is a tool that lets you talk to GitHub Copilot programmatically (through code, not just chat)
- It removes the need to build your own AI agent framework from scratch
- It works with multiple programming languages: Python, TypeScript/Node.js, Go, and .NET

**Important:** The SDK is for **assisting** development, testing, planning, and orchestration. It does **not** replace the code in this repository. Think of it as a helper that can understand your code and help you work with it.

---

## What Can the SDK Do?

The Copilot SDK provides several powerful capabilities:

| Feature | What It Means |
|---------|---------------|
| **Multi-step planning** | Copilot can break down complex tasks into steps and execute them one by one |
| **Multi-model routing** | The system can choose different AI models for different types of tasks |
| **Tool orchestration** | Copilot can run commands, edit files, and use tools on your behalf |
| **Error recovery** | If something fails, Copilot can try to fix it or suggest alternatives |
| **Session memory** | Copilot remembers what you discussed earlier in the same session |
| **MCP server support** | Copilot can connect to external tools and services through MCP (Model Context Protocol) servers |

---

## How the SDK Fits Into This Repository

The SDK is **optional** and **additive**. You do not need it to run ReUnity. However, it can help you:

- Explore and understand the codebase
- Reason about code changes before making them
- Run tests and simulations with guidance
- Debug issues by asking Copilot to explain errors
- Orchestrate workflows defined in the Makefile and scripts

**The SDK does NOT:**
- Automatically modify code unless you explicitly ask it to
- Replace any existing functionality in ReUnity
- Run without your approval for file changes

Use it as a **helper**, not a replacement for understanding the code yourself.

---

## Prerequisites

Before using the Copilot SDK, you need:

| Requirement | Description |
|-------------|-------------|
| **GitHub account** | A free or paid GitHub account |
| **GitHub Copilot access** | A Copilot subscription (Pro, Pro+, Business, or Enterprise). A free tier with limited usage is available. |
| **GitHub Copilot CLI** | The command-line tool that the SDK communicates with |
| **Terminal access** | Either GitHub Codespaces or a local terminal on your computer |

**Note:** You do NOT need a GPU. Everything runs on CPU.

---

## Installing and Authenticating Copilot CLI

### Step 1: Check if Copilot CLI is Already Installed

Open your terminal and run:

```bash
copilot --version
```

If you see a version number, Copilot CLI is already installed. Skip to Step 3.

If you see "command not found", continue to Step 2.

### Step 2: Install Copilot CLI

**On macOS (using Homebrew):**

```bash
brew install gh
gh extension install github/gh-copilot
```

**On Linux (using apt):**

```bash
sudo apt update
sudo apt install gh
gh extension install github/gh-copilot
```

**On Windows (using winget):**

```powershell
winget install GitHub.cli
gh extension install github/gh-copilot
```

**Alternative: Direct installation from GitHub:**

Visit the official installation guide: https://docs.github.com/en/copilot/how-tos/use-copilot-agents/use-copilot-cli

### Step 3: Authenticate with GitHub

Run the following command:

```bash
gh auth login
```

Follow the on-screen prompts:
1. Select "GitHub.com"
2. Choose your preferred authentication method (browser is easiest)
3. Complete the login in your browser

### Step 4: Verify Installation

Run:

```bash
copilot --help
```

You should see a list of available commands. If you see this, you are ready to use Copilot CLI.

---

## Using the Copilot SDK in This Repo

### Starting a Copilot CLI Session

1. Open your terminal
2. Navigate to the ReUnity repository:

```bash
cd /path/to/ReUnity
```

3. Start Copilot CLI:

```bash
copilot
```

4. Copilot will ask if you trust the files in this folder. Choose one of:
   - **Yes, proceed** (trust for this session only)
   - **Yes, and remember** (trust for all future sessions)

5. If prompted, log in using:

```
/login
```

### Basic Commands

Once in an interactive session, you can:

| Command | What It Does |
|---------|--------------|
| Type a question | Ask Copilot anything about the code |
| `@filename` | Include a specific file in your question (e.g., `@src/reunity/core/entropy.py`) |
| `/help` | Show available slash commands |
| `/agent` | Select a specialized agent for your task |
| `Esc` | Stop the current operation |
| `/cwd /path` | Change the working directory |
| `!command` | Run a shell command directly (e.g., `!make test`) |

### Example: Ask Copilot to Explain the Repo Structure

```
Explain the directory structure of this repository and what each folder contains.
```

### Example: Ask Copilot to Walk Through a Script

```
Walk me through what @scripts/run_sim_tests.py does step by step.
```

### Example: Ask Copilot to Help Run Tests

```
Help me run the simulation tests. What command should I use?
```

---

## Example Workflows for Faraz

Here are concrete examples of how to use Copilot in this repository:

### Workflow 1: Understanding a Script

**Goal:** Understand what `run_sim_tests.py` does.

```
Explain @scripts/run_sim_tests.py in simple terms. What does each function do?
```

### Workflow 2: Running Simulation Stage 2

**Goal:** Run the Pre-RAG gates simulation test.

```
Walk me through running sim-stage2. What does it test and how do I run it?
```

Copilot will explain and may suggest:

```bash
make sim-stage2
```

### Workflow 3: Understanding a Test Failure

**Goal:** Figure out why a test failed.

First, run the tests:

```bash
make test
```

If a test fails, ask Copilot:

```
The test test_entropy_stable_state failed. Can you explain what this test checks and why it might have failed?
```

### Workflow 4: Modifying a Config Flag Safely

**Goal:** Change a configuration setting without breaking anything.

```
I want to change the ENTROPY_THRESHOLD_HIGH value in the config. Where is it defined and what will changing it affect?
```

### Workflow 5: Exploring the Entropy Module

**Goal:** Understand how entropy detection works.

```
Explain how the EntropyStateDetector class in @src/reunity/core/entropy.py works. What are Shannon entropy and Jensen-Shannon divergence?
```

### Workflow 6: Getting Help with the Makefile

**Goal:** Understand what Make targets are available.

```
List all the Make targets in this repo and explain what each one does.
```

---

## What Not to Do

**Important warnings when using Copilot:**

| Do NOT | Why |
|--------|-----|
| Commit secrets or API keys | Copilot might accidentally include them in suggestions |
| Let Copilot auto-modify large sections without review | Always review changes before accepting them |
| Assume Copilot output is always correct | Copilot can make mistakes. Verify important information. |
| Skip running tests after changes | Always run `make test` after any code changes |
| Share sensitive personal information | Copilot sessions may be logged |

**Best practices:**
- Review every file change before accepting
- Run tests after any modification
- Ask Copilot to explain its reasoning
- Use Copilot as a helper, not a replacement for understanding

---

## Relationship to GitHub Models, Actions, and Codespaces

Understanding how these GitHub features relate to each other:

### GitHub Codespaces

- **What it is:** A cloud development environment (like a virtual computer in your browser)
- **Use it for:** Development and CPU-based testing
- **GPU support:** GitHub has deprecated GPU machine types in Codespaces. Do not rely on Codespaces for GPU workloads.
- **Copilot in Codespaces:** Copilot CLI works in Codespaces just like on your local machine

### GitHub Models

- **What it is:** A platform for testing models, prompts, and evaluations
- **Use it for:** Prompt engineering and model evaluation workflows
- **Not for:** Custom model hosting (by default)
- **Copilot relationship:** GitHub Models is separate from the Copilot SDK

### GitHub Actions

- **What it is:** Automation platform for CI/CD workflows
- **Use it for:** Running tests, builds, and deployments automatically
- **GPU support:** Available through GPU hosted runners (requires specific account/org plans) or self-hosted runners
- **Copilot relationship:** The SDK does not directly integrate with Actions, but you can use Copilot to help write Action workflows

### Summary Table

| Feature | Purpose | GPU Support | Copilot Integration |
|---------|---------|-------------|---------------------|
| Codespaces | Development environment | No (deprecated) | Yes, via CLI |
| GitHub Models | Model testing/evaluation | Depends on model | Separate from SDK |
| GitHub Actions | CI/CD automation | Via GPU runners | Indirect (helps write workflows) |
| Copilot SDK | Programmatic agent access | Not required | This is the SDK |

---

## Quick Reference Card

### Starting a Session

```bash
cd /path/to/ReUnity
copilot
```

### Useful Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/login` | Authenticate with GitHub |
| `/agent` | Select a specialized agent |
| `/cwd /path` | Change working directory |
| `/add-dir /path` | Add a trusted directory |

### Including Files in Prompts

```
Explain @src/reunity/core/entropy.py
```

### Running Shell Commands

```
!make test
```

### Stopping an Operation

Press `Esc`

---

## Additional Resources

- [GitHub Copilot SDK Repository](https://github.com/github/copilot-sdk)
- [Copilot CLI Documentation](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/use-copilot-cli)
- [Copilot CLI Installation Guide](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/install-copilot-cli)
- [About GitHub Copilot CLI](https://docs.github.com/en/copilot/concepts/agents/about-copilot-cli)

---

## Troubleshooting

### "copilot: command not found"

The CLI is not installed or not in your PATH. Follow the installation steps in Section 5.

### "Authentication required"

Run `/login` in your Copilot session or `gh auth login` in your terminal.

### "Permission denied" when running commands

Copilot asks for permission before modifying files. Choose "Yes" to allow, or "No" to reject.

### Session not remembering context

Make sure you are in the same interactive session. Starting a new `copilot` command creates a new session.

### Copilot suggests incorrect code

Always review suggestions before accepting. Run tests after any changes.

---

**DISCLAIMER:** ReUnity is NOT a clinical or treatment tool. The Copilot SDK is a development assistant and does not provide medical or psychological advice.
