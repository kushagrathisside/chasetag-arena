# Install

chasetag-arena runs from its source code. There is no installer yet, so you get the code, let
`uv` set up the Python environment, and start the server. The steps take a few minutes, and most
of that time is the first download.

## What you need

| Requirement | Why | Notes |
|---|---|---|
| **Python 3.12** | The game is tested on this version only. | `uv` can download it for you, so you do not need to install it yourself. |
| **uv** | It installs the Python environment and runs the game. | Free, from [docs.astral.sh/uv](https://docs.astral.sh/uv/). |
| **git** | It copies the code to your computer. | You can also download the code as a ZIP file from the repository page. |
| **A modern browser** | It shows the game. | Chrome, Edge, Firefox or Safari, on the same computer as the server. |
| **About 1 GB of disk** | The Python packages, mostly JAX. | The game needs no GPU. JAX runs on the CPU. |

## Step 1: install uv

Pick the command for your system. Each one is the official installer from the uv project.

=== "Linux and macOS"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows (PowerShell)"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

Close the terminal and open a new one, so that the `uv` command is on your path. Check it:

```bash
uv --version
```

## Step 2: get the code

```bash
git clone https://github.com/kushagrathisside/chasetag-arena.git
cd chasetag-arena
```

## Step 3: set up the environment

Run this once, inside the `chasetag-arena` folder:

```bash
uv sync
```

`uv sync` creates a `.venv` folder with Python 3.12 and every package the game needs. Run it again
only after you update the code.

## Step 4: check the install

```bash
uv run chasetag-arena --help
```

You should see the list of commands:

```text
usage: chasetag-arena [-h] COMMAND ...

A grounded 2D chase-tag game, played in a browser.

positional arguments:
  COMMAND
    serve     run the arcade on this machine

options:
  -h, --help  show this help message and exit
```

If you see that, the install works. Go on to [your first round](getting-started.md).

## Updating

To get a newer version of the game:

```bash
git pull
uv sync
```

## Where things go

| Path | What it is |
|---|---|
| `.venv/` | The Python environment that `uv` created. Safe to delete. `uv sync` rebuilds it. |
| `src/chasetag_arena/` | The game's code. |
| `docs/` | The documentation you are reading. |

## Running on Windows

Run the commands in PowerShell, or inside WSL (Windows Subsystem for Linux). When you run the game
inside WSL, the browser on Windows may not open by itself. If it does not, copy the address the
server prints, `http://127.0.0.1:8765/`, into your browser. The [troubleshooting](troubleshooting.md)
page covers the other common problems.

## Uninstall

Delete the `chasetag-arena` folder. Nothing else is installed on your computer.
