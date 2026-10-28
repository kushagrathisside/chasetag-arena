# Troubleshooting

Find your symptom in the tables below. Each row gives the likely cause and the fix.

## Starting the game

| Symptom | Likely cause | Fix |
|---|---|---|
| `uv: command not found` | The uv installer's path is not loaded yet. | Close the terminal, open a new one, and try again. |
| `uv sync` fails with a Python version error | uv could not find Python 3.12. | Run `uv python install 3.12`, then `uv sync` again. |
| The server stops with `Address already in use` | Another program, often a second copy of the game, is using the port. | Stop the other copy, or pick another port: `uv run chasetag-arena serve --port 8766`. |
| A message says `CUDA-enabled jaxlib is not installed. Falling back to cpu.` | JAX found a GPU but the game does not use one. | Nothing is wrong. The game runs on the CPU, as designed. |
| The browser does not open | The browser could not be started from the terminal, for example in WSL. | Copy `http://127.0.0.1:8765/` from the terminal into your browser. |
| Windows Firewall may ask about the program | The server asks to listen on a network. | Allow private networks, or cancel. The default address lets only this computer connect. |

## The page

| Symptom | Likely cause | Fix |
|---|---|---|
| The page says **Waiting** and the arena stays empty | The server is older than the page, so it sends a frame the page does not accept. | Stop the server with Ctrl+C, then start it again from the updated folder. |
| The status says **Reconnecting** | The server has stopped, or the connection dropped. | Check the terminal. If the server stopped, start it again, and the page reconnects. |
| The status says **Offline** | The computer has no network connection. | Connect again. The page reconnects by itself. Local play needs the server, not the internet. |
| The arrow keys do nothing | No seat is set to `keyboard`, a seat change is still pending, or a list has the focus. | Set a seat to `keyboard`, wait for the next round if the label says *Applies next round*, and click an empty part of the page. |
| Space or `.` does nothing | A button or a list has the focus. | Click an empty part of the page. |
| The body keeps moving after you release the keys | The page lost track of the keys. | Press and release an arrow key once. The body stops when the page next sends its command. |
| The speed list shows a speed you did not choose | The server runs at that speed, perhaps from another open page. | The list follows the server, so choose the speed you want again. |
| The planner choice is greyed out | No planner file was given, or the file was refused. | See [planners](planners.md#what-the-server-checks). |
| The Body list is greyed out | Only one body exists today. | Nothing is wrong. |

## Planners

The server checks a planner file before it starts, and it stops with a message when the file cannot
run. Each message is printed as `chasetag-arena: ` followed by the reason.

| Message ends with | What to do |
|---|---|
| `cannot read planner file ...` | Check the path, and that the file is not open or damaged. |
| `is not a planner file: it has no meta record` | The file is not a planner file. Use another file. |
| `is for the full-body model, which this arena does not play` | The file is for a different body model. Use a file made for this arena. |
| `decides every N control steps; the arena decides every 10` | The file's timing does not match. Use a file made for this arena. |
| `has a speed limit of ... m/s; the arena's is 1.5 m/s` | The file's speed limit does not match. Use a file made for this arena. |
| `uses activation ...; the arena supports ...` | The file's activation function is not supported. |
| `... has shape ..., expected ...` | The file's layers do not fit the arena. Use another file. |

See [planners](planners.md) for what each check means.

## Games that look wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| The round score reads 0.000 partway through a round | The round score is only set when the round ends. | This is expected. Wait for the round to end, and the score appears during the intermission. |
| The clock does not match the wall clock | The clock counts game time, and speed changes how fast it runs in wall-clock time. | This is expected. See [the game](the-game.md#how-a-round-ends). |
| The match totals did not reset | Only **Restart** clears the totals. Starting a new round does not. | Press **Restart**. |

## Still stuck?

1. Stop the server with Ctrl+C, and start it again with `uv run chasetag-arena serve`.
2. Reload the page.
3. Check the terminal for any message printed in red or with the word `Error`.

If the problem continues, open an issue on the
[project's GitHub page](https://github.com/kushagrathisside/chasetag-arena/issues), and describe the
steps that lead to it, the message in the terminal, and the browser you used.
