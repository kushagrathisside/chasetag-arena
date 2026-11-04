# Community and data

The data, trained planners and demos of chasetag-arena are shared by the **ChaseTag Arena**
organization on Hugging Face, at <https://huggingface.co/chasetag-arena>. Everyone who plays the
game or builds on it is welcome to take part.

## What is shared there

| Kind | Repository | What it is |
|---|---|---|
| Dataset | [chasetag-arena-bot-episodes](https://huggingface.co/datasets/chasetag-arena/chasetag-arena-bot-episodes) | 480,000 games between five scripted bots, in 24 seat pairings, at 20 s and 60 s rounds, recorded decision by decision |
| Models | [planner-k-20s](https://huggingface.co/chasetag-arena/planner-k-20s), [planner-k-60s](https://huggingface.co/chasetag-arena/planner-k-60s), [planner-k-180s](https://huggingface.co/chasetag-arena/planner-k-180s), [planner-k-all](https://huggingface.co/chasetag-arena/planner-k-all) | Trained [planners](planners.md) for the default body, one for each round length and one for all three |

Each repository has a card that says what it holds, how it was made, how well it did, and under
which license it may be used.

## Play against a published planner

Download a planner file, then load it with `--planner`. Use the round length that the planner's
card says it was trained for:

```bash
hf download chasetag-arena/planner-k-60s planner.npz --local-dir planners/k-60s
uv run chasetag-arena serve --planner planners/k-60s/planner.npz --time-limit 60
```

Then set a seat to **planner** on the page. A change applies at the next round. See
[Planners](planners.md) for what a planner sees and does.

## Contribute

You can share:

- trained planners (models);
- datasets of simulated games, from bots, planners or self-play;
- evaluation results.

Anyone with a Hugging Face account can open a pull request on a public repository in the
organization. A maintainer reviews every pull request before it is merged. The contribution guide
in the organization explains what each kind of contribution needs.

Recordings of real people's games are a special case, because they can reveal personal information.
They are accepted only when you recorded them with the game's own consent screen (see the next
section), and only once the final consent text is released. Until then, please do not upload any.

Never include personal data in a contribution: names, email addresses, device or network
identifiers, file paths from your computer, or tokens.

## Contribute your games

If you play the game, you can choose to record your own games and share them. This is off unless you
turn it on:

```bash
uv run chasetag-arena serve --contribute
```

Nothing is recorded until you press **Contribute games** in the page, read the consent text and
agree. Then:

- **What is recorded.** For each round you play with the keyboard: where both bodies are and which
  way they face, and the command each seat chose, every 0.1 s; who controlled each seat (you are
  recorded only as `human`); the round length, how it ended and the scores. Not your name, email,
  account, location, IP address, device, or anything you type. A round with two keyboard players is
  not recorded.
- **Where it goes.** Into a folder on your computer, in the same layout as the
  [bot episodes dataset](https://huggingface.co/datasets/chasetag-arena/chasetag-arena-bot-episodes).
  Nothing leaves your computer unless you choose to share it.
- **A random ID.** Your files carry a random contribution ID, made on your computer. It lets you or
  the maintainers find your games if you later ask for them to be removed.
- **Your controls.** **Review recorded games** shows the file exactly as it would be shared.
  **Stop recording** keeps what you have. **Export a package** makes a zip with your games and a
  receipt, ready to contribute by pull request. **Withdraw and delete everything** removes the
  games, the consent record and any packages this program wrote on your computer.
- **Sharing.** A shared package is published on Hugging Face under CC BY-NC 4.0, and others may copy
  it, so removing it later cannot undo copies that exist.

!!! warning "The consent text is a draft"
    The consent text has not been reviewed yet. Packages made now are marked as draft and will not be
    accepted. Please do not contribute real games until the final text is released.

The option only works on this computer: the server refuses `--contribute` on any address other than
`127.0.0.1`, `localhost` or `::1`.

## Licenses

Each repository's card states its license. Data and models are shared under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) unless a card says otherwise. The
game and simulator are licensed under the PolyForm Noncommercial License 1.0.0. See
[License](license.md).
