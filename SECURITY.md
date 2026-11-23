# Security Policy

## Supported versions

Security fixes are applied to the `main` branch. There is no published release yet; when one is
made, fixes will go to the latest release as well.

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Report privately in one of these ways:
- Use the **"Report a vulnerability"** button under the repository's
  **Security** tab (GitHub private reporting).
- Or email the maintainer, Kushagra Srivastava, at kushagrathisside@gmail.com

Helpful details to include:
- What the problem is and where it is (file, feature, or command)
- Steps to reproduce it
- The impact you think it could have

## What to expect

- We will acknowledge your report within 7 days.
- We will keep you updated while we investigate and fix it.
- Please give us reasonable time to fix the issue before
  disclosing it publicly.

## Scope

Examples of what is in scope: loading untrusted planner files,
handling of submitted match logs, and the local web server.

Known and documented, so not a vulnerability by itself: the training checkpoint
(`checkpoint.pkl`) is a Python pickle file, and loading a pickle runs code. `train --resume`
should only be used with a checkpoint you wrote yourself. Planner files are not pickles and
are loaded without executing anything.
