# Server synchronization without SSH or GitHub CLI

The GitHub repository is public. The compute server needs only `git` and HTTPS;
it does not need an SSH key, GitHub CLI, sudo, or a GitHub token.

## First download

```bash
cd ~/Documents/Projects
git clone https://github.com/C2H5O/endo3reval.git
cd endo3reval
cp configs/scared.example.json configs/scared.local.json
```

Edit `configs/scared.local.json` on the server. This file is ignored by Git, so
pulls will not overwrite the server-specific dataset, environment, repository,
or checkpoint paths.

## Later updates

Stop any process that is running code from the checkout, then run:

```bash
cd ~/Documents/Projects/endo3reval
git status --short
git pull --ff-only origin main
```

`git pull --ff-only` is compatible with older Git versions that do not provide
`git switch`. If `git status` reports local edits to tracked source files, do
not discard them blindly; copy or commit them before pulling. Normal outputs,
weights, external repositories, and `configs/*.local.json` are ignored and do
not block a pull.

## Optional source archive fallback

If outbound Git HTTPS is also blocked, download the repository ZIP on the local
Windows machine from GitHub, transfer it through the server's permitted file
channel, and extract it to a new directory. Git-based incremental updates will
not be available in that fallback.
