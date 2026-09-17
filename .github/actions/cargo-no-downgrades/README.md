# cargo-no-downgrades

Fails a pull request whose `Cargo.lock` pins any package below the version
the base branch locks.

The usual way a downgrade reaches `main`: a branch is cut, `main` bumps a
dependency, the branch is merged with its older pin and lock entry intact.
Cargo does not object, because the older version still satisfies the
branch's requirement.

```yaml
jobs:
  no-downgrades:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/cargo-no-downgrades
```

Inputs: `base` (defaults to the pull request's base branch) and `lockfile`
(defaults to `Cargo.lock`). Only packages the base already locks are
compared, by name and source, so a new dependency or a removed one never
trips it. Pre-releases sort below their release; build metadata is ignored.
It needs `python3` on the runner and nothing else.
