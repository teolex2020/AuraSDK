## What

Brief description of changes.

## Why

What problem this solves or what feature it adds.

## How

Key implementation details (if non-obvious).

## Testing

- [ ] Rust tests pass: `cargo test --no-default-features --features "encryption,audit"`
- [ ] Clippy clean: `cargo clippy --no-default-features --features "encryption,audit" -- -D warnings`
- [ ] Python tests pass: `pytest tests/ -v`
- [ ] New tests added for new functionality

## Checklist

- [ ] Code follows project style (rustfmt, no clippy warnings)
- [ ] No new dependencies added (or justified in description)
- [ ] Backward compatible (or breaking change documented)
- [ ] Docs updated if API changed
