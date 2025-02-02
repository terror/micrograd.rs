set dotenv-load

alias e := example
alias f := fmt
alias t := test

export EDITOR := 'nvim'

default:
  just --list

example:
  cargo run --manifest-path example/Cargo.toml

fmt:
  cargo fmt

test:
  cargo test
