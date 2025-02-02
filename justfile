set dotenv-load

alias f := fmt
alias t := test

export EDITOR := 'nvim'

default:
  just --list

fmt:
  cargo fmt

test:
  cargo test
