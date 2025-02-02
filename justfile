set dotenv-load

alias t := test

export EDITOR := 'nvim'

default:
  just --list

test:
  cargo test
