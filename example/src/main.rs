use {micrograd::Value, std::process};

fn run() -> Result {
  let a = Value::new(2.0)?;
  let b = Value::new(-3.0)?;
  let c = Value::new(10.0)?;
  let d = ((a * b)? + c)?;
  d.graph("expression.png")?;
  Ok(())
}

type Result<T = (), E = anyhow::Error> = std::result::Result<T, E>;

fn main() {
  if let Err(error) = run() {
    eprintln!("error: {error}");
    process::exit(1);
  }
}
