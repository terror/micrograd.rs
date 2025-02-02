use {
  graphviz_rust::{cmd::Format, exec, printer::PrinterContext},
  micrograd::Value,
  std::{fs, io},
};

fn main() -> io::Result<()> {
  let a = Value::new(2.0).unwrap();
  let b = Value::new(-3.0).unwrap();
  let c = Value::new(10.0).unwrap();
  let d = ((a * b).unwrap() + c).unwrap();

  let graph = d.to_graphviz_graph();

  let mut ctx = PrinterContext::default();

  let png_bytes = exec(graph, &mut ctx, vec![Format::Png.into()])?;

  fs::write("expression.png", png_bytes)?;

  println!("Wrote expression.png");

  Ok(())
}
