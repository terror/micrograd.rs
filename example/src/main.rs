use micrograd::Value;

fn main() {
  let a = Value::new(2.0).unwrap();
  let b = Value::new(3.0).unwrap();
  let result = (a + b).unwrap();
  println!("2 + 3 = {}", result.get());
}
