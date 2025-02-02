use {
  ordered_float::NotNan,
  std::ops::{Add, Div, Mul, Neg, Sub},
  thiserror::Error,
};

#[derive(Debug, Error, PartialEq)]
pub enum ValueError {
  #[error("division by zero")]
  DivisionByZero,
  #[error("not a number (NaN)")]
  NaN,
  #[error("operation resulted in overflow")]
  Overflow,
}

impl From<ordered_float::FloatIsNan> for ValueError {
  fn from(_: ordered_float::FloatIsNan) -> Self {
    ValueError::NaN
  }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Value {
  data: NotNan<f64>,
}

impl Value {
  pub fn new(data: f64) -> Result<Self, ValueError> {
    Ok(Self {
      data: NotNan::new(data).map_err(ValueError::from)?,
    })
  }

  pub fn get(&self) -> f64 {
    self.data.into_inner()
  }
}

impl Add for Value {
  type Output = Result<Self, ValueError>;

  fn add(self, other: Self) -> Result<Self, ValueError> {
    let result = self.data.into_inner() + other.data.into_inner();

    if result.is_infinite() {
      return Err(ValueError::Overflow);
    }

    Value::new(result)
  }
}

impl Sub for Value {
  type Output = Result<Self, ValueError>;

  fn sub(self, other: Self) -> Result<Self, ValueError> {
    let result = self.data.into_inner() - other.data.into_inner();

    if result.is_infinite() {
      return Err(ValueError::Overflow);
    }

    Value::new(result)
  }
}

impl Mul for Value {
  type Output = Result<Self, ValueError>;

  fn mul(self, other: Self) -> Result<Self, ValueError> {
    let result = self.data.into_inner() * other.data.into_inner();

    if result.is_infinite() {
      return Err(ValueError::Overflow);
    }

    Value::new(result)
  }
}

impl Div for Value {
  type Output = Result<Self, ValueError>;

  fn div(self, other: Self) -> Result<Self, ValueError> {
    if other.data.into_inner() == 0.0 {
      return Err(ValueError::DivisionByZero);
    }

    let result = self.data.into_inner() / other.data.into_inner();

    if result.is_infinite() {
      return Err(ValueError::Overflow);
    }

    Value::new(result)
  }
}

impl Neg for Value {
  type Output = Result<Self, ValueError>;

  fn neg(self) -> Result<Self, ValueError> {
    Value::new(-self.data.into_inner())
  }
}

impl Add for &Value {
  type Output = Result<Value, ValueError>;

  fn add(self, other: Self) -> Result<Value, ValueError> {
    let result = self.data.into_inner() + other.data.into_inner();

    if result.is_infinite() {
      return Err(ValueError::Overflow);
    }

    Value::new(result)
  }
}

impl Sub for &Value {
  type Output = Result<Value, ValueError>;

  fn sub(self, other: Self) -> Result<Value, ValueError> {
    let result = self.data.into_inner() - other.data.into_inner();

    if result.is_infinite() {
      return Err(ValueError::Overflow);
    }

    Value::new(result)
  }
}

impl Mul for &Value {
  type Output = Result<Value, ValueError>;

  fn mul(self, other: Self) -> Result<Value, ValueError> {
    let result = self.data.into_inner() * other.data.into_inner();

    if result.is_infinite() {
      return Err(ValueError::Overflow);
    }

    Value::new(result)
  }
}

impl Div for &Value {
  type Output = Result<Value, ValueError>;

  fn div(self, other: Self) -> Result<Value, ValueError> {
    if other.data.into_inner() == 0.0 {
      return Err(ValueError::DivisionByZero);
    }

    let result = self.data.into_inner() / other.data.into_inner();

    if result.is_infinite() {
      return Err(ValueError::Overflow);
    }

    Value::new(result)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_new() {
    let v = Value::new(42.0).unwrap();
    assert_eq!(v.get(), 42.0);
  }

  #[test]
  fn test_nan_error() {
    assert_eq!(Value::new(f64::NAN), Err(ValueError::NaN));
  }

  #[test]
  fn test_addition() {
    let v1 = Value::new(2.0).unwrap();
    let v2 = Value::new(3.0).unwrap();
    assert_eq!(v1 + v2, Ok(Value::new(5.0).unwrap()));
  }

  #[test]
  fn test_addition_overflow() {
    let v1 = Value::new(f64::MAX).unwrap();
    let v2 = Value::new(f64::MAX).unwrap();
    assert_eq!(v1 + v2, Err(ValueError::Overflow));
  }

  #[test]
  fn test_subtraction() {
    let v1 = Value::new(5.0).unwrap();
    let v2 = Value::new(3.0).unwrap();
    assert_eq!(v1 - v2, Ok(Value::new(2.0).unwrap()));
  }

  #[test]
  fn test_multiplication() {
    let v1 = Value::new(2.0).unwrap();
    let v2 = Value::new(3.0).unwrap();
    assert_eq!(v1 * v2, Ok(Value::new(6.0).unwrap()));
  }

  #[test]
  fn test_division() {
    let v1 = Value::new(6.0).unwrap();
    let v2 = Value::new(2.0).unwrap();
    assert_eq!(v1 / v2, Ok(Value::new(3.0).unwrap()));
  }

  #[test]
  fn test_division_by_zero() {
    let v1 = Value::new(6.0).unwrap();
    let v2 = Value::new(0.0).unwrap();
    assert_eq!(v1 / v2, Err(ValueError::DivisionByZero));
  }

  #[test]
  fn test_negation() {
    let v = Value::new(42.0).unwrap();
    assert_eq!(-v, Ok(Value::new(-42.0).unwrap()));
  }

  #[test]
  fn test_error_display() {
    assert_eq!(ValueError::DivisionByZero.to_string(), "division by zero");

    assert_eq!(ValueError::NaN.to_string(), "not a number (NaN)");

    assert_eq!(
      ValueError::Overflow.to_string(),
      "operation resulted in overflow"
    );
  }

  #[test]
  fn test_reference_operations() {
    let v1 = Value::new(2.0).unwrap();
    let v2 = Value::new(3.0).unwrap();
    assert_eq!(&v1 + &v2, Ok(Value::new(5.0).unwrap()));
    assert_eq!(&v1 - &v2, Ok(Value::new(-1.0).unwrap()));
    assert_eq!(&v1 * &v2, Ok(Value::new(6.0).unwrap()));
    assert_eq!(&v1 / &v2, Ok(Value::new(2.0 / 3.0).unwrap()));
  }
}
