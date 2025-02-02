use {
  ordered_float::NotNan,
  std::{
    collections::HashSet,
    hash::{Hash, Hasher},
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
  },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
  Input,
  Add,
  Sub,
  Mul,
  Div,
  Neg,
}

#[derive(Debug, Clone)]
pub struct Value {
  data: NotNan<f64>,
  operation: Operation,
  children: HashSet<Arc<Value>>,
}

impl PartialEq for Value {
  fn eq(&self, other: &Self) -> bool {
    self.data == other.data
  }
}

impl Eq for Value {}

impl Hash for Value {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.data.hash(state);
  }
}

impl Value {
  pub fn new(data: f64) -> Result<Self, ValueError> {
    Ok(Self {
      data: NotNan::new(data).map_err(ValueError::from)?,
      operation: Operation::Input,
      children: HashSet::new(),
    })
  }

  pub fn get(&self) -> f64 {
    self.data.into_inner()
  }

  pub fn children(&self) -> &HashSet<Arc<Value>> {
    &self.children
  }

  fn binary_op(
    left: Self,
    right: Self,
    op_fn: impl FnOnce(f64, f64) -> f64,
    operation: Operation,
  ) -> Result<Arc<Self>, ValueError> {
    let result = op_fn(left.data.into_inner(), right.data.into_inner());

    if result.is_infinite() {
      return Err(ValueError::Overflow);
    }

    let mut new_value = Self {
      data: NotNan::new(result).map_err(ValueError::from)?,
      operation,
      children: HashSet::new(),
    };

    new_value.children.insert(Arc::new(left));
    new_value.children.insert(Arc::new(right));

    Ok(Arc::new(new_value))
  }

  fn unary_op(
    val: Self,
    op_fn: impl FnOnce(f64) -> f64,
    operation: Operation,
  ) -> Result<Arc<Self>, ValueError> {
    let result = op_fn(val.data.into_inner());

    if result.is_infinite() {
      return Err(ValueError::Overflow);
    }

    let mut new_value = Self {
      data: NotNan::new(result).map_err(ValueError::from)?,
      operation,
      children: HashSet::new(),
    };

    new_value.children.insert(Arc::new(val));

    Ok(Arc::new(new_value))
  }
}

macro_rules! impl_value_binary_op {
  ($trait_name:ident, $func_name:ident, $op_symbol:tt, $op_variant:expr, $check:expr) => {
    impl $trait_name for Value {
      type Output = Result<Arc<Value>, ValueError>;

      fn $func_name(self, rhs: Self) -> Self::Output {
        $check(&rhs)?;
        Value::binary_op(self, rhs, |a, b| a $op_symbol b, $op_variant)
      }
    }

    impl $trait_name for &Value {
      type Output = Result<Arc<Value>, ValueError>;

      fn $func_name(self, rhs: Self) -> Self::Output {
        $check(rhs)?;
        Value::binary_op(self.clone(), rhs.clone(), |a, b| a $op_symbol b, $op_variant)
      }
    }
  };
}

macro_rules! impl_value_unary_op {
  ($trait_name:ident, $func_name:ident, $op_symbol:tt, $op_variant:expr) => {
    impl $trait_name for Value {
      type Output = Result<Arc<Value>, ValueError>;

      fn $func_name(self) -> Self::Output {
        Value::unary_op(self, |a| $op_symbol a, $op_variant)
      }
    }

    impl $trait_name for &Value {
      type Output = Result<Arc<Value>, ValueError>;

      fn $func_name(self) -> Self::Output {
        Value::unary_op(self.clone(), |a| $op_symbol a, $op_variant)
      }
    }
  };
}

impl_value_binary_op!(Add, add, +, Operation::Add, |_rhs: &Value| -> Result<(), ValueError> { Ok(()) });
impl_value_binary_op!(Sub, sub, -, Operation::Sub, |_rhs: &Value| -> Result<(), ValueError> { Ok(()) });
impl_value_binary_op!(Mul, mul, *, Operation::Mul, |_rhs: &Value| -> Result<(), ValueError> { Ok(()) });

impl_value_binary_op!(
  Div,
  div,
  /,
  Operation::Div,
  |rhs: &Value| -> Result<(), ValueError> {
    if rhs.data.into_inner() == 0.0 {
      Err(ValueError::DivisionByZero)
    } else {
      Ok(())
    }
  }
);

impl_value_unary_op!(Neg, neg, -, Operation::Neg);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn new() {
    let v = Value::new(42.0).unwrap();
    assert_eq!(v.get(), 42.0);
  }

  #[test]
  fn nan_error() {
    assert_eq!(Value::new(f64::NAN), Err(ValueError::NaN));
  }

  #[test]
  fn addition() {
    let v1 = Value::new(2.0).unwrap();
    let v2 = Value::new(3.0).unwrap();
    assert_eq!(v1 + v2, Ok(Arc::new(Value::new(5.0).unwrap())));
  }

  #[test]
  fn addition_overflow() {
    let v1 = Value::new(f64::MAX).unwrap();
    let v2 = Value::new(f64::MAX).unwrap();
    assert_eq!(v1 + v2, Err(ValueError::Overflow));
  }

  #[test]
  fn subtraction() {
    let v1 = Value::new(5.0).unwrap();
    let v2 = Value::new(3.0).unwrap();
    assert_eq!(v1 - v2, Ok(Arc::new(Value::new(2.0).unwrap())));
  }

  #[test]
  fn multiplication() {
    let v1 = Value::new(2.0).unwrap();
    let v2 = Value::new(3.0).unwrap();
    assert_eq!(v1 * v2, Ok(Arc::new(Value::new(6.0).unwrap())));
  }

  #[test]
  fn division() {
    let v1 = Value::new(6.0).unwrap();
    let v2 = Value::new(2.0).unwrap();
    assert_eq!(v1 / v2, Ok(Arc::new(Value::new(3.0).unwrap())));
  }

  #[test]
  fn division_by_zero() {
    let v1 = Value::new(6.0).unwrap();
    let v2 = Value::new(0.0).unwrap();
    assert_eq!(v1 / v2, Err(ValueError::DivisionByZero));
  }

  #[test]
  fn negation() {
    let v = Value::new(42.0).unwrap();
    assert_eq!(-v, Ok(Arc::new(Value::new(-42.0).unwrap())));
  }

  #[test]
  fn error_display() {
    assert_eq!(ValueError::DivisionByZero.to_string(), "division by zero");

    assert_eq!(ValueError::NaN.to_string(), "not a number (NaN)");

    assert_eq!(
      ValueError::Overflow.to_string(),
      "operation resulted in overflow"
    );
  }

  #[test]
  fn reference_operations() {
    let v1 = Value::new(2.0).unwrap();
    let v2 = Value::new(3.0).unwrap();
    assert_eq!(&v1 + &v2, Ok(Arc::new(Value::new(5.0).unwrap())));
    assert_eq!(&v1 - &v2, Ok(Arc::new(Value::new(-1.0).unwrap())));
    assert_eq!(&v1 * &v2, Ok(Arc::new(Value::new(6.0).unwrap())));
    assert_eq!(&v1 / &v2, Ok(Arc::new(Value::new(2.0 / 3.0).unwrap())));
  }

  #[test]
  fn addition_with_children() {
    let v1 = Value::new(2.0).unwrap();
    let v2 = Value::new(3.0).unwrap();
    let result = (v1 + v2).unwrap();

    assert_eq!(result.get(), 5.0);
    assert_eq!(result.children().len(), 2);

    let children_values: Vec<f64> =
      result.children().iter().map(|child| child.get()).collect();

    assert!(children_values.contains(&2.0));
    assert!(children_values.contains(&3.0));
  }

  #[test]
  fn multiplication_with_children() {
    let v1 = Value::new(2.0).unwrap();
    let v2 = Value::new(3.0).unwrap();
    let result = (v1 * v2).unwrap();

    assert_eq!(result.get(), 6.0);
    assert_eq!(result.children().len(), 2);

    let children_values: Vec<f64> =
      result.children().iter().map(|child| child.get()).collect();

    assert!(children_values.contains(&2.0));
    assert!(children_values.contains(&3.0));
  }

  #[test]
  fn negation_with_children() {
    let v = Value::new(42.0).unwrap();
    let result = (-v).unwrap();
    assert_eq!(result.get(), -42.0);
    assert_eq!(result.children().len(), 1);
    assert_eq!(result.children().iter().next().unwrap().get(), 42.0);
  }

  #[test]
  fn complex_expression_with_children() {
    let a = Value::new(2.0).unwrap();
    let b = Value::new(3.0).unwrap();
    let c = Value::new(4.0).unwrap();

    let sum = (a + b).unwrap();
    let result = ((*sum).clone() * c).unwrap();

    assert_eq!(result.get(), 20.0);
    assert_eq!(result.children().len(), 2);

    let children_values: Vec<f64> =
      result.children().iter().map(|child| child.get()).collect();

    assert!(children_values.contains(&5.0));
    assert!(children_values.contains(&4.0));

    let sum_child = result
      .children()
      .iter()
      .find(|child| child.get() == 5.0)
      .unwrap();

    assert_eq!(sum_child.children().len(), 2);
  }

  #[test]
  fn operation_for_input() {
    let v = Value::new(1.0).unwrap();
    assert_eq!(v.operation, Operation::Input);
  }

  #[test]
  fn operation_for_add() {
    let v1 = Value::new(1.0).unwrap();
    let v2 = Value::new(2.0).unwrap();
    let result = (v1 + v2).unwrap();
    assert_eq!(result.operation, Operation::Add);
  }

  #[test]
  fn operation_for_sub() {
    let v1 = Value::new(3.0).unwrap();
    let v2 = Value::new(1.0).unwrap();
    let result = (v1 - v2).unwrap();
    assert_eq!(result.operation, Operation::Sub);
  }

  #[test]
  fn operation_for_mul() {
    let v1 = Value::new(2.0).unwrap();
    let v2 = Value::new(4.0).unwrap();
    let result = (v1 * v2).unwrap();
    assert_eq!(result.operation, Operation::Mul);
  }

  #[test]
  fn operation_for_div() {
    let v1 = Value::new(6.0).unwrap();
    let v2 = Value::new(2.0).unwrap();
    let result = (v1 / v2).unwrap();
    assert_eq!(result.operation, Operation::Div);
  }

  #[test]
  fn operation_for_neg() {
    let v1 = Value::new(5.0).unwrap();
    let result = -v1;
    assert_eq!(result.unwrap().operation, Operation::Neg);
  }
}
