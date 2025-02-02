use {
  graphviz_rust::{
    dot_structures::{
      Attribute, Edge, EdgeTy, Graph, Id, Node, NodeId, Stmt, Vertex,
    },
    printer::{DotPrinter, PrinterContext},
  },
  ordered_float::NotNan,
  std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::{self, Display, Formatter},
    hash::{Hash, Hasher},
    ops::{Add, Div, Mul, Neg, Sub},
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
  Tanh,
}

impl Display for Operation {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    use Operation::*;

    write!(
      f,
      "{}",
      match self {
        Add => "+",
        Sub => "-",
        Mul => "×",
        Div => "÷",
        Neg => "-",
        Input => "",
        Tanh => "tanh",
      }
    )
  }
}

#[derive(Debug, Clone)]
pub struct Value {
  data: NotNan<f64>,
  operation: Operation,
  children: HashSet<Value>,
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

  pub fn children(&self) -> &HashSet<Value> {
    &self.children
  }

  fn binary_op(
    left: Self,
    right: Self,
    op_fn: impl FnOnce(f64, f64) -> f64,
    operation: Operation,
  ) -> Result<Self, ValueError> {
    let result = op_fn(left.data.into_inner(), right.data.into_inner());

    if result.is_infinite() {
      return Err(ValueError::Overflow);
    }

    let mut new_value = Self {
      data: NotNan::new(result).map_err(ValueError::from)?,
      operation,
      children: HashSet::new(),
    };

    new_value.children.insert(left);
    new_value.children.insert(right);

    Ok(new_value)
  }

  fn unary_op(
    val: Self,
    op_fn: impl FnOnce(f64) -> f64,
    operation: Operation,
  ) -> Result<Self, ValueError> {
    let result = op_fn(val.data.into_inner());

    if result.is_infinite() {
      return Err(ValueError::Overflow);
    }

    let mut new_value = Self {
      data: NotNan::new(result).map_err(ValueError::from)?,
      operation,
      children: HashSet::new(),
    };

    new_value.children.insert(val);

    Ok(new_value)
  }

  pub fn to_graphviz_graph(&self) -> Graph {
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();

    let mut nodes = Vec::new();
    let mut op_nodes = Vec::new();

    queue.push_back(self.clone());

    while let Some(cur) = queue.pop_front() {
      if visited.insert(cur.data) {
        nodes.push(cur.clone());

        if cur.operation != Operation::Input {
          op_nodes.push(cur.clone());
        }

        for child in &cur.children {
          queue.push_back(child.clone());
        }
      }
    }

    nodes.sort_by(|a, b| a.data.partial_cmp(&b.data).unwrap());
    op_nodes.sort_by(|a, b| a.data.partial_cmp(&b.data).unwrap());

    let mut ids = HashMap::new();
    let mut op_ids = HashMap::new();

    for (i, node) in nodes.iter().enumerate() {
      ids.insert(node.data, i);
    }

    for (i, node) in op_nodes.iter().enumerate() {
      op_ids.insert(node.data, i);
    }

    let mut stmts = Vec::new();

    for (i, node) in nodes.iter().enumerate() {
      let label_attr = Attribute(
        Id::Plain("label".to_string()),
        Id::Plain(format!("\"{}\"", node.get())),
      );

      let n = Node {
        id: NodeId(Id::Plain(format!("node{}", i)), None),
        attributes: vec![label_attr],
      };

      stmts.push(Stmt::Node(n));
    }

    for (i, node) in op_nodes.iter().enumerate() {
      let label_attr = Attribute(
        Id::Plain("label".to_string()),
        Id::Plain(format!("\"{}\"", node.operation.to_string())),
      );

      let n = Node {
        id: NodeId(Id::Plain(format!("op{}", i)), None),
        attributes: vec![label_attr],
      };

      stmts.push(Stmt::Node(n));
    }

    let mut edges = Vec::new();

    for node in &op_nodes {
      let op_i = op_ids[&node.data];
      let node_i = ids[&node.data];

      edges.push(Edge {
        ty: EdgeTy::Pair(
          Vertex::N(NodeId(Id::Plain(format!("op{}", op_i)), None)),
          Vertex::N(NodeId(Id::Plain(format!("node{}", node_i)), None)),
        ),
        attributes: vec![],
      });

      let mut sorted_children: Vec<_> = node.children.iter().collect();
      sorted_children.sort_by(|a, b| a.data.partial_cmp(&b.data).unwrap());

      for child in sorted_children {
        let ci = ids[&child.data];

        edges.push(Edge {
          ty: EdgeTy::Pair(
            Vertex::N(NodeId(Id::Plain(format!("node{}", ci)), None)),
            Vertex::N(NodeId(Id::Plain(format!("op{}", op_i)), None)),
          ),
          attributes: vec![],
        });
      }
    }

    edges.sort_by(|a, b| {
      let a_str = format!("{:?}", a);
      let b_str = format!("{:?}", b);
      a_str.cmp(&b_str)
    });

    stmts.extend(edges.into_iter().map(Stmt::Edge));

    Graph::DiGraph {
      id: Id::Plain("G".to_string()),
      strict: false,
      stmts,
    }
  }

  pub fn to_dot_string(&self) -> String {
    let g = self.to_graphviz_graph();
    let mut ctx = PrinterContext::default();
    format!("{}\n", g.print(&mut ctx))
  }
}

macro_rules! impl_value_binary_op {
  ($trait_name:ident, $func_name:ident, $op_symbol:tt, $op_variant:expr, $check:expr) => {
    impl $trait_name for Value {
      type Output = Result<Value, ValueError>;

      fn $func_name(self, rhs: Self) -> Self::Output {
        $check(&rhs)?;
        Value::binary_op(self, rhs, |a, b| a $op_symbol b, $op_variant)
      }
    }

    impl $trait_name for &Value {
      type Output = Result<Value, ValueError>;

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
      type Output = Result<Value, ValueError>;

      fn $func_name(self) -> Self::Output {
        Value::unary_op(self, |a| $op_symbol a, $op_variant)
      }
    }

    impl $trait_name for &Value {
      type Output = Result<Value, ValueError>;

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

pub trait Tanh {
  type Output;

  fn tanh(self) -> Self::Output;
}

impl Tanh for Value {
  type Output = Result<Value, ValueError>;

  fn tanh(self) -> Self::Output {
    Value::unary_op(self, |x| x.tanh(), Operation::Tanh)
  }
}

impl Tanh for &Value {
  type Output = Result<Value, ValueError>;

  fn tanh(self) -> Self::Output {
    Value::unary_op(self.clone(), |x| x.tanh(), Operation::Tanh)
  }
}

#[cfg(test)]
mod tests {
  use {super::*, indoc::indoc};

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
    assert_eq!(v1 + v2, Ok(Value::new(5.0).unwrap()));
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
    assert_eq!(v1 - v2, Ok(Value::new(2.0).unwrap()));
  }

  #[test]
  fn multiplication() {
    let v1 = Value::new(2.0).unwrap();
    let v2 = Value::new(3.0).unwrap();
    assert_eq!(v1 * v2, Ok(Value::new(6.0).unwrap()));
  }

  #[test]
  fn division() {
    let v1 = Value::new(6.0).unwrap();
    let v2 = Value::new(2.0).unwrap();
    assert_eq!(v1 / v2, Ok(Value::new(3.0).unwrap()));
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
    assert_eq!(-v, Ok(Value::new(-42.0).unwrap()));
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
    assert_eq!(&v1 + &v2, Ok(Value::new(5.0).unwrap()));
    assert_eq!(&v1 - &v2, Ok(Value::new(-1.0).unwrap()));
    assert_eq!(&v1 * &v2, Ok(Value::new(6.0).unwrap()));
    assert_eq!(&v1 / &v2, Ok(Value::new(2.0 / 3.0).unwrap()));
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
    let result = (sum.clone() * c).unwrap();

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

  #[test]
  fn dot_generation_test() {
    let a = Value::new(2.0).unwrap();
    let b = Value::new(3.0).unwrap();

    let sum = (a + b).unwrap();

    let dot = sum.to_dot_string();

    pretty_assertions::assert_eq!(
      dot,
      indoc! {"
        digraph G {
          node0[label=\"2\"]
          node1[label=\"3\"]
          node2[label=\"5\"]
          op0[label=\"+\"]
          node0 -> op0
          node1 -> op0
          op0 -> node2
        }
      "}
    );
  }

  #[test]
  fn complex_graph_test() {
    let a = Value::new(2.0).unwrap();
    let b = Value::new(3.0).unwrap();
    let c = Value::new(4.0).unwrap();

    let sum = (a + b).unwrap();
    let result = ((sum).clone() * c).unwrap();

    let dot = result.to_dot_string();

    pretty_assertions::assert_eq!(
      dot,
      indoc! {"
        digraph G {
          node0[label=\"2\"]
          node1[label=\"3\"]
          node2[label=\"4\"]
          node3[label=\"5\"]
          node4[label=\"20\"]
          op0[label=\"+\"]
          op1[label=\"×\"]
          node0 -> op0
          node1 -> op0
          node2 -> op1
          node3 -> op1
          op0 -> node3
          op1 -> node4
        }
      "}
    );
  }

  #[test]
  fn tanh_operation() {
    let v = Value::new(0.0).unwrap();
    let result = v.tanh().unwrap();
    assert_eq!(result.operation, Operation::Tanh);
    assert_eq!(result.get(), 0.0);

    let v = Value::new(1.0).unwrap();
    let result = v.tanh().unwrap();
    assert_eq!(result.get(), 0.7615941559557649); // tanh(1)

    let v = Value::new(-2.0).unwrap();
    let result = v.tanh().unwrap();
    assert_eq!(result.get(), -0.9640275800758169); // tanh(-2)
  }

  #[test]
  fn tanh_with_children() {
    let v = Value::new(1.0).unwrap();
    let result = v.tanh().unwrap();
    assert_eq!(result.children().len(), 1);
    assert_eq!(result.children().iter().next().unwrap().get(), 1.0);
  }

  #[test]
  fn reference_tanh() {
    let v = Value::new(1.0).unwrap();
    let result = (&v).tanh().unwrap();
    assert_eq!(result.get(), 0.7615941559557649);
  }
}
