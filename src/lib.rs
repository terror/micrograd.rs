use {
  graphviz_rust::{
    cmd::Format,
    dot_structures::{
      Attribute, Edge, EdgeTy, Graph, Id, Node, NodeId, Stmt, Vertex,
    },
    exec,
    printer::{DotPrinter, PrinterContext},
  },
  ordered_float::NotNan,
  std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::{self, Display, Formatter},
    fs,
    hash::{Hash, Hasher},
    io,
    ops::{Add, Div, Mul, Neg, Sub},
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
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
    write!(
      f,
      "{}",
      match self {
        Self::Add => "+",
        Self::Sub => "-",
        Self::Mul => "×",
        Self::Div => "÷",
        Self::Neg => "-",
        Self::Input => "",
        Self::Tanh => "tanh",
      }
    )
  }
}

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct Value {
  id: usize,
  data: NotNan<f64>,
  operation: Operation,
  children: HashSet<Value>,
}

impl Display for Value {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    let g: Graph = self.clone().into();
    let mut ctx = PrinterContext::default();
    write!(f, "{}\n", g.print(&mut ctx))
  }
}

impl PartialEq for Value {
  fn eq(&self, other: &Self) -> bool {
    self.data == other.data
  }
}

impl Eq for Value {}

impl Hash for Value {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.id.hash(state);
  }
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

impl Into<Graph> for Value {
  fn into(self) -> Graph {
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();

    let mut nodes = Vec::new();
    let mut op_nodes = Vec::new();

    queue.push_back(self.clone());

    while let Some(cur) = queue.pop_front() {
      if visited.insert(cur.id) {
        nodes.push(cur.clone());

        if cur.operation != Operation::Input {
          op_nodes.push(cur.clone());
        }

        for child in &cur.children {
          queue.push_back(child.clone());
        }
      }
    }

    nodes.sort_by_key(|v| v.id);
    op_nodes.sort_by_key(|v| v.id);

    let mut ids = HashMap::new();
    let mut op_ids = HashMap::new();

    for (i, node) in nodes.iter().enumerate() {
      ids.insert(node.id, i);
    }

    for (i, node) in op_nodes.iter().enumerate() {
      op_ids.insert(node.id, i);
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
      let op_i = op_ids[&node.id];
      let node_i = ids[&node.id];

      edges.push(Edge {
        ty: EdgeTy::Pair(
          Vertex::N(NodeId(Id::Plain(format!("op{}", op_i)), None)),
          Vertex::N(NodeId(Id::Plain(format!("node{}", node_i)), None)),
        ),
        attributes: vec![],
      });

      let mut sorted_children: Vec<_> = node.children.iter().collect();
      sorted_children.sort_by_key(|child| child.id);

      for child in sorted_children {
        let ci = ids[&child.id];

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
}

impl Value {
  pub fn new(data: f64) -> Result<Self, ValueError> {
    Ok(Self {
      id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
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
      id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
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
      id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
      data: NotNan::new(result).map_err(ValueError::from)?,
      operation,
      children: HashSet::new(),
    };

    new_value.children.insert(val);

    Ok(new_value)
  }

  pub fn graph(self, path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();

    let format = path
      .extension()
      .and_then(|ext| ext.to_str())
      .and_then(|ext| match ext.to_lowercase().as_str() {
        "png" => Some(Format::Png),
        "svg" => Some(Format::Svg),
        "pdf" => Some(Format::Pdf),
        "jpg" | "jpeg" => Some(Format::Jpg),
        "gif" => Some(Format::Gif),
        "dot" => Some(Format::Dot),
        _ => None,
      })
      .ok_or_else(|| {
        io::Error::new(
          io::ErrorKind::InvalidInput,
          "Unsupported or missing file extension. Supported formats: png, svg, pdf, jpg/jpeg, gif, dot"
        )
      })?;

    fs::write(
      path,
      exec(
        self.into(),
        &mut PrinterContext::default(),
        vec![format.into()],
      )?,
    )
  }
}

#[cfg(test)]
mod tests {
  use {super::*, indoc::indoc, tempdir::TempDir};

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

    let dot = sum.to_string();

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

    let dot = result.to_string();

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

  #[test]
  fn png_output() -> io::Result<()> {
    let dir = TempDir::new("test")?;
    let file_path = dir.path().join("test.png");

    let a = Value::new(2.0).unwrap();
    let b = Value::new(3.0).unwrap();
    let c = (a + b).unwrap();

    c.graph(&file_path)?;

    let metadata = fs::metadata(&file_path)?;
    assert!(metadata.len() > 0);

    Ok(())
  }

  #[test]
  fn svg_output() -> io::Result<()> {
    let dir = TempDir::new("test")?;
    let file_path = dir.path().join("test.svg");

    let a = Value::new(2.0).unwrap();
    let b = Value::new(3.0).unwrap();
    let c = (a + b).unwrap();

    c.graph(&file_path)?;

    let metadata = fs::metadata(&file_path)?;
    assert!(metadata.len() > 0);

    let content = fs::read_to_string(&file_path)?;
    assert!(content.contains("<svg"));
    assert!(content.contains("</svg>"));

    Ok(())
  }

  #[test]
  fn dot_output() -> io::Result<()> {
    let dir = TempDir::new("test")?;
    let file_path = dir.path().join("test.dot");

    let a = Value::new(2.0).unwrap();
    let b = Value::new(3.0).unwrap();
    let c = (a + b).unwrap();

    c.graph(&file_path)?;

    let metadata = fs::metadata(&file_path)?;
    assert!(metadata.len() > 0);

    let content = fs::read_to_string(&file_path)?;
    assert!(content.contains("digraph G {"));
    assert!(content.contains("}"));

    Ok(())
  }

  #[test]
  fn invalid_extension() {
    let a = Value::new(1.0).unwrap();

    let result = a.graph("test.invalid");

    assert!(result.is_err());

    if let Err(e) = result {
      assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
      assert!(e
        .to_string()
        .contains("Unsupported or missing file extension"));
    }
  }

  #[test]
  fn missing_extension() {
    let a = Value::new(1.0).unwrap();

    let result = a.graph("test");

    assert!(result.is_err());

    if let Err(e) = result {
      assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
      assert!(e
        .to_string()
        .contains("Unsupported or missing file extension"));
    }
  }

  #[test]
  fn empty_filename() {
    let a = Value::new(1.0).unwrap();

    let result = a.graph("");

    assert!(result.is_err());

    if let Err(e) = result {
      assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
      assert!(e
        .to_string()
        .contains("Unsupported or missing file extension"));
    }
  }
}
