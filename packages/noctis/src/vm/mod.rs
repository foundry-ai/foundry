pub mod ast;

use ordered_float::OrderedFloat;

// A type consists of an identifier
// and a uuid
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Type(pub String, pub Option<String>);

#[derive(Debug,Hash,Eq,PartialEq,Clone)]
pub enum Constant {
    Integer(i64),
    Float(OrderedFloat<f64>),
    Bool(bool),
    String(String),
    Unit
}



// Display implementations

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(ref s) = self.1 {
            write!(f, "{0}#{1}", self.0, s)
        } else {
            write!(f, "{0}", self.0)
        }
    }
}

impl std::fmt::Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Constant::Integer(i) => write!(f, "{}", i),
            Constant::Float(fl) => write!(f, "{}", fl),
            Constant::Bool(b) => write!(f, "{}", b),
            Constant::String(s) => write!(f, "{}", s),
            Constant::Unit => write!(f, "()")
        }
    }
}