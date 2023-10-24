use pyo3::exceptions::PySyntaxError;
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyDict};

use rustpython_parser::{Parse, ast};

#[pyclass]
pub struct Ast(ast::Stmt);

#[pymethods]
impl Ast {
    fn __repr__(&self) -> String {
        return format!("{:?}", self.0);
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Expr {

}

// An "unbound" expresssion which may contain free variables.
// Contains a stanza expression + a reference to the python scope
// for acquiring free variables. At the time the function
// object is created, the scope may not contain all the free variables
// needed, so the free variables can only be resolved at the time the function
// is called. The bind_free() method returns an Expr with the free variables bound.
#[pyclass]
pub struct Function {
    expr: Expr,
    scope: Py<PyDict>,
}

#[pymethods]
impl Function {
    #[new]
    fn new(expr: Expr, dict: &PyDict) -> PyResult<Self> {
        Ok(Function {expr, scope: dict.into()})
    }
}

#[pyclass]
pub struct Entrypoint {

}

#[pyclass]
pub struct DeviceType {

}

#[pyclass]
pub struct Type {

}

#[pyfunction]
pub fn parse(_py: Python, code: &str, source_path: &str, offset: u32) -> PyResult<Ast> {
    Ok(Ast(ast::Stmt::parse_starts_at(code, source_path, ast::TextSize::new(offset))
        .map_err(|_| PySyntaxError::new_err("Syntax error"))?))
}

#[pyfunction]
pub fn transpile(_py: Python, _code: &Ast) -> PyResult<Expr> {
    Ok(Expr {})
}

#[pyfunction]
pub fn compile(_py: Python, _expr: &Expr, _device_type: &DeviceType) -> PyResult<Entrypoint> {
    Ok(Entrypoint {})
}

#[pyfunction]
pub fn type_of(_py: Python, _obj: &PyAny) -> PyResult<Type> {
    Ok(Type {})
}

/// A Python module implemented in Rust.
#[pymodule]
pub fn _stanza(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Ast>()?;
    m.add_class::<Type>()?;
    m.add_class::<Expr>()?;
    m.add_class::<Function>()?;
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(transpile, m)?)?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    m.add_function(wrap_pyfunction!(type_of, m)?)?;
    Ok(())
}