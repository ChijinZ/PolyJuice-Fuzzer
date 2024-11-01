use egg::{AstSize, EGraph, Extractor, Id, Language, RecExpr, Runner};
use log::Level::Trace;
use log::{error, info, log_enabled, trace};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::borrow::BorrowMut;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::ffi::CString;
use std::fmt::{Debug, Display, Formatter};
use std::time::{Duration, Instant};
use std::{env, ffi};

mod graph;
mod model;
mod rules;

use crate::graph::GraphConverter;
use crate::graph::TensorInfo;
use crate::model::{AnalysisOfCEIR, ComputExprIR};
use crate::rules::rules;

// type AnalysisOfCEIR = ();

const DEFAULT_HELPER_NODE_LIMIT: usize = 100;
const DEFAULT_HELPER_TIME_LIMIT: u64 = 3;
const DEFAULT_HELPER_ITER_LIMIT: usize = 2;

static mut ENV_LOGGER: Option<env_logger::Builder> = None;

type BuildHasher = fxhash::FxBuildHasher;
type IndexSet<K> = indexmap::IndexSet<K, BuildHasher>;

type ComputationGraphFromEgg = RecExpr<ComputExprIR>;

trait PrettyPrint {
    fn pretty_str(&self) -> String;
}

impl PrettyPrint for ComputationGraphFromEgg {
    fn pretty_str(&self) -> String {
        let mut vec: Vec<String> = vec![];
        for (index, ir) in self.as_ref().iter().enumerate() {
            vec.push(format!("{}: {:?}", index, ir));
        }
        return vec.join("\n");
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct BuildGraphError {
    error_op: String,
    reason: String,
}

impl Display for BuildGraphError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "error_op: {}\nreason: {}", self.error_op, self.reason)
    }
}

impl Debug for BuildGraphError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "error_op: {}\nreason: {}", self.error_op, self.reason)
    }
}

impl Error for BuildGraphError {}

impl BuildGraphError {
    pub fn new(error_op: String, reason: String) -> Self {
        BuildGraphError { error_op, reason }
    }
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct GraphFromExtern {
    pub insts: Vec<InstFromExtern>,
    pub vars: HashMap<String, VarFromExtern>,
    #[serde(skip)]
    pub converter: Option<GraphConverter>,
    #[serde(skip)]
    pub egraph_runner: Option<Runner<ComputExprIR, AnalysisOfCEIR>>,
}

#[derive(Default, Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct VarFromExtern {
    pub shape: Vec<usize>,
    // pub dimension: usize,
    pub dtype: String,
}

impl From<&TensorInfo> for VarFromExtern {
    fn from(tensor_info: &TensorInfo) -> Self {
        VarFromExtern {
            shape: tensor_info.shape.clone(),
            // dimension: tensor_info.shape.len(),
            dtype: tensor_info.dtype.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum OperatorFromExtern {
    Input {
        op_index: u64,
    },
    Constant {
        op_index: u64,
    },
    // --- Unary ops
    ReLU,
    // LeakyReLU,
    // PReLU,
    // GELU,
    Sigmoid,
    Tanh,
    Transpose {
        dim0: usize,
        dim1: usize,
    },
    // Conv2d {
    //     in_channels: usize,
    //     out_channels: usize,
    //     kernel_h_size: usize,
    //     kernel_w_size: usize,
    //     stride: usize,
    //     padding: usize,
    //     dilation_h: usize,
    //     dilation_w: usize,
    // },
    MaxPool2d {
        kh: usize,
        kw: usize,
        stride: usize,
        padding: usize,
    },
    AvgPool2d {
        kh: usize,
        kw: usize,
        stride: usize,
        padding: usize,
    },
    // ---
    // --- Binary ops
    Add,
    Mul,
    MatMul,
    Concat2 {
        axis: usize,
    },
    Split2 {
        axis: usize,
    },
    GraphOutput {
        var_name: String,
    },
    // --- others
    Other {
        op_index: u64,
    },
}

impl OperatorFromExtern {
    pub fn op_name(&self) -> String {
        let name = match self {
            OperatorFromExtern::Input { .. } => "Input",
            OperatorFromExtern::Constant { .. } => "Constant",
            OperatorFromExtern::ReLU => "ReLU",
            OperatorFromExtern::Sigmoid => "Sigmoid",
            OperatorFromExtern::Tanh => "Tanh",
            OperatorFromExtern::Transpose { .. } => "Transpose",
            OperatorFromExtern::MaxPool2d { .. } => "MaxPool2d",
            OperatorFromExtern::AvgPool2d { .. } => "AvgPool2d",
            OperatorFromExtern::Add => "Add",
            OperatorFromExtern::Mul => "Mul",
            OperatorFromExtern::MatMul => "MatMul",
            OperatorFromExtern::Concat2 { .. } => "Concat2",
            OperatorFromExtern::Split2 { .. } => "Split2",
            OperatorFromExtern::Other { .. } => "Other",
            OperatorFromExtern::GraphOutput { .. } => "GraphOutput",
        };
        return name.to_string();
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InstFromExtern {
    pub op: OperatorFromExtern,
    pub return_values: Vec<String>,
    pub input_args: Vec<String>,
    pub attributes: HashMap<String, String>,
}

#[allow(unused)]
#[no_mangle]
pub extern "C" fn init_helper() {
    if log_enabled!(Trace) {
        trace!("call init_helper");
    }
    if unsafe { ENV_LOGGER.is_none() } {
        let mut log_builder = env_logger::Builder::from_default_env();
        log_builder
            .format_module_path(false)
            .format_target(false)
            .format_timestamp(None)
            .init();
        unsafe { ENV_LOGGER = Some(log_builder) };
    }
}

#[allow(unused)]
#[no_mangle]
pub extern "C" fn initialize_graph(graph_json_str_: *mut ffi::c_char) -> *mut GraphFromExtern {
    if log_enabled!(Trace) {
        trace!("call initialize_graph");
    }
    let json_str_ = unsafe { ffi::CStr::from_ptr(graph_json_str_) };
    let json_str = json_str_.to_str().unwrap();
    let mut graph: GraphFromExtern = serde_json::from_str(&json_str).expect(&format!(
        "cannot parse graph json. original str: {:?}",
        &json_str
    ));
    graph.build_egraph_converter();
    let graph_box = Box::new(graph);
    // info!("successfully build converter for graph");
    return Box::into_raw(graph_box);
}

#[allow(unused)]
#[no_mangle]
pub unsafe extern "C" fn deallocate_graph(graph_: *mut GraphFromExtern) {
    if log_enabled!(Trace) {
        trace!("call deallocate_graph");
    }
    let mut graph_box: Box<GraphFromExtern> = Box::from_raw(graph_);
    graph_box.egraph_runner = None;
    graph_box.converter = None;
    graph_box.insts.clear();
    graph_box.vars.clear();
    drop(graph_box);
}

#[allow(unused)]
#[no_mangle]
pub extern "C" fn run_saturation(graph_: *mut GraphFromExtern) {
    if log_enabled!(Trace) {
        trace!("call run_saturation");
    }
    let graph = unsafe { (*graph_).borrow_mut() };
    graph.run_saturation();
}

#[allow(unused)]
#[no_mangle]
pub extern "C" fn output_self_graph(
    src_graph_: *mut GraphFromExtern,
    out_graph_json_str_: *mut ffi::c_char,
) {
    if log_enabled!(Trace) {
        trace!("call output_self_graph");
    }
    let graph = unsafe { (*src_graph_).borrow_mut() };
    let original_egraph_from_egg = graph.converter.as_ref().unwrap().rec_expr.clone();
    let new_graph = graph.build_from_egraph(&original_egraph_from_egg).unwrap();

    let graph_json_str = serde_json::to_string(&new_graph).unwrap();
    let tmp_str = ffi::CString::new(graph_json_str).unwrap();
    unsafe { copy_cstring_to_ffi_chars(&tmp_str, out_graph_json_str_) };
}

#[allow(unused)]
#[no_mangle]
pub extern "C" fn find_the_most_simplified_equivalent_graph(
    src_graph_: *mut GraphFromExtern,
    out_graph_json_str_: *mut ffi::c_char,
) {
    let graph = unsafe { (*src_graph_).borrow_mut() };
    if graph.egraph_runner.is_none() || graph.converter.is_none() {
        return;
    }

    let egraph_runner = graph.egraph_runner.as_ref().unwrap();
    let egraph = &egraph_runner.egraph;
    let root = graph.egraph_runner.as_ref().unwrap().roots[0];

    let extractor = Extractor::new(egraph, AstSize);
    let (_cost, mut simple_expr) = extractor.find_best(root);
    let new_graph = match graph.build_from_egraph(&simple_expr) {
        Ok(g) => g,
        Err(e) => {
            // this unlikely happens, but sometimes happen because of trans() shape error
            let randomly_picked_graph = random_pick(root, egraph, &graph);
            if randomly_picked_graph.is_none() {
                return;
            }
            randomly_picked_graph.unwrap()
        }
    };

    let graph_json_str = serde_json::to_string(&new_graph).unwrap();
    let tmp_str = ffi::CString::new(graph_json_str).unwrap();
    unsafe { copy_cstring_to_ffi_chars(&tmp_str, out_graph_json_str_) };
}

#[allow(unused)]
#[no_mangle]
pub extern "C" fn find_the_most_complex_equivalent_graph(
    src_graph_: *mut GraphFromExtern,
    out_graph_json_str_: *mut ffi::c_char,
) {
    let graph = unsafe { (*src_graph_).borrow_mut() };
    if graph.egraph_runner.is_none() || graph.converter.is_none() {
        return;
    }

    let root = graph.egraph_runner.as_ref().unwrap().roots[0].clone();
    let egraph_runner = graph.egraph_runner.as_ref().unwrap();
    let egraph = &egraph_runner.egraph;

    assert_eq!(egraph[root].nodes.len(), 1);

    // return true if there is a cycle
    fn dfs(
        id: Id,
        back_edges: &mut HashSet<(Id, Id)>,
        visited: &mut HashSet<Id>,
        current_stack: &mut HashSet<Id>,
        egraph: &EGraph<ComputExprIR, AnalysisOfCEIR>,
    ) -> bool {
        visited.insert(id);
        current_stack.insert(id);

        // for each node in this class
        for node in egraph[id].nodes.iter() {
            // for each child of this node
            for child in node.children().iter() {
                if back_edges.contains(&(child.clone(), id)) {
                    continue;
                }

                if !visited.contains(child) {
                    if dfs(child.clone(), back_edges, visited, current_stack, egraph) {
                        return true;
                    }
                } else if current_stack.contains(child) {
                    back_edges.insert((child.clone(), id));
                    return true;
                }
            }
        }
        current_stack.remove(&id);
        return false;
    };

    // return true if there is a cycle
    fn remove_cycles(
        back_edges: &mut HashSet<(Id, Id)>,
        egraph: &EGraph<ComputExprIR, AnalysisOfCEIR>,
    ) -> bool {
        let mut visited = HashSet::default();
        let mut current_stack = HashSet::default();
        for class in egraph.classes() {
            if visited.contains(&class.id) {
                continue;
            }
            if dfs(
                class.id,
                back_edges,
                &mut visited,
                &mut current_stack,
                egraph,
            ) {
                return true;
            }
        }
        return false;
    }

    // first, we mark back edges
    let mut back_edges = HashSet::default(); // (child, parent)
    while remove_cycles(&mut back_edges, egraph) {}

    // second, for each node, we find its most simplified sub-tree
    let simple_extractor = Extractor::new(egraph, AstSize);

    // third, we conduct a fix-point iteration to find the costs for each node
    let costs = fix_point_for_complex(&egraph, &back_edges, |id| {
        return simple_extractor.find_best_cost(id);
    });

    let expr = construct_a_complex_expr(costs, &egraph, &back_edges, root, |id| {
        return simple_extractor.find_best(id).1;
    });

    // egraph.dot().to_svg("./a.png").unwrap();

    // info!("new: {}", expr.pretty_str());
    // info!(
    //     "original: {}",
    //     graph.converter.as_ref().unwrap().rec_expr.pretty_str()
    // );

    let new_graph = match graph.build_from_egraph(&expr) {
        Ok(g) => g,
        Err(e) => {
            error!("cannot find a good equivalent graph because {}", e);
            let randomly_picked_graph = random_pick(root, egraph, &graph);
            if randomly_picked_graph.is_none() {
                return;
            }
            randomly_picked_graph.unwrap()
        }
    };

    let graph_json_str = serde_json::to_string(&new_graph).unwrap();
    let tmp_str = ffi::CString::new(graph_json_str).unwrap();
    unsafe { copy_cstring_to_ffi_chars(&tmp_str, out_graph_json_str_) };
}

// similar to Language::try_build_recexpr, but we need to traverse the egraph to find the most complex expression
// the most important thing is to add expr to e-graph
fn construct_a_complex_expr<F>(
    costs: HashMap<Id, (usize, Option<ComputExprIR>)>,
    egraph: &EGraph<ComputExprIR, AnalysisOfCEIR>,
    backedges: &HashSet<(Id, Id)>,
    root: Id,
    func_for_simplified: F,
) -> RecExpr<ComputExprIR>
where
    F: Fn(Id) -> RecExpr<ComputExprIR>,
{
    let mut set = IndexSet::<ComputExprIR>::default();
    let mut ids = HashMap::<Id, Id>::default();
    let mut backedge_ids = HashMap::<(Id, Id), Id>::default();
    assert_eq!(egraph[root].nodes.len(), 1);
    let root_node = egraph[root].nodes[0].clone();

    let mut todo: Vec<Id> = root_node.children().to_vec();

    // deal with backedges
    for (child, parent) in backedges.iter() {
        let expr = func_for_simplified(*child);

        let mut expr_to_indexset = HashMap::<Id, Id>::default();

        for (expr_index, node) in expr.as_ref().iter().enumerate() {
            let new_node = node.clone().map_children(|x| expr_to_indexset[&x]);
            let new_id = set.insert_full(new_node.clone()).0;
            expr_to_indexset.insert(Id::from(expr_index), Id::from(new_id));

            if expr_index == expr.as_ref().len() - 1 {
                // this is the root node of the expression
                backedge_ids.insert((child.clone(), parent.clone()), Id::from(new_id));
            }
        }
    }

    while let Some(id) = todo.last().copied() {
        if ids.contains_key(&id) {
            todo.pop();
            continue;
        }

        // info!("{:?}", todo);

        if let Some(node) = &costs[&id].1 {
            // "node" is the selected node for the e-class of this id

            // check to see if we can do this node yet
            let mut ids_has_all_children = true;
            for child in node.children() {
                if backedges.contains(&(*child, id)) {
                    continue;
                }
                if !ids.contains_key(child) {
                    ids_has_all_children = false;
                    todo.push(*child);
                }
            }

            // all children are processed, so we can lookup this node safely
            if ids_has_all_children {
                let node = node.clone().map_children(|child_id| {
                    if backedges.contains(&(child_id, id)) {
                        backedge_ids[&(child_id, id)]
                    } else {
                        ids[&child_id]
                    }
                });
                let new_id = set.insert_full(node.clone()).0;
                ids.insert(id, Id::from(new_id));
                todo.pop();
            }
        } else {
            // expr for this e-class
            let expr = func_for_simplified(id);

            let mut expr_to_indexset = HashMap::<Id, Id>::default();

            for (expr_index, node) in expr.as_ref().iter().enumerate() {
                let new_node = node.clone().map_children(|x| expr_to_indexset[&x]);
                let new_id = set.insert_full(new_node.clone()).0;
                expr_to_indexset.insert(Id::from(expr_index), Id::from(new_id));

                if expr_index == expr.as_ref().len() - 1 {
                    // this is the root node of the expression
                    ids.insert(id, Id::from(new_id));
                }
            }
            todo.pop();
        }
    }

    // finally, add the root node and create the expression
    let mut nodes: Vec<ComputExprIR> = set.into_iter().collect();
    nodes.push(root_node.map_children(|id| ids[&id]));
    RecExpr::from(nodes)
}

/// conduct a fix-point iteration to find the most complex equivalent graph, omitting the back edges in the e-graph.
fn fix_point_for_complex<F>(
    egraph: &EGraph<ComputExprIR, AnalysisOfCEIR>,
    backedges: &HashSet<(Id, Id)>,
    func_for_simplified: F,
) -> HashMap<Id, (usize, Option<ComputExprIR>)>
where
    F: Fn(Id) -> usize,
{
    let mut costs: HashMap<Id, (usize, Option<ComputExprIR>)> = HashMap::default();

    let mut no_any_equivalent = true;
    for class in egraph.classes() {
        assert_ne!(class.nodes.len(), 0);
        // costs.insert(class.id, (1, None));
        // mini_costs.insert(class.id, (1, None));

        if class.nodes.len() > 0 {
            no_any_equivalent = false;
        }
    }

    if no_any_equivalent {
        info!("no equivalence, meaning that no rule is applied");
    }

    let mut did_something = true;
    let eclasses = egraph.classes().cloned().collect::<Vec<_>>();
    while did_something {
        did_something = false;
        for class in eclasses.iter() {
            let mut max_cost = 0;
            let mut max_cost_node = None;

            for node in class.nodes.iter() {
                let mut cost_of_this_node: Option<usize> = Some(1);

                if node.children().len() != 0 {
                    for child in node.children().iter() {
                        if backedges.contains(&(*child, class.id)) {
                            // this node cannot be used
                            // cost_of_this_node = None;
                            // break;

                            // this node should be replaced with the simplified one since it is a back edge
                            let cost = func_for_simplified(*child);
                            cost_of_this_node = Some(cost_of_this_node.unwrap() + cost);
                        } else {
                            if let Some((cost, _)) = costs.get(child) {
                                assert!(cost_of_this_node.is_some());
                                cost_of_this_node = Some(cost_of_this_node.unwrap() + cost);
                            } else {
                                // this node cannot be used for now, wait for the next iteration
                                cost_of_this_node = None;
                                break;
                            }
                        }
                    }
                }

                if cost_of_this_node.is_some() {
                    if cost_of_this_node.unwrap() > max_cost {
                        max_cost = cost_of_this_node.unwrap();
                        max_cost_node = Some(node.clone());
                    }
                }
            }
            if max_cost_node.is_some() {
                if costs.get(&class.id).is_none() || max_cost > costs[&class.id].0 {
                    costs.insert(class.id, (max_cost, max_cost_node));
                    did_something = true;
                }
            }
        }
    }

    for class in egraph.classes() {
        if costs.get(&class.id).is_none() {
            costs.insert(class.id, (0, None));
        }
    }

    return costs;
}

#[allow(unused)]
#[no_mangle]
pub extern "C" fn randomly_find_an_equivalent_graph(
    src_graph_: *mut GraphFromExtern,
    out_graph_json_str_: *mut ffi::c_char,
) {
    if log_enabled!(Trace) {
        trace!("call randomly_find_an_equivalent_graph");
    }
    let graph = unsafe { (*src_graph_).borrow_mut() };
    if graph.egraph_runner.is_none() || graph.converter.is_none() {
        return;
    }

    let egraph_runner = graph.egraph_runner.as_ref().unwrap();
    let egraph = &egraph_runner.egraph;
    let root = graph.egraph_runner.as_ref().unwrap().roots[0];

    let new_graph = match random_pick(root, egraph, &graph) {
        Some(g) => g,
        None => {
            return;
        }
    };

    let graph_json_str = serde_json::to_string(&new_graph).unwrap();
    let tmp_str = ffi::CString::new(graph_json_str).unwrap();
    unsafe { copy_cstring_to_ffi_chars(&tmp_str, out_graph_json_str_) };
}

fn random_pick(
    root: Id,
    egraph: &EGraph<ComputExprIR, AnalysisOfCEIR>,
    graph: &GraphFromExtern,
) -> Option<GraphFromExtern> {
    let mut rng = thread_rng();
    let random_pick = |id: Id| {
        let random_enode = egraph[id].nodes.choose(&mut rng).unwrap();
        return random_enode.clone();
    };

    let new_graph = match extract_a_graph_from_egraph(graph, root, random_pick) {
        Ok(g) => g,
        Err(e) => {
            error!("cannot find a good equivalent graph because {}", e);
            return None;
        }
    };
    return Some(new_graph);
}

fn extract_a_graph_from_egraph<F>(
    graph: &GraphFromExtern,
    root: Id,
    mut traverse_func: F,
) -> Result<GraphFromExtern, BuildGraphError>
where
    F: FnMut(Id) -> ComputExprIR,
{
    trace!("start building rec_expr");
    let egraph = &graph.egraph_runner.as_ref().unwrap().egraph;
    let mut rec_expr = egraph[root].nodes[0].build_recexpr(&mut traverse_func);
    trace!("finish building rec_expr");
    // let mut new_graph = GraphFromExtern::default();
    let start_time = Instant::now();
    let mut failed_cnt = 0u64;
    loop {
        let time_duration = (Instant::now() - start_time).as_millis();
        if time_duration > 10 {
            error!(
                "cannot find a good equivalent graph within {} ms, \
                have retried {} times, use the original one",
                time_duration, failed_cnt
            );
            let original_egraph_from_egg = graph.converter.as_ref().unwrap().rec_expr.clone();
            let new_graph = match graph.build_from_egraph(&original_egraph_from_egg) {
                Ok(g) => g,
                Err(e) => {
                    info!("cannot build from egraph because {}", e);
                    return Err(BuildGraphError::new(
                        "cannot build from egraph".to_string(),
                        e.to_string(),
                    ));
                }
            };
            return Ok(new_graph);
        }
        trace!("start building egraph");
        let graph_res = graph.build_from_egraph(&rec_expr);
        trace!("finish building egraph");
        if graph_res.is_ok() {
            let new_graph = graph_res.unwrap();
            return Ok(new_graph);
        } else {
            failed_cnt += 1;
            trace!(
                "{}-th fail, retry. error: {}",
                failed_cnt,
                graph_res.unwrap_err()
            );
            trace!("start building rec_expr");
            rec_expr = egraph[root].nodes[0].build_recexpr(&mut traverse_func);
            trace!("finish building rec_expr");
        }
    }
}

impl GraphFromExtern {
    fn set_egraph(&mut self, graph_converter: GraphConverter) {
        self.converter = Some(graph_converter);
    }
    /// run equality saturation and store them to self.egraph_runner.
    pub fn run_saturation(&mut self) {
        if self.converter.is_none() {
            return;
        }
        let graph_converter = self.converter.as_ref().unwrap();
        let node_limit = match env::var("HELPER_NODE_LIMIT") {
            Ok(val) => match val.parse::<usize>() {
                Ok(value) => value,
                Err(e) => {
                    info!("cannot parse HELPER_NODE_LIMIT because {}", e);
                    DEFAULT_HELPER_NODE_LIMIT
                }
            },
            Err(e) => match e {
                env::VarError::NotPresent => DEFAULT_HELPER_NODE_LIMIT,
                _ => {
                    info!("cannot parse HELPER_NODE_LIMIT because {}", e);
                    DEFAULT_HELPER_NODE_LIMIT
                }
            },
        };

        let time_limit_sec = match env::var("HELPER_TIME_LIMIT") {
            Ok(val) => match val.parse::<u64>() {
                Ok(value) => value,
                Err(e) => {
                    info!("cannot parse HELPER_TIME_LIMIT because {}", e);
                    DEFAULT_HELPER_TIME_LIMIT
                }
            },
            Err(e) => match e {
                env::VarError::NotPresent => DEFAULT_HELPER_TIME_LIMIT,
                _ => {
                    info!("cannot parse HELPER_TIME_LIMIT because {}", e);
                    DEFAULT_HELPER_TIME_LIMIT
                }
            },
        };

        let iter_limit = match env::var("HELPER_ITER_LIMIT") {
            Ok(val) => match val.parse::<usize>() {
                Ok(value) => value,
                Err(e) => {
                    info!("cannot parse HELPER_ITER_LIMIT because {}", e);
                    DEFAULT_HELPER_ITER_LIMIT
                }
            },
            Err(e) => match e {
                env::VarError::NotPresent => DEFAULT_HELPER_ITER_LIMIT,
                _ => {
                    info!("cannot parse HELPER_ITER_LIMIT because {}", e);
                    DEFAULT_HELPER_ITER_LIMIT
                }
            },
        };

        let rules = rules();

        let runner = Runner::<ComputExprIR, AnalysisOfCEIR, ()>::default()
            // .with_explanations_enabled()
            .with_node_limit(node_limit)
            .with_time_limit(Duration::from_secs(time_limit_sec))
            .with_iter_limit(iter_limit)
            .with_expr(&graph_converter.rec_expr)
            .run(&rules);
        info!("saturation stop reason: {:?}", runner.stop_reason);
        self.egraph_runner = Some(runner);
    }

    /// check if the tensor is identical to the one defined in external graph
    #[allow(unused)]
    fn is_tensor_shape_valid(&self, input_tensor: &TensorInfo, tensor_name: &String) -> bool {
        let var_info = self.vars.get(tensor_name).unwrap();
        if var_info.shape.len() != input_tensor.shape.len() {
            return false;
        }
        for i in 0..var_info.shape.len() {
            if var_info.shape[i] != input_tensor.shape[i] {
                return false;
            }
        }
        return true;
    }

    pub fn build_egraph_converter(&mut self) {
        let mut graph_converter = GraphConverter::default();
        // let mut tensor_map: TensorNameMap = HashMap::default();

        // for (var_name, var_info) in external_graph.vars.iter() {
        //     let mut shape: Vec<i32> = vec![0; var_info.shape.len()];
        //     for (index, shape_dim) in var_info.shape.iter().enumerate() {
        //         shape_dim[index] = shape_dim.clone() as i32;
        //     }
        //     let tensor_info = graph_converter.new_input(var_name.clone(), &shape);
        //     tensor_map.insert(var_name.clone(), tensor_info);
        // }

        let insts: &Vec<InstFromExtern> = self.insts.as_ref();
        let vars: &HashMap<String, VarFromExtern> = &self.vars;
        let mut is_used: Vec<bool> = vec![false; insts.len()];
        let mut used_cnt: usize = 0;

        while used_cnt != is_used.len() {
            let is_used_old = is_used.clone();
            for (index, inst) in insts.iter().enumerate() {
                if is_used[index] {
                    continue;
                }
                let mut are_all_inputs_recorded = true;
                for input_arg in inst.input_args.iter() {
                    if !graph_converter.tensor_name_map.contains_key(input_arg) {
                        are_all_inputs_recorded = false;
                        break;
                    }
                }
                // check if all inputs of this op have been recorded
                if !are_all_inputs_recorded {
                    continue;
                }

                // after all above checks, we can ensure that all the inputs of this instruction have
                // been recorded in the tensor_map, so we can deal with this instruction.
                match &inst.op {
                    OperatorFromExtern::Input { op_index }
                    | OperatorFromExtern::Constant { op_index } => {
                        assert_eq!(inst.input_args.len(), 0);
                        assert_eq!(inst.return_values.len(), 1);
                        let return_val_name = &inst.return_values[0];
                        let var_info = self.vars.get(return_val_name).unwrap();
                        let tensor_type = inst.op.op_name(); // input/constant/placeholder
                        graph_converter.new_input(
                            return_val_name,
                            &tensor_type,
                            var_info,
                            *op_index,
                        );
                    }
                    OperatorFromExtern::ReLU => {
                        assert_eq!(inst.input_args.len(), 1);
                        assert_eq!(inst.return_values.len(), 1);
                        let input_arg_name = &inst.input_args[0];
                        let return_val_name = &inst.return_values[0];
                        let output_info = self.vars.get(return_val_name).unwrap();

                        let input_tensor = graph_converter
                            .get_tensor_info_by_name(input_arg_name)
                            .clone();

                        graph_converter.relu(&input_tensor, return_val_name, output_info);
                    }
                    OperatorFromExtern::Sigmoid => {
                        assert_eq!(inst.input_args.len(), 1);
                        assert_eq!(inst.return_values.len(), 1);

                        let input_arg_name = &inst.input_args[0];
                        let return_val_name = &inst.return_values[0];
                        let output_info = self.vars.get(return_val_name).unwrap();

                        let input_tensor = graph_converter
                            .get_tensor_info_by_name(input_arg_name)
                            .clone();

                        graph_converter.sigmoid(&input_tensor, return_val_name, output_info);
                    }
                    OperatorFromExtern::Tanh => {
                        assert_eq!(inst.input_args.len(), 1);
                        assert_eq!(inst.return_values.len(), 1);

                        let input_arg_name = &inst.input_args[0];
                        let return_val_name = &inst.return_values[0];
                        let output_info = self.vars.get(return_val_name).unwrap();

                        let input_tensor = graph_converter
                            .get_tensor_info_by_name(input_arg_name)
                            .clone();

                        graph_converter.tanh(&input_tensor, return_val_name, output_info);
                    }
                    OperatorFromExtern::Transpose { dim0, dim1 } => {
                        assert_eq!(inst.input_args.len(), 1);
                        assert_eq!(inst.return_values.len(), 1);

                        let input_arg_name = &inst.input_args[0];
                        let return_val_name = &inst.return_values[0];
                        let output_info = self.vars.get(return_val_name).unwrap();

                        let input_tensor = graph_converter
                            .get_tensor_info_by_name(input_arg_name)
                            .clone();

                        graph_converter.transpose(
                            &input_tensor,
                            *dim0 as i64,
                            *dim1 as i64,
                            return_val_name,
                            output_info,
                        );
                    }
                    OperatorFromExtern::Add => {
                        assert_eq!(inst.input_args.len(), 2);
                        assert_eq!(inst.return_values.len(), 1);
                        let input_arg0_name = &inst.input_args[0];
                        let input_arg1_name = &inst.input_args[1];
                        let return_val_name = &inst.return_values[0];

                        let output_info = self.vars.get(return_val_name).unwrap();

                        let input_tensor0 = graph_converter
                            .get_tensor_info_by_name(input_arg0_name)
                            .clone();

                        let input_tensor1 = graph_converter
                            .get_tensor_info_by_name(input_arg1_name)
                            .clone();

                        graph_converter.add(
                            &input_tensor0,
                            &input_tensor1,
                            return_val_name,
                            output_info,
                        );
                    }
                    OperatorFromExtern::Mul => {
                        assert_eq!(inst.input_args.len(), 2);
                        assert_eq!(inst.return_values.len(), 1);
                        let input_arg0_name = &inst.input_args[0];
                        let input_arg1_name = &inst.input_args[1];
                        let return_val_name = &inst.return_values[0];

                        let output_info = self.vars.get(return_val_name).unwrap();

                        let input_tensor0 = graph_converter
                            .get_tensor_info_by_name(input_arg0_name)
                            .clone();

                        let input_tensor1 = graph_converter
                            .get_tensor_info_by_name(input_arg1_name)
                            .clone();

                        graph_converter.mul(
                            &input_tensor0,
                            &input_tensor1,
                            return_val_name,
                            output_info,
                        );
                    }
                    OperatorFromExtern::MatMul => {
                        assert_eq!(inst.input_args.len(), 2);
                        assert_eq!(inst.return_values.len(), 1);
                        let input_arg0_name = &inst.input_args[0];
                        let input_arg1_name = &inst.input_args[1];
                        let return_val_name = &inst.return_values[0];

                        let output_info = self.vars.get(return_val_name).unwrap();

                        let input_tensor0 = graph_converter
                            .get_tensor_info_by_name(input_arg0_name)
                            .clone();

                        let input_tensor1 = graph_converter
                            .get_tensor_info_by_name(input_arg1_name)
                            .clone();

                        graph_converter.matmul(
                            &input_tensor0,
                            &input_tensor1,
                            return_val_name,
                            output_info,
                        );
                    }
                    // OperatorFromExtern::Conv2d {
                    //     in_channels,
                    //     out_channels,
                    //     kernel_h_size,
                    //     kernel_w_size,
                    //     stride, padding,
                    //     dilation_h,
                    //     dilation_w
                    // } => {
                    //     assert_eq!(inst.input_args.len(), 1);
                    //     assert_eq!(inst.return_values.len(), 1);
                    //     // weight is a big problem since weight in pytorch or tensorflow is leanable parameter
                    //     graph_converter.conv2d()
                    // }
                    OperatorFromExtern::AvgPool2d {
                        kh,
                        kw,
                        stride,
                        padding,
                    } => {
                        assert_eq!(inst.input_args.len(), 1);
                        assert_eq!(inst.return_values.len(), 1);

                        let input_arg_name = &inst.input_args[0];
                        let return_val_name = &inst.return_values[0];
                        let output_info = self.vars.get(return_val_name).unwrap();

                        let input_tensor = graph_converter
                            .get_tensor_info_by_name(input_arg_name)
                            .clone();

                        graph_converter.avgpool2d(
                            &input_tensor,
                            *kh as i64,
                            *kw as i64,
                            *stride as i64,
                            *padding as i64,
                            return_val_name,
                            output_info,
                        );
                    }
                    OperatorFromExtern::MaxPool2d {
                        kh,
                        kw,
                        stride,
                        padding,
                    } => {
                        assert_eq!(inst.input_args.len(), 1);
                        assert_eq!(inst.return_values.len(), 1);

                        let input_arg_name = &inst.input_args[0];
                        let return_val_name = &inst.return_values[0];
                        let output_info = self.vars.get(return_val_name).unwrap();

                        let input_tensor = graph_converter
                            .get_tensor_info_by_name(input_arg_name)
                            .clone();

                        graph_converter.maxpool2d(
                            &input_tensor,
                            *kh as i64,
                            *kw as i64,
                            *stride as i64,
                            *padding as i64,
                            return_val_name,
                            output_info,
                        );
                    }
                    OperatorFromExtern::Concat2 { axis } => {
                        assert_eq!(inst.input_args.len(), 2);
                        assert_eq!(inst.return_values.len(), 1);

                        let input_arg0_name = &inst.input_args[0];
                        let input_arg1_name = &inst.input_args[1];
                        let return_val_name = &inst.return_values[0];

                        let output_info = self.vars.get(return_val_name).unwrap();

                        let input_tensor0 = graph_converter
                            .get_tensor_info_by_name(input_arg0_name)
                            .clone();

                        let input_tensor1 = graph_converter
                            .get_tensor_info_by_name(input_arg1_name)
                            .clone();

                        graph_converter.concat(
                            &input_tensor0,
                            &input_tensor1,
                            *axis as i64,
                            return_val_name,
                            output_info,
                        );
                    }

                    OperatorFromExtern::Split2 { axis } => {
                        assert_eq!(inst.input_args.len(), 1);
                        assert_eq!(inst.return_values.len(), 2);
                        let input_arg_name = &inst.input_args[0];
                        let return_val_name_0 = &inst.return_values[0];
                        let return_val_name_1 = &inst.return_values[1];

                        let input_tensor = graph_converter
                            .get_tensor_info_by_name(input_arg_name)
                            .clone();
                        let output_info_0 = self.vars.get(return_val_name_0).unwrap();
                        let output_info_1 = self.vars.get(return_val_name_1).unwrap();

                        graph_converter.split2(
                            &input_tensor,
                            *axis as i64,
                            return_val_name_0,
                            output_info_0,
                            return_val_name_1,
                            output_info_1,
                        );
                    }

                    OperatorFromExtern::Other { op_index } => {
                        let mut input_tensors = vec![];
                        for input_arg_name in inst.input_args.iter() {
                            let input_tensor =
                                graph_converter.get_tensor_info_by_name(input_arg_name);
                            input_tensors.push(input_tensor.clone());
                        }

                        let mut output_infos = vec![];
                        let mut output_names = vec![];
                        for output_arg_name in inst.return_values.iter() {
                            output_infos.push(vars.get(output_arg_name).unwrap());
                            output_names.push(output_arg_name.as_str());
                        }

                        graph_converter.other(
                            *op_index,
                            input_tensors.iter().collect(),
                            output_names,
                            output_infos,
                        );
                    }
                    OperatorFromExtern::GraphOutput { var_name } => {
                        assert_eq!(inst.input_args.len(), 1);
                        assert_eq!(inst.return_values.len(), 0);
                        let input_arg_name = &inst.input_args[0];
                        assert_eq!(var_name, input_arg_name);

                        let input_tensor = graph_converter
                            .get_tensor_info_by_name(input_arg_name)
                            .clone();
                        graph_converter.graph_output(&input_tensor);
                    }
                };
                is_used[index] = true;
                used_cnt += 1;
            }
            if is_used == is_used_old {
                // jump out the loop in case it has outliers
                if is_used
                    .iter()
                    .map(|x| if *x { 1usize } else { 0usize })
                    .sum::<usize>()
                    != is_used.len()
                {
                    error!("there are outliers in the graph: {:?}", self);
                }
                break;
            }
        }
        graph_converter.assemble_outputs();
        self.set_egraph(graph_converter);
    }

    /// build ``GraphFromExtern`` from ``ComputationGraphFromEgg``.
    pub fn build_from_egraph(
        &self,
        inner_graph: &ComputationGraphFromEgg,
    ) -> Result<Self, BuildGraphError> {
        assert!(inner_graph.is_dag());
        assert!(self.converter.is_some());
        let inner_graph = remove_redundant_nodes(inner_graph);

        let converter = self.converter.as_ref().unwrap();

        let mut external_graph = Self::default(); // the new external graph that used as output of this function

        let mut id_name_map: HashMap<Id, String> = HashMap::default(); // record the enode id to its new tensor name
        let mut is_used: Vec<bool> = vec![false; inner_graph.as_ref().len()]; // record if each op in the "inner_graph" has been processed
        let mut used_cnt: usize = 0; // record the op number that we have been processed

        // in this loop, we need to maintain "external_graph", "id_name_map", "is_used", and "used_cnt"
        while used_cnt != is_used.len() {
            let is_used_old = is_used.clone();
            for (enode_index, enode) in inner_graph.as_ref().iter().enumerate() {
                if is_used[enode_index] {
                    continue;
                }

                let enode_id = Id::from(enode_index);
                match enode {
                    ComputExprIR::Tensor(args) => {
                        let tensor_name = inner_graph[args[0]].try_parse_to_var().unwrap();
                        let op_type = inner_graph[args[1]]
                            .try_parse_to_var()
                            .expect(&format!("{:?}", inner_graph[args[1]]));
                        let op_index = inner_graph[args[2]].try_parse_to_num().unwrap();
                        let new_tensor_name = format!("v{}_0", external_graph.insts.len());

                        // deal with VarFromExtern
                        let tensor_info = converter.get_tensor_info_by_name(&tensor_name);
                        let var = VarFromExtern::from(tensor_info);
                        let _ = external_graph.vars.insert(new_tensor_name.clone(), var);
                        id_name_map.insert(enode_id, new_tensor_name.to_string());

                        // deal with InstFromExtern
                        let op = match op_type {
                            "Input" => OperatorFromExtern::Input {
                                op_index: op_index as u64,
                            },
                            "Constant" => OperatorFromExtern::Constant {
                                op_index: op_index as u64,
                            },
                            _ => {
                                unreachable!()
                            }
                        };
                        let inst = InstFromExtern {
                            op: op,
                            input_args: vec![],
                            return_values: vec![new_tensor_name.to_string()],
                            attributes: HashMap::default(),
                        };
                        external_graph.insts.push(inst);

                        is_used[enode_index] = true;
                        used_cnt += 1;
                    }
                    ComputExprIR::Ewadd(args)
                    | ComputExprIR::Ewmul(args)
                    | ComputExprIR::Matmul(args) => {
                        if !is_used[usize::from(args[0])] || !is_used[usize::from(args[1])] {
                            continue;
                        }

                        let input_tensor0_name = id_name_map[&args[0]].clone();
                        let input_tensor1_name = id_name_map[&args[1]].clone();
                        let output_tensor_name = format!("v{}_0", external_graph.insts.len());

                        // deal with output VarFromExtern
                        let input_tensor_0 = &external_graph.vars[&input_tensor0_name];
                        let input_tensor_1 = &external_graph.vars[&input_tensor1_name];
                        let mut output_shapes = enode.shape_transfer(
                            &vec![&input_tensor_0.shape, &input_tensor_1.shape],
                            |x| &inner_graph[x],
                        )?;
                        let mut output_dtypes = enode
                            .dtype_transfer(&vec![&input_tensor_0.dtype, &input_tensor_1.dtype])?;
                        assert_eq!(output_shapes.len(), 1);
                        assert_eq!(output_dtypes.len(), 1);
                        let output_shape = output_shapes.pop().unwrap();
                        let output_dtype = output_dtypes.pop().unwrap();
                        let var = VarFromExtern {
                            shape: output_shape,
                            dtype: output_dtype,
                        };
                        let _ = external_graph.vars.insert(output_tensor_name.clone(), var);
                        id_name_map.insert(enode_id, output_tensor_name.clone());

                        // deal with InstFromExtern
                        let op = match enode {
                            ComputExprIR::Ewadd(_) => OperatorFromExtern::Add,
                            ComputExprIR::Ewmul(_) => OperatorFromExtern::Mul,
                            ComputExprIR::Matmul(_) => OperatorFromExtern::MatMul,
                            _ => {
                                unreachable!()
                            }
                        };
                        let inst = InstFromExtern {
                            op: op,
                            input_args: vec![input_tensor0_name, input_tensor1_name],
                            return_values: vec![output_tensor_name],
                            attributes: HashMap::default(),
                        };
                        external_graph.insts.push(inst);

                        is_used[enode_index] = true;
                        used_cnt += 1;
                    }
                    ComputExprIR::Transpose(args) => {
                        if !is_used[usize::from(args[0])] {
                            continue;
                        }

                        if !id_name_map.contains_key(&args[0]) {
                            error!("enode: {:?}", enode);
                            error!("cannot find tensor");
                            error!("graph: {}", inner_graph.pretty_str());
                            error!("id map: {:?}", id_name_map);
                            error!("input tensor: {:?}", args[0]);
                        }
                        let input_tensor_name = id_name_map
                            .get(&args[0])
                            .ok_or(BuildGraphError::new(
                                enode.to_string(),
                                format!("out of range in id_name_map"),
                            ))?
                            .clone();

                        let dim0 = inner_graph[args[1]].try_parse_to_num().expect(&format!(
                            "enode: {:?}\ngraph: {}",
                            enode,
                            inner_graph.pretty_str()
                        )) as usize;
                        let dim1 = inner_graph[args[2]].try_parse_to_num().expect(&format!(
                            "enode: {:?}\ngraph: {}",
                            enode,
                            inner_graph.pretty_str()
                        )) as usize;
                        let output_tensor_name = format!("v{}_0", external_graph.insts.len());

                        // deal with output VarFromExtern
                        let input_tensor = &external_graph.vars[&input_tensor_name];
                        let mut output_shapes = enode
                            .shape_transfer(&vec![&input_tensor.shape], |x| &inner_graph[x])?;
                        let mut output_dtypes = enode.dtype_transfer(&vec![&input_tensor.dtype])?;
                        assert_eq!(output_shapes.len(), 1);
                        assert_eq!(output_dtypes.len(), 1);
                        let output_shape = output_shapes.pop().unwrap();
                        let output_dtype = output_dtypes.pop().unwrap();
                        let var = VarFromExtern {
                            shape: output_shape,
                            dtype: output_dtype,
                        };
                        let _ = external_graph.vars.insert(output_tensor_name.clone(), var);
                        id_name_map.insert(enode_id, output_tensor_name.clone());

                        // deal with output InstFromExtern
                        let op = OperatorFromExtern::Transpose { dim0, dim1 };
                        let inst = InstFromExtern {
                            op: op,
                            input_args: vec![input_tensor_name],
                            return_values: vec![output_tensor_name],
                            attributes: HashMap::default(),
                        };
                        external_graph.insts.push(inst);

                        is_used[enode_index] = true;
                        used_cnt += 1;
                    }
                    ComputExprIR::Relu(arg)
                    | ComputExprIR::Tanh(arg)
                    | ComputExprIR::Sigmoid(arg) => {
                        if !is_used[usize::from(*arg)] {
                            continue;
                        }

                        let input_tensor_name = id_name_map[arg].clone();

                        let output_tensor_name = format!("v{}_0", external_graph.insts.len());

                        // deal with output VarFromExtern
                        let input_tensor = &external_graph.vars[&input_tensor_name];
                        let mut output_shapes = enode
                            .shape_transfer(&vec![&input_tensor.shape], |x| &inner_graph[x])?;
                        let mut output_dtypes = enode.dtype_transfer(&vec![&input_tensor.dtype])?;
                        assert_eq!(output_shapes.len(), 1);
                        assert_eq!(output_dtypes.len(), 1);
                        let output_shape = output_shapes.pop().unwrap();
                        let output_dtype = output_dtypes.pop().unwrap();
                        let var = VarFromExtern {
                            shape: output_shape,
                            dtype: output_dtype,
                        };
                        let _ = external_graph.vars.insert(output_tensor_name.clone(), var);
                        id_name_map.insert(enode_id, output_tensor_name.clone());

                        // deal with output InstFromExtern
                        let op = match enode {
                            ComputExprIR::Relu(_) => OperatorFromExtern::ReLU,
                            ComputExprIR::Tanh(_) => OperatorFromExtern::Tanh,
                            ComputExprIR::Sigmoid(_) => OperatorFromExtern::Sigmoid,
                            _ => {
                                unreachable!()
                            }
                        };
                        let inst = InstFromExtern {
                            op: op,
                            input_args: vec![input_tensor_name],
                            return_values: vec![output_tensor_name],
                            attributes: HashMap::default(),
                        };
                        external_graph.insts.push(inst);

                        is_used[enode_index] = true;
                        used_cnt += 1;
                    }
                    ComputExprIR::Poolmax(args) | ComputExprIR::Poolavg(args) => {
                        // args: input, kernel_h, kernel_w, stride, padding

                        if !is_used[usize::from(args[0])] {
                            continue;
                        }

                        let input_tensor_name = id_name_map[&args[0]].clone();

                        if !external_graph.vars.contains_key(&input_tensor_name) {
                            continue;
                        }
                        let kh = inner_graph[args[1]].try_parse_to_num().unwrap() as usize;
                        let kw = inner_graph[args[2]].try_parse_to_num().unwrap() as usize;
                        let stride = inner_graph[args[3]].try_parse_to_num().unwrap() as usize;
                        let padding = inner_graph[args[4]].try_parse_to_num().unwrap() as usize;

                        let output_tensor_name = format!("v{}_0", external_graph.insts.len());

                        // deal with output VarFromExtern
                        let input_tensor = &external_graph.vars[&input_tensor_name];
                        let mut output_shapes = enode
                            .shape_transfer(&vec![&input_tensor.shape], |x| &inner_graph[x])?;
                        let mut output_dtypes = enode.dtype_transfer(&vec![&input_tensor.dtype])?;
                        assert_eq!(output_shapes.len(), 1);
                        assert_eq!(output_dtypes.len(), 1);
                        let output_shape = output_shapes.pop().unwrap();
                        let output_dtype = output_dtypes.pop().unwrap();
                        let var = VarFromExtern {
                            shape: output_shape,
                            dtype: output_dtype,
                        };
                        let _ = external_graph.vars.insert(output_tensor_name.clone(), var);
                        id_name_map.insert(enode_id, output_tensor_name.clone());

                        // deal with output InstFromExtern
                        let op = match enode {
                            ComputExprIR::Poolmax(_) => OperatorFromExtern::MaxPool2d {
                                kh,
                                kw,
                                stride,
                                padding,
                            },
                            ComputExprIR::Poolavg(_) => OperatorFromExtern::AvgPool2d {
                                kh,
                                kw,
                                stride,
                                padding,
                            },
                            _ => {
                                unreachable!()
                            }
                        };
                        let inst = InstFromExtern {
                            op: op,
                            input_args: vec![input_tensor_name],
                            return_values: vec![output_tensor_name],
                            attributes: HashMap::default(),
                        };
                        external_graph.insts.push(inst);

                        is_used[enode_index] = true;
                        used_cnt += 1;
                    }
                    ComputExprIR::Concat(args) => {
                        // args: input1, input2, axis

                        if !is_used[usize::from(args[0])] || !is_used[usize::from(args[1])] {
                            continue;
                        }

                        if !id_name_map.contains_key(&args[0])
                            || !id_name_map.contains_key(&args[1])
                        {
                            error!("cannot find tensors");
                            error!("graph: {}", inner_graph.pretty_str());
                            error!("id map: {:?}", id_name_map);
                            error!("input tensor 0: {:?}", args[0]);
                            error!("input tensor 1: {:?}", args[1]);
                        }

                        let input_tensor0_name = id_name_map[&args[0]].clone();
                        let input_tensor1_name = id_name_map[&args[1]].clone();

                        let axis = inner_graph[args[2]].try_parse_to_num().unwrap() as usize;
                        let output_tensor_name = format!("v{}_0", external_graph.insts.len());

                        // deal with output VarFromExtern
                        let input_tensor_0 = &external_graph.vars[&input_tensor0_name];
                        let input_tensor_1 = &external_graph.vars[&input_tensor1_name];
                        let mut output_shapes = enode.shape_transfer(
                            &vec![&input_tensor_0.shape, &input_tensor_1.shape],
                            |x| &inner_graph[x],
                        )?;
                        let mut output_dtypes = enode
                            .dtype_transfer(&vec![&input_tensor_0.dtype, &input_tensor_1.dtype])?;
                        assert_eq!(output_shapes.len(), 1);
                        assert_eq!(output_dtypes.len(), 1);
                        let output_shape = output_shapes.pop().unwrap();
                        let output_dtype = output_dtypes.pop().unwrap();
                        let var = VarFromExtern {
                            shape: output_shape,
                            dtype: output_dtype,
                        };
                        let _ = external_graph.vars.insert(output_tensor_name.clone(), var);
                        id_name_map.insert(enode_id, output_tensor_name.clone());

                        // deal with InstFromExtern
                        let op = OperatorFromExtern::Concat2 { axis };
                        let inst = InstFromExtern {
                            op: op,
                            input_args: vec![input_tensor0_name, input_tensor1_name],
                            return_values: vec![output_tensor_name],
                            attributes: HashMap::default(),
                        };
                        external_graph.insts.push(inst);

                        is_used[enode_index] = true;
                        used_cnt += 1;
                    }
                    ComputExprIR::SplitOut(_) => {
                        unimplemented!()
                    }
                    ComputExprIR::Split2_0(args) | ComputExprIR::Split2_1(args) => {
                        if !is_used[usize::from(args[0])] {
                            continue;
                        }

                        let order = match enode {
                            ComputExprIR::Split2_0(_) => 0usize,
                            ComputExprIR::Split2_1(_) => 1usize,
                            _ => unreachable!(),
                        };
                        let input_tensor_name = id_name_map[&args[0]].clone();
                        let input_tensor = &external_graph.vars[&input_tensor_name];
                        let axis = inner_graph[args[1]].try_parse_to_num().unwrap() as usize;

                        // find all other ops that have the same inputs with this op
                        // [(op_index, split_order, op)]
                        let mut split_nodes: Vec<(usize, usize, &ComputExprIR)> = {
                            let mut nodes = vec![];
                            for (other_index, other_enode) in
                                inner_graph.as_ref().iter().enumerate()
                            {
                                if enode_index == other_index {
                                    // this is the same op with the current op, skip it
                                    continue;
                                }
                                match other_enode {
                                    ComputExprIR::Split2_0(other_args) => {
                                        if args[0] == other_args[0] && args[1] == other_args[1] {
                                            nodes.push((other_index, 0, other_enode));
                                        }
                                    }
                                    ComputExprIR::Split2_1(other_args) => {
                                        if args[0] == other_args[0] && args[1] == other_args[1] {
                                            nodes.push((other_index, 1, other_enode));
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            nodes
                        };
                        split_nodes.push((enode_index, order, enode));
                        // TODO: I have observed there is a very small chance (1/1600) that
                        // there are duplicated enodes in the egraph, perhaps we just ignore it.
                        if split_nodes.len() > 2 {
                            return Err(BuildGraphError {
                                error_op: "split".to_string(),
                                reason: format!(
                                    "the number of split_nodes is {}, greater than 2. {:?}.\nexpression: {}",
                                    split_nodes.len(),
                                    split_nodes, inner_graph.pretty_str()
                                ),
                            });
                        }

                        let output_shapes = enode
                            .shape_transfer(&vec![&input_tensor.shape], |x| &inner_graph[x])?;
                        let output_dtypes = enode.dtype_transfer(&vec![&input_tensor.dtype])?;
                        assert_eq!(output_shapes.len(), 1);
                        assert_eq!(output_dtypes.len(), 1);

                        let index_of_this_split_op = external_graph.insts.len();
                        let output_shape = &output_shapes[0];
                        let output_dtype = &output_dtypes[0];

                        let mut output_tensor_names = vec![];
                        for (split_index, split_order, _split_op) in split_nodes.iter() {
                            // deal with output VarFromExtern
                            let output_var = VarFromExtern {
                                shape: output_shape.clone(),
                                dtype: output_dtype.clone(),
                            };
                            let output_tensor_name =
                                format!("v{}_{}", index_of_this_split_op, split_order);
                            let _ = external_graph
                                .vars
                                .insert(output_tensor_name.clone(), output_var);
                            id_name_map.insert(Id::from(*split_index), output_tensor_name.clone());

                            output_tensor_names.push(output_tensor_name);

                            is_used[*split_index] = true;
                            used_cnt += 1;
                        }

                        assert_eq!(split_nodes.len(), output_tensor_names.len());
                        if output_tensor_names.len() == 1 {
                            let order = {
                                if split_nodes[0].1 == 0 {
                                    1usize
                                } else {
                                    0usize
                                }
                            };
                            let output_tensor_name =
                                format!("v{}_{}", index_of_this_split_op, order);
                            let output_var = VarFromExtern {
                                shape: output_shape.clone(),
                                dtype: output_dtype.clone(),
                            };
                            let _ = external_graph
                                .vars
                                .insert(output_tensor_name.clone(), output_var);
                            output_tensor_names.push(output_tensor_name);
                        }

                        output_tensor_names.sort();

                        // deal with InstFromExtern
                        let op = OperatorFromExtern::Split2 { axis: axis };
                        let inst = InstFromExtern {
                            op: op,
                            input_args: vec![input_tensor_name],
                            return_values: output_tensor_names,
                            attributes: HashMap::default(),
                        };
                        external_graph.insts.push(inst);
                    }
                    ComputExprIR::Split(_) => {
                        unimplemented!()
                    }
                    ComputExprIR::OtherOut(_) => {
                        // we process this when processing ``ComputExprIR::Other``
                        is_used[enode_index] = true;
                        used_cnt += 1;
                    }
                    ComputExprIR::Other(args) => {
                        let mut all_inputs_satisfied = true;
                        for index in 1..args.len() {
                            let arg = args[index];
                            if !is_used[usize::from(arg)] {
                                all_inputs_satisfied = false;
                                break;
                            }
                        }
                        if !all_inputs_satisfied {
                            continue;
                        }

                        let mut input_names = vec![];
                        for index in 1..args.len() {
                            let arg = args[index];
                            let name = id_name_map[&arg].as_str();
                            input_names.push(name.to_string());
                        }

                        let op_index = inner_graph[args[0]].try_parse_to_hash().expect(&format!(
                            "enode: {:?}\ngraph: {}\noriginal_graph: {}",
                            enode,
                            inner_graph.pretty_str(),
                            self.converter.as_ref().unwrap().rec_expr
                        ));
                        let info_of_other_op = &converter.info_map_of_other_op[&op_index];

                        // deal with output VarFromExtern
                        let mut out_names: Vec<String> =
                            vec!["".to_string(); info_of_other_op.outputs.len()];
                        let var_name_prefix_id = external_graph.insts.len();
                        for (other_out_enode_index, other_out_enode) in
                            inner_graph.as_ref().iter().enumerate()
                        {
                            match other_out_enode {
                                ComputExprIR::OtherOut(args) => {
                                    let other_id = args[0];
                                    if &inner_graph[other_id] != enode {
                                        continue;
                                    }
                                    let child_index =
                                        inner_graph[args[1]].try_parse_to_num().unwrap() as usize;
                                    let child_var = info_of_other_op.outputs[child_index].clone();
                                    let output_tensor_name =
                                        format!("v{}_{}", var_name_prefix_id, child_index);
                                    out_names[child_index] = output_tensor_name.clone();
                                    let _ = external_graph
                                        .vars
                                        .insert(output_tensor_name.clone(), child_var);
                                    id_name_map.insert(
                                        Id::from(other_out_enode_index),
                                        output_tensor_name,
                                    );
                                }
                                _ => {}
                            }
                        }

                        // deal with output InstFromExtern
                        let op = OperatorFromExtern::Other { op_index: op_index };
                        let inst = InstFromExtern {
                            op: op,
                            input_args: input_names,
                            return_values: out_names,
                            attributes: HashMap::default(),
                        };
                        external_graph.insts.push(inst);

                        is_used[enode_index] = true;
                        used_cnt += 1;
                    }
                    ComputExprIR::Num(_)
                    | ComputExprIR::Var(_)
                    | ComputExprIR::HashVal(_)
                    | ComputExprIR::SinkOutput(_)
                    | ComputExprIR::Shape(_) => {
                        // pass these because they are not Inst
                        is_used[enode_index] = true;
                        used_cnt += 1;
                    }
                    ComputExprIR::GraphOutput(args) => {
                        if !is_used[usize::from(args[0])] {
                            continue;
                        }
                        let tensor_id = &args[0];
                        let output_original_var_name_id = &args[1];
                        let output_original_var_name = inner_graph[*output_original_var_name_id]
                            .try_parse_to_var()
                            .unwrap();
                        let input_tensor_name = id_name_map[tensor_id].clone();

                        // deal with output InstFromExtern
                        let op = OperatorFromExtern::GraphOutput {
                            var_name: output_original_var_name.to_string(),
                        };
                        let inst = InstFromExtern {
                            op: op,
                            input_args: vec![input_tensor_name],
                            return_values: vec![],
                            attributes: HashMap::default(),
                        };
                        external_graph.insts.push(inst);

                        is_used[enode_index] = true;
                        used_cnt += 1;
                    }
                }
            }
            if is_used == is_used_old {
                // jump out the loop in case it has outliers
                if is_used
                    .iter()
                    .map(|x| if *x { 1usize } else { 0usize })
                    .sum::<usize>()
                    != is_used.len()
                {
                    error!("there are outliers in the graph: {:?}", inner_graph);
                }
                break;
            }
        }

        return Ok(external_graph);
    }
}

#[allow(unused)]
unsafe fn copy_cstring_to_ffi_chars(tmp_str: &CString, char_array: *mut ffi::c_char) {
    let length = tmp_str.as_bytes().len();

    for (index, byte) in tmp_str.as_bytes().iter().enumerate() {
        let p = char_array.add(index);
        *p = byte.clone() as ffi::c_char;
    }
    let end = char_array.add(length);
    *end = 0u8 as ffi::c_char;
}

/// remove duplicated nodes in the RecExpr.
#[allow(unused)]
fn remove_redundant_nodes(graph: &ComputationGraphFromEgg) -> ComputationGraphFromEgg {
    // because RecExpr.nodes has a requirement that all the children of a node should be in front of it,
    // therefore, we can ensure all children have been processed before the parent node.
    let mut set = IndexSet::<ComputExprIR>::default();
    let mut ids = HashMap::<Id, Id>::default();
    let mut new_node_list: Vec<ComputExprIR> = vec![];

    for (old_id, node) in graph.as_ref().iter().enumerate() {
        let old_id = Id::from(old_id);
        let new_node = node.clone().map_children(|id| {
            assert!(ids.contains_key(&id));
            ids[&id]
        });
        let (new_id, is_new_node) = set.insert_full(new_node.clone());
        ids.insert(old_id, Id::from(new_id));
        if is_new_node {
            new_node_list.push(new_node);
        }
    }

    return RecExpr::from(new_node_list);
}
