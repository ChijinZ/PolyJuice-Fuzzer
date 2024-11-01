use crate::model::*;
use crate::VarFromExtern;
use egg::*;
use std::collections::{HashMap, HashSet};

/// Struct for converting a model specified using our Rust interface to RecExpr
///
/// The RecExpr is growed on the fly when member functions are called. Uses a
/// Hashmap to store the map of scalar nodes and symbol nodes to their indices into the RexExpr to
/// avoid replication.
#[derive(Default, Debug, Clone)]
pub struct GraphConverter {
    pub rec_expr: RecExpr<ComputExprIR>,
    pub scalar_map: HashMap<i64, Id>,
    pub symbol_map: HashMap<String, Id>,
    pub shape_map: HashMap<Vec<usize>, Id>,
    pub tensor_name_map: HashMap<String, TensorInfo>,
    pub operator_map: HashMap<Id, OperatorInfo>,
    pub info_map_of_other_op: HashMap<u64, InfoAttachedWithOtherOp>,
    pub root: Option<Id>,
}

#[derive(Default, Debug, Clone)]
pub struct InfoAttachedWithOtherOp {
    pub info_index: u64,
    pub outputs: Vec<VarFromExtern>,
}

/// Struct for storing information of a tensor.
#[derive(Clone, Default, Debug)]
pub struct TensorInfo {
    pub id: Id,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub name: String,
}

/// Struct for storing information of an operator.
#[derive(Default, Debug, Clone)]
pub struct OperatorInfo {
    pub output_id: Id,
    pub input_ids: Vec<Id>,
    pub attributes: Option<HashMap<String, String>>,
}

impl GraphConverter {
    #[allow(unused)]
    /// Gets the RexExpr after graph is constructed
    pub fn rec_expr(&self) -> &RecExpr<ComputExprIR> {
        &self.rec_expr
    }

    pub fn get_tensor_info_by_name(&self, name: &str) -> &TensorInfo {
        &self.tensor_name_map[name]
    }

    fn add_new_tensor(&mut self, node_id: Id, tensor_name: &str, var: &VarFromExtern) {
        let tensor_info = TensorInfo {
            id: node_id.clone(),
            shape: var.shape.clone(),
            dtype: var.dtype.clone(),
            name: tensor_name.to_string(),
        };

        let _ = self
            .tensor_name_map
            .insert(tensor_name.to_string(), tensor_info);
    }

    fn add_new_operator(
        &mut self,
        op_node_id: Id,
        inputs: Vec<&TensorInfo>,
        op_attributes: Option<&HashMap<String, String>>,
    ) {
        let attributes = match op_attributes {
            Some(a) => Some(a.clone()),
            None => None,
        };
        let operator = OperatorInfo {
            output_id: op_node_id,
            input_ids: inputs.iter().map(|x| x.id).collect(),
            attributes: attributes,
        };
        let _ = self.operator_map.insert(op_node_id, operator);
    }

    /// Takes in the parameters for the new input, construct the node in RexExpr,
    /// return the Id (index) of this input node in the RecExpr. This is the
    /// pattern for all these op functions.
    pub fn new_input(
        &mut self,
        name: &str,
        tensor_type: &str,
        var: &VarFromExtern,
        info_index: u64,
    ) {
        let name_id = self.add_or_get_symbol(name);
        let tensor_type_id = self.add_or_get_symbol(tensor_type);
        let info_index_node_id = self.add_or_get_val(info_index as i64);
        let shape_id = self.add_or_get_shape(&var.shape);

        let new_node =
            ComputExprIR::Tensor([name_id, tensor_type_id, info_index_node_id, shape_id]);
        let node_id = self.rec_expr.add(new_node);

        self.add_new_tensor(node_id, name, var);
    }

    // pub fn conv2d(
    //     &mut self,
    //     inpt: TensorInfo,
    //     wght: TensorInfo,
    //     stride_h: i64,
    //     stride_w: i64,
    //     padding: i64,
    //     activation: i64,
    // ) -> TensorInfo {
    //     let stride_h_id = self.add_or_get_val(stride_h);
    //     let stride_w_id = self.add_or_get_val(stride_w);
    //     let padding_id = self.add_or_get_val(padding);
    //     let activation_id = self.add_or_get_val(activation);
    //
    //     let new_node = Mdl::Conv2d([
    //         stride_h_id,
    //         stride_w_id,
    //         padding_id,
    //         activation_id,
    //         inpt.id,
    //         wght.id,
    //     ]);
    //
    //     // Get shape
    //     let mut shape = [0; MAX_DIM];
    //     let input_h = inpt.shape[2];
    //     let input_w = inpt.shape[3];
    //     let kernel_h = wght.shape[2];
    //     let kernel_w = wght.shape[3];
    //
    //     let (output_h, output_w) = self.get_conv_shape(
    //         input_h, input_w, stride_h, stride_w, kernel_h, kernel_w, padding,
    //     );
    //     shape[0] = inpt.shape[0];
    //     shape[1] = wght.shape[0];
    //     shape[2] = output_h;
    //     shape[3] = output_w;
    //
    //     TensorInfo {
    //         id: self.rec_expr.add(new_node),
    //         shape: shape,
    //         n_dim: 4,
    //         dtype: inpt.dtype.clone(),
    //     }
    // }

    pub fn relu(&mut self, input: &TensorInfo, output_name: &str, output: &VarFromExtern) {
        let new_node = ComputExprIR::Relu(input.id);
        let node_id = self.rec_expr.add(new_node);

        self.add_new_operator(node_id, vec![input], None);
        self.add_new_tensor(node_id, output_name, output);
    }

    pub fn tanh(&mut self, input: &TensorInfo, output_name: &str, output: &VarFromExtern) {
        let new_node = ComputExprIR::Tanh(input.id);
        let node_id = self.rec_expr.add(new_node);

        self.add_new_operator(node_id, vec![input], None);
        self.add_new_tensor(node_id, output_name, output);
    }

    pub fn sigmoid(&mut self, input: &TensorInfo, output_name: &str, output: &VarFromExtern) {
        let new_node = ComputExprIR::Sigmoid(input.id);
        let node_id = self.rec_expr.add(new_node);

        self.add_new_operator(node_id, vec![input], None);
        self.add_new_tensor(node_id, output_name, output);
    }

    // pub fn batchnorm(&mut self, inpt: TensorInfo, scale: TensorInfo, bias: TensorInfo, mean: TensorInfo, var: TensorInfo) -> TensorInfo {
    //     let new_node = Mdl::BatchNorm([inpt.id, scale.id, bias.id, mean.id, var.id]);
    //     TensorInfo {
    //         id: self.rec_expr.add(new_node),
    //         ..inpt
    //     }
    // }

    pub fn add(
        &mut self,
        inpt_1: &TensorInfo,
        inpt_2: &TensorInfo,
        output_name: &str,
        output: &VarFromExtern,
    ) {
        assert_eq!(inpt_1.dtype, inpt_2.dtype);
        let new_node = ComputExprIR::Ewadd([inpt_1.id, inpt_2.id]);
        let node_id = self.rec_expr.add(new_node);

        self.add_new_operator(node_id, vec![inpt_1, inpt_2], None);
        self.add_new_tensor(node_id, output_name, output);
    }

    pub fn matmul(
        &mut self,
        inpt_1: &TensorInfo,
        inpt_2: &TensorInfo,
        output_name: &str,
        output: &VarFromExtern,
    ) {
        assert_eq!(inpt_1.dtype, inpt_2.dtype);
        let new_node = ComputExprIR::Matmul([inpt_1.id, inpt_2.id]);
        let node_id = self.rec_expr.add(new_node);

        self.add_new_operator(node_id, vec![inpt_1, inpt_2], None);
        self.add_new_tensor(node_id, output_name, output);
    }

    pub fn mul(
        &mut self,
        inpt_1: &TensorInfo,
        inpt_2: &TensorInfo,
        output_name: &str,
        output: &VarFromExtern,
    ) {
        assert_eq!(inpt_1.dtype, inpt_2.dtype);
        let new_node = ComputExprIR::Ewmul([inpt_1.id, inpt_2.id]);
        let node_id = self.rec_expr.add(new_node);

        self.add_new_operator(node_id, vec![inpt_1, inpt_2], None);
        self.add_new_tensor(node_id, output_name, output);
    }

    pub fn concat(
        &mut self,
        inpt_1: &TensorInfo,
        inpt_2: &TensorInfo,
        axis: i64,
        output_name: &str,
        output: &VarFromExtern,
    ) {
        assert_eq!(inpt_1.dtype, inpt_2.dtype);

        let axis_id = self.add_or_get_val(axis);

        let new_node = ComputExprIR::Concat([inpt_1.id, inpt_2.id, axis_id]);
        let node_id = self.rec_expr.add(new_node);

        self.add_new_operator(node_id, vec![inpt_1, inpt_2], None);
        self.add_new_tensor(node_id, output_name, output);
    }

    // pub fn concat_multi(&mut self, axis: i64, inputs: &[TensorInfo]) -> TensorInfo {
    //     let n_inputs = inputs.len();
    //     // We can add supports for other number of inputs later when needed.
    //     // We need to add a new Concat op for each number of inputs
    //     assert!(n_inputs <= 5);
    //
    //     let n_dim = inputs[0].n_dim;
    //     let axis_id = self.add_or_get_val(axis);
    //     let ndim_id = self.add_or_get_val(n_dim as i64);
    //
    //     let new_node = match n_inputs {
    //         2 => {
    //             Mdl::Concat([
    //                 axis_id,
    //                 ndim_id,
    //                 inputs[0].id,
    //                 inputs[1].id,
    //             ])
    //         }
    //         3 => {
    //             Mdl::Concat3([
    //                 axis_id,
    //                 ndim_id,
    //                 inputs[0].id,
    //                 inputs[1].id,
    //                 inputs[2].id,
    //             ])
    //         }
    //         4 => {
    //             Mdl::Concat4([
    //                 axis_id,
    //                 ndim_id,
    //                 inputs[0].id,
    //                 inputs[1].id,
    //                 inputs[2].id,
    //                 inputs[3].id,
    //             ])
    //         }
    //         5 => {
    //             Mdl::Concat5([
    //                 axis_id,
    //                 ndim_id,
    //                 inputs[0].id,
    //                 inputs[1].id,
    //                 inputs[2].id,
    //                 inputs[3].id,
    //                 inputs[4].id,
    //             ])
    //         }
    //         _ => panic!("Number of input for concat not supported"),
    //     };
    //
    //     let mut shape = inputs[0].shape;
    //     shape[axis as usize] += (1..n_inputs)
    //         .map(|i| inputs[i].shape[axis as usize])
    //         .sum::<i64>();
    //
    //     TensorInfo {
    //         id: self.rec_expr.add(new_node),
    //         shape,
    //         n_dim,
    //     }
    // }

    pub fn maxpool2d(
        &mut self,
        inpt: &TensorInfo,
        kernel_h: i64,
        kernel_w: i64,
        stride: i64,
        padding: i64,
        output_name: &str,
        output: &VarFromExtern,
    ) {
        let kernel_h_id = self.add_or_get_val(kernel_h);
        let kernel_w_id = self.add_or_get_val(kernel_w);
        let stride_id = self.add_or_get_val(stride);
        let padding_id = self.add_or_get_val(padding);

        let new_node =
            ComputExprIR::Poolmax([inpt.id, kernel_h_id, kernel_w_id, stride_id, padding_id]);
        let node_id = self.rec_expr.add(new_node);

        self.add_new_operator(node_id, vec![inpt], None);
        self.add_new_tensor(node_id, output_name, output);
    }

    pub fn avgpool2d(
        &mut self,
        inpt: &TensorInfo,
        kernel_h: i64,
        kernel_w: i64,
        stride: i64,
        padding: i64,
        output_name: &str,
        output: &VarFromExtern,
    ) {
        let kernel_h_id = self.add_or_get_val(kernel_h);
        let kernel_w_id = self.add_or_get_val(kernel_w);
        let stride_id = self.add_or_get_val(stride);
        let padding_id = self.add_or_get_val(padding);

        let new_node =
            ComputExprIR::Poolavg([inpt.id, kernel_h_id, kernel_w_id, stride_id, padding_id]);
        let node_id = self.rec_expr.add(new_node);

        self.add_new_operator(node_id, vec![inpt], None);
        self.add_new_tensor(node_id, output_name, output);
    }

    // pub fn enlarge(&mut self, inpt_1: TensorInfo, inpt_2: TensorInfo) -> TensorInfo {
    //     let mut shape = inpt_1.shape;
    //     shape[2] = inpt_2.shape[2];
    //     shape[3] = inpt_2.shape[3];
    //
    //     let new_node = Mdl::Enlarge([inpt_1.id, inpt_2.id]);
    //
    //     TensorInfo {
    //         id: self.rec_expr.add(new_node),
    //         shape: shape,
    //         n_dim: 4,
    //     }
    // }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split
    // https://pytorch.org/docs/stable/generated/torch.chunk.html
    // https://www.tensorflow.org/api_docs/python/tf/split
    pub fn split(
        &mut self,
        inpt: &TensorInfo,
        chunk_num: i64,
        axis: i64,
        output_names: Vec<&str>,
        output_vars: Vec<&VarFromExtern>,
    ) {
        assert_eq!(output_names.len(), output_vars.len());

        let chunk_num_id = self.add_or_get_val(chunk_num);
        let axis_id = self.add_or_get_val(axis);

        let split_node = ComputExprIR::Split([inpt.id, chunk_num_id, axis_id]);
        let split_node_id = self.rec_expr.add(split_node);

        self.add_new_operator(split_node_id, vec![inpt], None);

        for index in 0..output_names.len() {
            let output_name = output_names[index];
            let output_var = output_vars[index];
            let index_id = self.add_or_get_val(index as i64);

            let output_node = ComputExprIR::SplitOut([split_node_id, index_id]);
            let output_id = self.rec_expr.add(output_node);

            self.add_new_tensor(output_id, output_name, output_var);
        }
    }

    // pub fn reshape(&mut self, inpt: TensorInfo, shape: &[i64]) -> TensorInfo {
    //     let shape_name = &shape.iter().join("_");
    //     let node = Mdl::Var(Symbol::from(shape_name));
    //     let shape_name_id = self.rec_expr.add(node);
    //
    //     let new_node = Mdl::Reshape([inpt.id, shape_name_id]);
    //     let (shape_new, n_dim) = self.shape_from_dim(shape);
    //     TensorInfo {
    //         id: self.rec_expr.add(new_node),
    //         shape: shape_new,
    //         n_dim: n_dim,
    //     }
    // }

    pub fn transpose(
        &mut self,
        inpt: &TensorInfo,
        dim0: i64,
        dim1: i64,
        output_name: &str,
        output_var: &VarFromExtern,
    ) {
        let dim0_id = self.add_or_get_val(dim0);
        let dim1_id = self.add_or_get_val(dim1);

        let new_node = ComputExprIR::Transpose([inpt.id, dim0_id, dim1_id]);
        let node_id = self.rec_expr.add(new_node);

        self.add_new_operator(node_id, vec![inpt], None);
        self.add_new_tensor(node_id, output_name, output_var);
    }

    pub fn split2(
        &mut self,
        inpt: &TensorInfo,
        axis: i64,
        output_name_0: &str,
        output_var_0: &VarFromExtern,
        output_name_1: &str,
        output_var_1: &VarFromExtern,
    ) {
        let axis_id = self.add_or_get_val(axis);

        let new_node_0 = ComputExprIR::Split2_0([inpt.id, axis_id]);
        let new_node_0_id = self.rec_expr.add(new_node_0);
        self.add_new_operator(new_node_0_id, vec![inpt], None);
        self.add_new_tensor(new_node_0_id, output_name_0, output_var_0);

        let new_node_1 = ComputExprIR::Split2_1([inpt.id, axis_id]);
        let new_node_1_id = self.rec_expr.add(new_node_1);
        self.add_new_operator(new_node_1_id, vec![inpt], None);
        self.add_new_tensor(new_node_1_id, output_name_1, output_var_1);
    }

    // pub fn noop(&mut self, inpt_1: TensorInfo, inpt_2: TensorInfo) -> TensorInfo {
    //     let new_node = Mdl::Noop([inpt_1.id, inpt_2.id]);
    //     TensorInfo {
    //         id: self.rec_expr.add(new_node),
    //         shape: [0; MAX_DIM],
    //         n_dim: inpt_1.n_dim,
    //     }
    // }

    /// If a scalar value is in the RecExpr, gets the Id. Otherwise creates one.
    fn add_or_get_val(&mut self, val: i64) -> Id {
        match self.scalar_map.get(&val) {
            Some(id) => *id,
            None => {
                let node = ComputExprIR::Num(val);
                let id = self.rec_expr.add(node);
                self.scalar_map.insert(val, id);
                id
            }
        }
    }

    /// If a symbol is in the RecExpr, gets the Id. Otherwise creates one.
    fn add_or_get_symbol(&mut self, symbol_str: &str) -> Id {
        match self.symbol_map.get(symbol_str) {
            Some(id) => *id,
            None => {
                let node = ComputExprIR::Var(Symbol::from(symbol_str));
                let id = self.rec_expr.add(node);
                self.symbol_map.insert(symbol_str.to_string(), id);
                id
            }
        }
    }

    fn add_or_get_shape(&mut self, shape: &[usize]) -> Id {
        match self.shape_map.get(shape) {
            Some(id) => *id,
            None => {
                let shape_vec = shape
                    .iter()
                    .map(|x| {
                        let x = *x as i64;
                        let num_id = self.add_or_get_val(x);
                        num_id
                    })
                    .collect();

                let node = ComputExprIR::Shape(shape_vec);
                let id = self.rec_expr.add(node);
                self.shape_map.insert(shape.to_vec(), id);
                id
            }
        }
    }

    pub fn other(
        &mut self,
        info_index: u64,
        inputs: Vec<&TensorInfo>,
        output_names: Vec<&str>,
        output_vars: Vec<&VarFromExtern>,
    ) {
        assert_eq!(output_names.len(), output_vars.len());

        let mut input_ids = vec![];

        let info_index_node = ComputExprIR::HashVal(info_index);
        let info_index_id = self.rec_expr.add(info_index_node);
        input_ids.push(info_index_id);

        for input_tensor in inputs.iter() {
            input_ids.push(input_tensor.id);
        }

        let node = ComputExprIR::Other(input_ids);
        let other_node_id = self.rec_expr.add(node);

        self.add_new_operator(other_node_id, inputs, None);

        // let id_of_other_node_id = self.add_or_get_val(usize::from(other_node_id) as i64);

        let mut output_tensors = vec![];
        for index in 0..output_names.len() {
            let output_name = output_names[index];
            let output_var = output_vars[index];
            let index_id = self.add_or_get_val(index as i64);
            let shape_id = self.add_or_get_shape(&output_var.shape);

            let output_node = ComputExprIR::OtherOut([other_node_id, index_id, shape_id]);
            let output_id = self.rec_expr.add(output_node);

            self.add_new_tensor(output_id, output_name, output_var);
            output_tensors.push(output_var.clone());
        }

        let info_of_other_op = InfoAttachedWithOtherOp {
            info_index: info_index,
            outputs: output_tensors,
        };
        let _ = self
            .info_map_of_other_op
            .insert(info_index, info_of_other_op);
    }

    pub fn graph_output(&mut self, tensor: &TensorInfo) {
        let tensor_name = &tensor.name;
        let tensor_name_id = self.add_or_get_symbol(tensor_name);
        let graph_output_node = ComputExprIR::GraphOutput([tensor.id.clone(), tensor_name_id]);
        let _graph_output_id = self.rec_expr.add(graph_output_node);
    }

    pub fn assemble_outputs(&mut self) {
        if self.root.is_some() {
            return;
        }

        let mut used_enode: HashSet<Id> = HashSet::default();
        let mut all_enode: HashSet<Id> = HashSet::default();
        for (index, enode) in self.rec_expr.as_ref().iter().enumerate() {
            let node_id = Id::from(index);
            let _ = all_enode.insert(node_id);
            for child in enode.children() {
                let _ = used_enode.insert(child.clone());
            }
        }
        assert!(
            all_enode.is_superset(&used_enode),
            "used_enode: {:?};\nall_enode: {:?}",
            used_enode,
            all_enode
        );
        let leaves: Vec<Id> = all_enode.difference(&used_enode).map(|x| *x).collect();
        for id in leaves.iter() {
            let enode = &self.rec_expr[*id];
            match enode {
                ComputExprIR::GraphOutput(_args) => {
                    // println!("rust: {:?}", self.rec_expr[args[0]]);
                }
                _ => {
                    unreachable!("error enode: {:?}", enode);
                }
            }
        }
        let assemble_output = ComputExprIR::SinkOutput(leaves);
        let root = self.rec_expr.add(assemble_output);
        self.root = Some(root);
    }

    #[allow(unused)]
    pub fn root(&self) -> Option<Id> {
        self.root.clone()
    }
}
