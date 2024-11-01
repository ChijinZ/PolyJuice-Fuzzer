use crate::BuildGraphError;
use egg::*;
use log::error;
use serde::{Deserialize, Serialize};
use std::cmp::max;

define_language! {
    pub enum ComputExprIR {
        "tensor"     = Tensor([Id; 4]), // tensor_name, tensor_type (e.g., Input/Constant/Placeholder), info_index, shape
        // "weight"    = Weight([Id; 1]), // takes a Var, format : name@dim1_dim2...
        "ewadd"     = Ewadd([Id; 2]),
        "ewmul"     = Ewmul([Id; 2]),
        // "smul"      = Smul([Id; 2]),
        "transpose" = Transpose([Id; 3]), // input, dim0, dim1
        "matmul"    = Matmul([Id; 2]), // input1, input2
        // "conv2d"    = Conv2d([Id; 6]), // _stride_h, _stride_w, _pad, _act, _inpt, _wght
        // "enlarge"   = Enlarge([Id; 2]), // input_to_enlarge, ref_input
        // "dropout"   = Dropout(Id),
        "relu"      = Relu(Id),
        "tanh"      = Tanh(Id),
        "sigmoid"   = Sigmoid(Id),
        "poolmax"   = Poolmax([Id; 5]), // input, kernel_h, kernel_w, stride, padding
        "poolavg"   = Poolavg([Id; 5]), // input, kernel_h, kernel_w, stride, padding
        "concat"    = Concat([Id; 3]), // input1, input2, axis
        // "concat3"    = Concat3([Id; 5]), // axis, ndim, input1, input2. input3, ndim is for using in CheckApply only
        // "concat4"    = Concat4([Id; 6]), // axis, ndim, input1, input2. input3, input4, ndim is for using in CheckApply only
        // "concat5"    = Concat5([Id; 7]), // axis, ndim, input1, input2, input3, input4, input5. ndim is for using in CheckApply only
        // Add a concat for each number of inputs if needed
        "split_out"   = SplitOut([Id; 2]), // input, index
        "split2_0"  = Split2_0([Id; 2]), // input, index
        "split2_1"  = Split2_1([Id; 2]), // input, index
        "split"     = Split([Id; 3]), // input, chunk_num, axis
        // "Cpool"     = Cpool([Id; 2]),
        // "Iconv"     = Iconv([Id; 2]),
        // "Imatmul"   = Imatmul,
        // "Iewmul"    = Iewmul,
        // "merge"     = Merge([Id; 2]), // merge_gconv, takes [weight, count]
        // "reshape"   = Reshape([Id; 2]), // input, shape_name (format: dim1_dim2...)
        // "batchnorm" = BatchNorm([Id; 5]), // input, scale, bias, mean, var
        "other_out" = OtherOut([Id; 3]), // other_id, index_id, shape
        "other" = Other(Vec<Id>), // original_id, input0, input1, ...
        "output" = GraphOutput([Id; 2]), // input, original_name_id
        "sink_output" = SinkOutput(Vec<Id>), // output node to assemble all output nodes (leaves) because egg works with single root graph
        Num(i64),
        HashVal(u64),
        Var(Symbol),
        "shape" = Shape(Vec<Id>), // each Id is a Num
        // Other_1(Id),
    }
}

impl ComputExprIR {
    pub fn try_parse_to_var(&self) -> Option<&str> {
        match self {
            ComputExprIR::Var(symbol) => Some(symbol.as_str()),
            _ => None,
        }
    }

    pub fn try_parse_to_num(&self) -> Option<i64> {
        match self {
            ComputExprIR::Num(i) => Some(*i),
            ComputExprIR::HashVal(i) => Some(*i as i64),
            _ => None,
        }
    }

    pub fn try_parse_to_hash(&self) -> Option<u64> {
        match self {
            ComputExprIR::HashVal(i) => Some(*i),
            _ => None,
        }
    }

    pub fn try_parse_to_shape<'a, F>(&self, get_node_from_id: F) -> Option<Vec<usize>>
    where
        F: Fn(&Id) -> &'a ComputExprIR,
    {
        match self {
            ComputExprIR::Shape(shape) => {
                let shape = shape
                    .iter()
                    .map(|num_id| {
                        let num_node = get_node_from_id(num_id);
                        let num = num_node.try_parse_to_num().unwrap() as usize;
                        num
                    })
                    .collect();
                Some(shape)
            }
            _ => None,
        }
    }

    // https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
    fn broadcast_shapes(
        input_shape0: &Vec<usize>,
        input_shape1: &Vec<usize>,
    ) -> Result<Vec<usize>, ()> {
        let max_dim = max(input_shape0.len(), input_shape1.len());
        let mut output_shapes = vec![0; max_dim];

        // prepend
        let mut prepended_shape0 = input_shape0.clone();
        let mut prepended_shape1 = input_shape1.clone();

        while prepended_shape0.len() != max_dim {
            prepended_shape0.insert(0, 1usize);
        }
        while prepended_shape1.len() != max_dim {
            prepended_shape1.insert(0, 1usize);
        }

        for i in 0..max_dim {
            if prepended_shape0[i] == prepended_shape1[i] {
                output_shapes[i] = prepended_shape1[i];
            } else if prepended_shape0[i] == 1 {
                output_shapes[i] = prepended_shape1[i];
            } else if prepended_shape1[i] == 1 {
                output_shapes[i] = prepended_shape0[i];
            } else {
                error!("{:?}, {:?}", input_shape0, input_shape1);
                return Err(());
            }
        }

        return Ok(output_shapes);
    }

    pub fn dtype_transfer(
        &self,
        input_dtypes: &Vec<&String>,
    ) -> Result<Vec<String>, BuildGraphError> {
        match self {
            ComputExprIR::Tensor(_) => {
                unreachable!()
            }
            ComputExprIR::Ewadd(_) => {
                assert_eq!(input_dtypes.len(), 2);
                return Ok(vec![input_dtypes[0].clone()]);
            }
            ComputExprIR::Ewmul(_) => {
                assert_eq!(input_dtypes.len(), 2);
                return Ok(vec![input_dtypes[0].clone()]);
            }
            ComputExprIR::Transpose(_) => {
                assert_eq!(input_dtypes.len(), 1);
                return Ok(vec![input_dtypes[0].clone()]);
            }
            ComputExprIR::Matmul(_) => {
                assert_eq!(input_dtypes.len(), 2);
                return Ok(vec![input_dtypes[0].clone()]);
            }
            ComputExprIR::Relu(_) => {
                assert_eq!(input_dtypes.len(), 1);
                return Ok(vec![input_dtypes[0].clone()]);
            }
            ComputExprIR::Tanh(_) => {
                assert_eq!(input_dtypes.len(), 1);
                return Ok(vec![input_dtypes[0].clone()]);
            }
            ComputExprIR::Sigmoid(_) => {
                assert_eq!(input_dtypes.len(), 1);
                return Ok(vec![input_dtypes[0].clone()]);
            }
            ComputExprIR::Poolmax(_) => {
                assert_eq!(input_dtypes.len(), 1);
                return Ok(vec![input_dtypes[0].clone()]);
            }
            ComputExprIR::Poolavg(_) => {
                assert_eq!(input_dtypes.len(), 1);
                return Ok(vec![input_dtypes[0].clone()]);
            }
            ComputExprIR::Concat(_) => {
                assert_eq!(input_dtypes.len(), 2);
                return Ok(vec![input_dtypes[0].clone()]);
            }
            ComputExprIR::SplitOut(_) => {
                unimplemented!()
            }
            ComputExprIR::Split(_) => {
                unimplemented!()
            }
            ComputExprIR::OtherOut(_) => {
                unreachable!()
            }
            ComputExprIR::Split2_0(_) | ComputExprIR::Split2_1(_) => {
                assert_eq!(input_dtypes.len(), 1);
                return Ok(vec![input_dtypes[0].clone()]);
            }
            ComputExprIR::Other(_) => {
                unreachable!()
            }
            ComputExprIR::GraphOutput(_) => {
                assert_eq!(input_dtypes.len(), 1);
                return Ok(vec![input_dtypes[0].clone()]);
            }
            ComputExprIR::Num(_)
            | ComputExprIR::Var(_)
            | ComputExprIR::HashVal(_)
            | ComputExprIR::SinkOutput(_)
            | ComputExprIR::Shape(_) => {
                unreachable!()
            }
        };
    }

    pub fn shape_transfer<'a, F>(
        &self,
        input_shapes: &Vec<&Vec<usize>>,
        get_enode_from_id: F,
        // graph_expr: &ComputationGraphFromEgg,
    ) -> Result<Vec<Vec<usize>>, BuildGraphError>
    where
        F: Fn(Id) -> &'a ComputExprIR,
    {
        match self {
            ComputExprIR::Tensor(_) => {
                unreachable!()
            }
            ComputExprIR::Ewadd(_) => {
                assert_eq!(input_shapes.len(), 2);
                let output_shape = Self::broadcast_shapes(&input_shapes[0], &input_shapes[1])
                    .map_err(|_| {
                        BuildGraphError::new(self.to_string(), "broadcast shape error".to_string())
                    })?;
                return Ok(vec![output_shape]);
            }
            ComputExprIR::Ewmul(_) => {
                assert_eq!(input_shapes.len(), 2);
                let output_shape = Self::broadcast_shapes(&input_shapes[0], &input_shapes[1])
                    .map_err(|_| {
                        BuildGraphError::new(self.to_string(), "broadcast shape error".to_string())
                    })?;
                return Ok(vec![output_shape]);
            }
            ComputExprIR::Transpose(args) => {
                assert_eq!(input_shapes.len(), 1);
                let mut output_shape = input_shapes[0].clone();
                let dim0 = get_enode_from_id(args[1]).try_parse_to_num().unwrap() as usize;
                let dim1 = get_enode_from_id(args[2]).try_parse_to_num().unwrap() as usize;
                if dim0 >= output_shape.len() || dim1 >= output_shape.len() {
                    return Err(BuildGraphError::new(
                        self.to_string(),
                        "shape transfer index error".to_string(),
                    ));
                }

                let dim_0_new_val = output_shape[dim1];
                let dim_1_new_val = output_shape[dim0];
                output_shape[dim0] = dim_0_new_val;
                output_shape[dim1] = dim_1_new_val;
                return Ok(vec![output_shape]);
            }
            ComputExprIR::Matmul(_) => {
                // similar to NNSmith's matmul type_transfer
                // https://pytorch.org/docs/stable/generated/torch.matmul.html#torch.matmul
                assert_eq!(input_shapes.len(), 2);
                let lhs = &input_shapes[0];
                let rhs = &input_shapes[1];

                let lrc = &lhs[(lhs.len() - 2)..];
                let rrc = &rhs[(rhs.len() - 2)..];
                let mut orc = vec![lrc[0], rrc[1]];

                let lbatch = lhs[0..lhs.len() - 2].to_vec();
                let rbatch = rhs[0..rhs.len() - 2].to_vec();
                let mut batches = if lbatch.len() > rbatch.len() {
                    let mut batches = lbatch[0..lbatch.len() - rbatch.len()].to_vec();
                    for (x, y) in lbatch[batches.len()..].iter().zip(rbatch.iter()) {
                        batches.push(max(*x, *y));
                    }
                    batches
                } else {
                    let mut batches = rbatch[0..rbatch.len() - lbatch.len()].to_vec();
                    for (x, y) in lbatch.iter().zip(rbatch[batches.len()..].iter()) {
                        batches.push(max(*x, *y));
                    }
                    batches
                };

                batches.append(&mut orc);

                return Ok(vec![batches]);
            }
            ComputExprIR::Relu(_) => {
                assert_eq!(input_shapes.len(), 1);
                let output_shape = input_shapes[0].clone();
                return Ok(vec![output_shape]);
            }
            ComputExprIR::Tanh(_) => {
                assert_eq!(input_shapes.len(), 1);
                let output_shape = input_shapes[0].clone();
                return Ok(vec![output_shape]);
            }
            ComputExprIR::Sigmoid(_) => {
                assert_eq!(input_shapes.len(), 1);
                let output_shape = input_shapes[0].clone();
                return Ok(vec![output_shape]);
            }
            ComputExprIR::Poolmax(args) => {
                assert_eq!(input_shapes.len(), 1);
                let kh = get_enode_from_id(args[1]).try_parse_to_num().unwrap() as usize;
                let kw = get_enode_from_id(args[2]).try_parse_to_num().unwrap() as usize;
                let stride = get_enode_from_id(args[3]).try_parse_to_num().unwrap() as usize;
                let padding = get_enode_from_id(args[4]).try_parse_to_num().unwrap() as usize;

                let input_shape = &input_shapes[0];
                let mut output_shape = vec![input_shape[0], input_shape[1]];

                output_shape.push((((input_shape[2] - kh) + 2 * padding) / stride) + 1);

                output_shape.push((((input_shape[3] - kw) + 2 * padding) / stride) + 1);

                return Ok(vec![output_shape]);
            }
            ComputExprIR::Poolavg(args) => {
                assert_eq!(input_shapes.len(), 1);
                let kh = get_enode_from_id(args[1]).try_parse_to_num().unwrap() as usize;
                let kw = get_enode_from_id(args[2]).try_parse_to_num().unwrap() as usize;
                let stride = get_enode_from_id(args[3]).try_parse_to_num().unwrap() as usize;
                let padding = get_enode_from_id(args[4]).try_parse_to_num().unwrap() as usize;

                let input_shape = &input_shapes[0];
                let mut output_shape = vec![input_shape[0], input_shape[1]];

                output_shape.push((((input_shape[2] - kh) + 2 * padding) / stride) + 1);

                output_shape.push((((input_shape[3] - kw) + 2 * padding) / stride) + 1);

                return Ok(vec![output_shape]);
            }
            ComputExprIR::Concat(args) => {
                assert_eq!(input_shapes.len(), 2);
                let axis = get_enode_from_id(args[2]).try_parse_to_num().unwrap() as usize;
                let input_0_shape = &input_shapes[0];
                let input_1_shape = &input_shapes[1];
                if axis >= input_0_shape.len() || axis >= input_1_shape.len() {
                    return Err(BuildGraphError::new(
                        self.to_string(),
                        "shape transfer index error".to_string(),
                    ));
                }
                let mut output_shape = (*input_0_shape).clone();
                output_shape[axis] = input_0_shape[axis] + input_1_shape[axis];

                return Ok(vec![output_shape]);
            }
            ComputExprIR::SplitOut(_) => {
                unimplemented!()
            }
            ComputExprIR::Split2_0(args) | ComputExprIR::Split2_1(args) => {
                assert_eq!(input_shapes.len(), 1);
                let axis = get_enode_from_id(args[1]).try_parse_to_num().unwrap() as usize;
                let input_shape = input_shapes[0];
                if input_shape.len() <= axis {
                    return Err(BuildGraphError::new(
                        "split".to_string(),
                        "input_shape.len() <= axis during shape transfer".to_string(),
                    ));
                }
                if input_shape[axis] % 2 != 0 {
                    return Err(BuildGraphError::new(
                        "split".to_string(),
                        "input_shape[axis] % 2 != 0 during shape transfer".to_string(),
                    ));
                }
                let mut output_shape = input_shape.clone();
                output_shape[axis] = output_shape[axis] / 2;
                return Ok(vec![output_shape]);
            }
            ComputExprIR::Split(_) => {
                unimplemented!()
            }
            ComputExprIR::OtherOut(_) => {
                unreachable!()
            }
            ComputExprIR::Other(_) => {
                unreachable!()
            }
            ComputExprIR::GraphOutput(_) => {
                assert_eq!(input_shapes.len(), 1);
                let output_shape = input_shapes[0].clone();
                return Ok(vec![output_shape]);
            }
            ComputExprIR::Num(_)
            | ComputExprIR::Var(_)
            | ComputExprIR::HashVal(_)
            | ComputExprIR::SinkOutput(_)
            | ComputExprIR::Shape(_) => {
                unreachable!()
            }
        }
    }
}

pub trait TensorInfoForAnalysis {
    fn shape(&self) -> Vec<usize>;
}

#[derive(Default, Debug, Clone)]
pub struct AnalysisOfCEIR {}

#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InfoAttachedWithEClass {
    shape: Vec<usize>, // shape is vec![] means that the eclass has no actual output tensor
}

impl Analysis<ComputExprIR> for AnalysisOfCEIR {
    type Data = InfoAttachedWithEClass;
    fn make(egraph: &EGraph<ComputExprIR, Self>, enode: &ComputExprIR) -> Self::Data {
        let get_attached_info = |i: &Id| egraph[*i].data.clone();
        match enode {
            ComputExprIR::Tensor([_tensor_name, _tensor_type, _info_index, shape_id]) => {
                let shape_eclass = &egraph[*shape_id];
                assert_eq!(shape_eclass.len(), 1);
                let shape_enode = &shape_eclass.nodes[0];
                let shape = shape_enode
                    .try_parse_to_shape(|x| {
                        let eclass = &egraph[*x];
                        assert_eq!(eclass.len(), 1);
                        &eclass.nodes[0]
                    })
                    .expect("this is not a shape node");
                InfoAttachedWithEClass { shape }
            }
            ComputExprIR::Ewadd([a, b])
            | ComputExprIR::Ewmul([a, b])
            | ComputExprIR::Matmul([a, b]) => {
                let a_shape = get_attached_info(a).shape;
                let b_shape = get_attached_info(b).shape;
                let input_shapes = vec![&a_shape, &b_shape];
                let output_shapes = enode
                    .shape_transfer(&input_shapes, |id| {
                        let eclass = &egraph[id];
                        assert_eq!(eclass.len(), 1);
                        &eclass.nodes[0]
                    })
                    .unwrap();
                assert_eq!(output_shapes.len(), 1);
                InfoAttachedWithEClass {
                    shape: output_shapes[0].clone(),
                }
            }
            ComputExprIR::Transpose([input, _dim0, _dim1]) => {
                let input_shape = get_attached_info(input).shape;
                let input_shapes = vec![&input_shape];
                let output_shapes = enode
                    .shape_transfer(&input_shapes, |id| {
                        let eclass = &egraph[id];
                        assert_eq!(eclass.len(), 1);
                        &eclass.nodes[0]
                    })
                    .unwrap();
                assert_eq!(output_shapes.len(), 1);
                InfoAttachedWithEClass {
                    shape: output_shapes[0].clone(),
                }
            }
            ComputExprIR::Relu(input)
            | ComputExprIR::Tanh(input)
            | ComputExprIR::Sigmoid(input) => {
                let input_shape = get_attached_info(input).shape;
                let input_shapes = vec![&input_shape];
                let output_shapes = enode
                    .shape_transfer(&input_shapes, |id| {
                        let eclass = &egraph[id];
                        assert_eq!(eclass.len(), 1);
                        &eclass.nodes[0]
                    })
                    .unwrap();
                assert_eq!(output_shapes.len(), 1);
                InfoAttachedWithEClass {
                    shape: output_shapes[0].clone(),
                }
            }
            ComputExprIR::Poolavg([input, _kh, _kw, _stride, _padding])
            | ComputExprIR::Poolmax([input, _kh, _kw, _stride, _padding]) => {
                let input_shape = get_attached_info(input).shape;
                let input_shapes = vec![&input_shape];
                let output_shapes = enode
                    .shape_transfer(&input_shapes, |id| {
                        let eclass = &egraph[id];
                        assert_eq!(eclass.len(), 1);
                        &eclass.nodes[0]
                    })
                    .unwrap();
                assert_eq!(output_shapes.len(), 1);
                InfoAttachedWithEClass {
                    shape: output_shapes[0].clone(),
                }
            }
            ComputExprIR::Concat([input1, input2, _axis]) => {
                let input1_shape = get_attached_info(input1).shape;
                let input2_shape = get_attached_info(input2).shape;
                let input_shapes = vec![&input1_shape, &input2_shape];
                let output_shapes = enode
                    .shape_transfer(&input_shapes, |id| {
                        let eclass = &egraph[id];
                        assert_eq!(eclass.len(), 1);
                        &eclass.nodes[0]
                    })
                    .unwrap();
                assert_eq!(output_shapes.len(), 1);
                InfoAttachedWithEClass {
                    shape: output_shapes[0].clone(),
                }
            }
            ComputExprIR::Split2_0([input, _axis]) | ComputExprIR::Split2_1([input, _axis]) => {
                let input_shape = get_attached_info(input).shape;
                let input_shapes = vec![&input_shape];
                let output_shapes = enode
                    .shape_transfer(&input_shapes, |id| {
                        let eclass = &egraph[id];
                        assert_eq!(eclass.len(), 1);
                        &eclass.nodes[0]
                    })
                    .unwrap();
                assert_eq!(output_shapes.len(), 1);
                InfoAttachedWithEClass {
                    shape: output_shapes[0].clone(),
                }
            }
            ComputExprIR::Split(_) | ComputExprIR::SplitOut(_) => {
                unimplemented!()
            }
            ComputExprIR::OtherOut([_other_id, _index_id, shape_id]) => {
                let shape_eclass = &egraph[*shape_id];
                assert_eq!(shape_eclass.len(), 1);
                let shape_enode = &shape_eclass.nodes[0];
                let shape = shape_enode
                    .try_parse_to_shape(|x| {
                        let eclass = &egraph[*x];
                        assert_eq!(eclass.len(), 1);
                        &eclass.nodes[0]
                    })
                    .expect("this is not a shape node");
                InfoAttachedWithEClass { shape }
            }
            ComputExprIR::Other(_)
            | ComputExprIR::GraphOutput(_)
            | ComputExprIR::SinkOutput(_)
            | ComputExprIR::Num(_)
            | ComputExprIR::HashVal(_)
            | ComputExprIR::Var(_)
            | ComputExprIR::Shape(_) => InfoAttachedWithEClass { shape: vec![] },
        }
    }
    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        assert_eq!(a.shape, b.shape, "cannot merge: \na:{:?}\nb:{:?}", a, b);
        return DidMerge(false, false);
    }
}

impl TensorInfoForAnalysis for InfoAttachedWithEClass {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}
